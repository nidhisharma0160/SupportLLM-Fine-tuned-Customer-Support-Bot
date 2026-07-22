import os
import sys
import torch
import asyncio
import uuid
from typing import AsyncGenerator, Dict, List, Optional

# Try importing vLLM
try:
    import vllm
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.sampling_params import SamplingParams
    from vllm.lora.request import LoRARequest
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class BaseServingEngine:
    async def generate(self, prompt: str, max_tokens: int = 50, temperature: float = 0.0) -> AsyncGenerator[str, None]:
        raise NotImplementedError

class VLLMServingEngine(BaseServingEngine):
    def __init__(self, model_id: str, adapter_path: Optional[str] = None):
        print(f"Initializing vLLM Engine with model {model_id}...")
        engine_args = AsyncEngineArgs(
            model=model_id,
            enable_lora=True if adapter_path else False,
            max_loras=1 if adapter_path else 0,
            # Adjust memory usage for smaller GPUs if needed
            gpu_memory_utilization=0.8,
            trust_remote_code=True,
            # Run on CPU if CUDA is not available
            device="cpu" if not torch.cuda.is_available() else "cuda"
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.adapter_path = adapter_path
        self.lora_request = None
        if adapter_path:
            self.lora_request = LoRARequest("support_lora", 1, lora_path=adapter_path)

    async def generate(self, prompt: str, max_tokens: int = 50, temperature: float = 0.0) -> AsyncGenerator[str, None]:
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            pad_token_id=151643 # Qwen pad token
        )
        request_id = str(uuid.uuid4())
        
        results_generator = self.engine.generate(
            prompt,
            sampling_params,
            request_id,
            lora_request=self.lora_request
        )
        
        last_text = ""
        async for request_output in results_generator:
            text = request_output.outputs[0].text
            delta = text[len(last_text):]
            last_text = text
            yield delta

class HFServingEngine(BaseServingEngine):
    """
    Fallback serving engine using Hugging Face Transformers.
    Implements a background dynamic batching queue to simulate continuous batching on CPU/MPS.
    """
    def __init__(self, model_id: str, adapter_path: Optional[str] = None):
        print(f"Initializing Hugging Face Serving Engine with model {model_id}...")
        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.float16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.dtype = torch.float16
        else:
            self.device = "cpu"
            self.dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        print(f"Loading base model to {self.device}...")
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=self.dtype).to(self.device)
        
        if adapter_path and os.path.exists(adapter_path):
            print(f"Loading LoRA adapter from {adapter_path}...")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            
        self.queue = asyncio.Queue()
        self.batch_size = 16
        self.accumulate_time_seconds = 0.02 # 20ms accumulation window
        self.loop_task = asyncio.create_task(self._batch_loop())

    async def _batch_loop(self):
        while True:
            # Wait for at least one item
            req = await self.queue.get()
            batch = [req]
            
            # Try to accumulate more items up to batch_size
            start_time = asyncio.get_event_loop().time()
            while len(batch) < self.batch_size:
                elapsed = asyncio.get_event_loop().time() - start_time
                remaining = self.accumulate_time_seconds - elapsed
                if remaining <= 0:
                    break
                try:
                    # Non-blocking get or wait briefly
                    next_req = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                    batch.append(next_req)
                except asyncio.TimeoutError:
                    break
            
            # Process the batch
            try:
                await self._process_batch(batch)
            except Exception as e:
                print(f"Error in batch processing: {e}")
                for req in batch:
                    if not req["future"].done():
                        req["future"].set_exception(e)
            finally:
                for _ in range(len(batch)):
                    self.queue.task_done()

    async def _process_batch(self, batch: List[Dict]):
        prompts = [req["prompt"] for req in batch]
        max_tokens = max(req["max_tokens"] for req in batch)
        
        # We run generation in a thread pool to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        outputs = await loop.run_in_executor(None, self._generate_sync, prompts, max_tokens)
        
        # Distribute results
        for req, output_text in zip(batch, outputs):
            if not req["future"].done():
                req["future"].set_result(output_text)

    def _generate_sync(self, prompts: List[str], max_tokens: int) -> List[str]:
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        results = []
        input_length = inputs["input_ids"].shape[1]
        for i in range(len(prompts)):
            gen_tokens = outputs[i][input_length:]
            decoded = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            results.append(decoded)
        return results

    async def generate(self, prompt: str, max_tokens: int = 50, temperature: float = 0.0) -> AsyncGenerator[str, None]:
        future = asyncio.get_event_loop().create_future()
        self.queue.put_nowait({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "future": future
        })
        
        # Wait for the batch execution to finish
        result_text = await future
        
        # Yield the generated text in one go or chunked to simulate stream
        chunk_size = 5
        for i in range(0, len(result_text), chunk_size):
            yield result_text[i:i+chunk_size]
            await asyncio.sleep(0.001)

def get_serving_engine(model_id: str, adapter_path: Optional[str] = None) -> BaseServingEngine:
    if HAS_VLLM:
        try:
            return VLLMServingEngine(model_id, adapter_path)
        except Exception as e:
            print(f"Failed to start vLLM engine: {e}. Falling back to HF Serving Engine.")
    return HFServingEngine(model_id, adapter_path)
