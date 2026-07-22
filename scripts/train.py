import os
import sys
import yaml
import time
import json
import torch
import mlflow
import argparse
from datetime import datetime
from sklearn.metrics import f1_score

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import load_and_split_dataset, format_instruction_prompt

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from peft import LoraConfig, get_peft_model, TaskType

def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clean_prediction(pred_text, intents):
    pred = pred_text.strip().lower()
    for intent in intents:
        if intent.lower() == pred:
            return intent
    for intent in intents:
        if intent.lower() in pred:
            return intent
    return intents[0]

def evaluate_predictions(model, tokenizer, df, intents, device, batch_size=8):
    model.eval()
    prompts = [format_instruction_prompt(row["instruction"], intents) for _, row in df.iterrows()]
    y_true = df["intent"].tolist()
    y_pred = []
    
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=15,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
        input_length = inputs["input_ids"].shape[1]
        for j in range(len(batch_prompts)):
            gen_tokens = outputs[j][input_length:]
            decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            cleaned = clean_prediction(decoded, intents)
            y_pred.append(cleaned)
            
    tokenizer.padding_side = orig_padding_side
    
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    accuracy = sum(1 for gt, pr in zip(y_true, y_pred) if gt == pr) / len(y_true)
    
    return macro_f1, weighted_f1, accuracy, y_pred

def prepare_dataset_for_sft(df, tokenizer, intents, max_seq_length):
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    for _, row in df.iterrows():
        prompt = format_instruction_prompt(row["instruction"], intents)
        target = f" {row['intent']}<|im_end|>"
        
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        
        # Combine input and set loss mask
        combined_input_ids = prompt_ids + target_ids
        combined_labels = [-100] * len(prompt_ids) + target_ids
        
        # Truncate if too long
        if len(combined_input_ids) > max_seq_length:
            combined_input_ids = combined_input_ids[:max_seq_length]
            combined_labels = combined_labels[:max_seq_length]
            
        # Pad to max_seq_length
        padding_length = max_seq_length - len(combined_input_ids)
        if padding_length > 0:
            combined_input_ids = combined_input_ids + [pad_token_id] * padding_length
            combined_labels = combined_labels + [-100] * padding_length
            
        attention_mask = [1] * (max_seq_length - padding_length) + [0] * padding_length
        if padding_length < 0:
            attention_mask = [1] * max_seq_length
            
        input_ids_list.append(combined_input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(combined_labels)
        
    class SFTDataset(torch.utils.data.Dataset):
        def __init__(self, input_ids, attention_mask, labels):
            self.input_ids = input_ids
            self.attention_mask = attention_mask
            self.labels = labels
            
        def __len__(self):
            return len(self.input_ids)
            
        def __getitem__(self, idx):
            return {
                "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
                "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long)
            }
            
    return SFTDataset(input_ids_list, attention_mask_list, labels_list)

def run_training(config_override=None, is_sweep=False):
    with open("configs/train.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    if config_override:
        config.update(config_override)
        
    set_seed(config["seed"])
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        device = "cuda"
    elif torch.backends.mps.is_available():
        gpu_name = "Apple Silicon MPS"
        device = "mps"
    else:
        gpu_name = "CPU Only"
        device = "cpu"
        
    print(f"Device: {device}")
    print(f"Hardware: {gpu_name}")
    
    if not is_sweep:
        run_config = {
            "model_id": config["model_id"],
            "device": device,
            "hardware": gpu_name,
            "timestamp": datetime.now().isoformat(),
            "is_sweep": is_sweep
        }
        os.makedirs("results", exist_ok=True)
        with open("results/run_config.json", "w") as f:
            json.dump(run_config, f, indent=4)

    model_id = config["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    if device == "cuda":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    else:
        torch_dtype = torch.float16 if device == "mps" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype
        )
        model.to(device)

    train_df, val_df, _, intents = load_and_split_dataset(
        train_size_limit=2000,
        seed=config["seed"]
    )
    
    if is_sweep and len(val_df) > 50:
        val_df = val_df.sample(n=50, random_state=config["seed"])
        
    train_dataset = prepare_dataset_for_sft(train_df, tokenizer, intents, config["max_seq_length"])
    val_dataset = prepare_dataset_for_sft(val_df, tokenizer, intents, config["max_seq_length"])
    
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    mlflow.set_experiment("SupportLLM-IntentClassification")
    while mlflow.active_run() is not None:
        mlflow.end_run()
        
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if is_sweep:
        run_name = f"sweep_trial"
        
    start_time = time.time()
    
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(config)
        mlflow.log_param("device", device)
        mlflow.log_param("gpu_name", gpu_name)
        
        training_args = TrainingArguments(
            output_dir="./tmp_results",
            learning_rate=config["learning_rate"],
            num_train_epochs=config.get("num_train_epochs", 1) if "max_steps" not in config else -1,
            max_steps=config.get("max_steps", -1),
            per_device_train_batch_size=config["per_device_train_batch_size"],
            per_device_eval_batch_size=config["per_device_eval_batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            weight_decay=config["weight_decay"],
            warmup_ratio=config["warmup_ratio"],
            logging_steps=config["logging_steps"],
            eval_strategy=config["eval_strategy"],
            eval_steps=config["eval_steps"],
            save_strategy=config["save_strategy"],
            seed=config["seed"],
            fp16=(device == "mps" or (device == "cuda" and not torch.cuda.is_bf16_supported())),
            bf16=(device == "cuda" and torch.cuda.is_bf16_supported()),
            report_to="none",
            dataloader_pin_memory=False if device == "cpu" else True
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=default_data_collator # Use default stack collator
        )
        
        trainer.train()
        wall_clock_time = time.time() - start_time
        
        eval_metrics = trainer.evaluate()
        val_loss = eval_metrics.get("eval_loss", 999.0)
        
        val_macro_f1, val_weighted_f1, val_acc, _ = evaluate_predictions(
            model, tokenizer, val_df, intents, device, batch_size=8
        )
        
        mlflow.log_metric("train_runtime_seconds", wall_clock_time)
        mlflow.log_metric("eval_loss", val_loss)
        mlflow.log_metric("val_macro_f1", val_macro_f1)
        mlflow.log_metric("val_weighted_f1", val_weighted_f1)
        mlflow.log_metric("val_accuracy", val_acc)
            
        print(f"Wall-clock training time: {wall_clock_time:.2f} seconds")
        print(f"Val Loss: {val_loss:.4f} | Val Macro-F1: {val_macro_f1:.4f} | Val Acc: {val_acc:.4f}")
        
        if not is_sweep:
            adapter_path = "model_assets/adapter/"
            os.makedirs(adapter_path, exist_ok=True)
            model.save_pretrained(adapter_path)
            tokenizer.save_pretrained(adapter_path)
            print(f"Adapter saved to {adapter_path}")
            
            mlflow.log_artifacts(adapter_path, artifact_path="adapter")
            
            with open("results/run_config.json", "r") as f:
                run_config = json.load(f)
            run_config["train_time_seconds"] = wall_clock_time
            with open("results/run_config.json", "w") as f:
                json.dump(run_config, f, indent=4)
                
    return val_macro_f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    run_training(config_override=config)
