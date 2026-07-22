import os
import sys
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    # Determine base model from config
    config_path = "configs/train.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {"model_id": "Qwen/Qwen2.5-0.5B-Instruct"}
        
    model_id = config["model_id"]
    dataset_name = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
    
    print(f"Pre-downloading dataset '{dataset_name}'...")
    dataset = load_dataset(dataset_name)
    print(f"Dataset downloaded. Size: {len(dataset['train'])} rows.")
    
    print(f"Pre-downloading base model tokenizer and weights for '{model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # We load to CPU to cache the files on disk (in HF home cache directory)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    print(f"Model and tokenizer downloaded and cached successfully!")

if __name__ == "__main__":
    main()
