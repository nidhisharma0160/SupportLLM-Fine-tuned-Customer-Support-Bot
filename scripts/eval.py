import os
import sys
import yaml
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from rouge_score import rouge_scorer

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import load_and_split_dataset, format_instruction_prompt

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def clean_prediction(pred_text, intents):
    pred = pred_text.strip().lower()
    for intent in intents:
        if intent.lower() == pred:
            return intent
    for intent in intents:
        if intent.lower() in pred:
            return intent
    return intents[0]

def evaluate_classification(model, tokenizer, df, intents, device, batch_size=8):
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
            y_pred.append(clean_prediction(decoded, intents))
            
    tokenizer.padding_side = orig_padding_side
    
    # Compute classification report
    report = classification_report(y_true, y_pred, labels=intents, output_dict=True, zero_division=0)
    
    # Accuracy
    accuracy = sum(1 for gt, pr in zip(y_true, y_pred) if gt == pr) / len(y_true)
    
    return accuracy, report, y_pred

def evaluate_responses(model, tokenizer, df, device, batch_size=4):
    """
    Generate responses using instruction + intent and evaluate using ROUGE-L.
    """
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    prompts = []
    for _, row in df.iterrows():
        p = f"<|im_start|>system\nYou are a helpful customer support agent. Answer the user query based on the intent '{row['intent']}'.<|im_end|>\n<|im_start|>user\n{row['instruction']}<|im_end|>\n<|im_start|>assistant\n"
        prompts.append(p)
        
    y_true_responses = df["response"].tolist()
    y_pred_responses = []
    
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
                max_new_tokens=60, # Responses are typically short sentences
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
        input_length = inputs["input_ids"].shape[1]
        for j in range(len(batch_prompts)):
            gen_tokens = outputs[j][input_length:]
            decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            y_pred_responses.append(decoded)
            
    tokenizer.padding_side = orig_padding_side
    
    # Compute ROUGE-L
    rouge_l_scores = []
    for gt, pred in zip(y_true_responses, y_pred_responses):
        score = scorer.score(gt, pred)
        rouge_l_scores.append(score['rougeL'].fmeasure)
        
    mean_rouge_l = np.mean(rouge_l_scores)
    return mean_rouge_l

def save_confusion_matrix(y_true, y_pred, intents, filename="results/confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred, labels=intents)
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Show only labels that are actually present or subset for readability if too large
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=intents)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation="vertical")
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def main():
    # Load configuration
    config_path = "configs/best.yaml" if os.path.exists("configs/best.yaml") else "configs/train.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    model_id = config["model_id"]
    
    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = torch.float16
    else:
        device = "cpu"
        torch_dtype = torch.float32
        
    print(f"Evaluating model: {model_id} on {device}")
    
    # Load dataset
    _, _, test_df, intents = load_and_split_dataset(seed=config["seed"])
    
    # We can use a subset of test set if running on CPU/MPS to speed things up
    if device in ["cpu", "mps"] and len(test_df) > 100:
        print(f"Sub-sampling test set to 100 examples for fast evaluation on {device}...")
        test_df = test_df.sample(n=100, random_state=config["seed"])
        
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # --- Part 1: Zero-shot Baseline ---
    print("\n--- Running Zero-Shot Baseline ---")
    base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype).to(device)
    
    base_acc, base_report, _ = evaluate_classification(base_model, tokenizer, test_df, intents, device)
    base_rouge_l = evaluate_responses(base_model, tokenizer, test_df, device)
    
    print(f"Baseline Accuracy: {base_acc:.4f}")
    print(f"Baseline Macro-F1: {base_report['macro avg']['f1-score']:.4f}")
    print(f"Baseline ROUGE-L: {base_rouge_l:.4f}")
    
    # --- Part 2: Fine-Tuned Model ---
    print("\n--- Running Fine-Tuned Model ---")
    adapter_path = "model_assets/adapter/"
    if not os.path.exists(adapter_path):
        print(f"Error: Adapter path {adapter_path} not found. Please train the model first.")
        sys.exit(1)
        
    # Load adapter
    ft_model = PeftModel.from_pretrained(base_model, adapter_path)
    
    ft_acc, ft_report, ft_predictions = evaluate_classification(ft_model, tokenizer, test_df, intents, device)
    ft_rouge_l = evaluate_responses(ft_model, tokenizer, test_df, device)
    
    print(f"Fine-tuned Accuracy: {ft_acc:.4f}")
    print(f"Fine-tuned Macro-F1: {ft_report['macro avg']['f1-score']:.4f}")
    print(f"Fine-tuned ROUGE-L: {ft_rouge_l:.4f}")
    
    # Format per-intent F1 comparison
    per_intent_f1 = {}
    for intent in intents:
        per_intent_f1[intent] = {
            "baseline": base_report.get(intent, {}).get("f1-score", 0.0),
            "fine_tuned": ft_report.get(intent, {}).get("f1-score", 0.0)
        }
        
    # Output metrics.json
    metrics = {
        "dataset_name": "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        "model_id": model_id,
        "test_set_size": len(test_df),
        "baseline": {
            "accuracy": base_acc,
            "macro_f1": base_report["macro avg"]["f1-score"],
            "weighted_f1": base_report["weighted avg"]["f1-score"],
            "rouge_l": base_rouge_l
        },
        "fine_tuned": {
            "accuracy": ft_acc,
            "macro_f1": ft_report["macro avg"]["f1-score"],
            "weighted_f1": ft_report["weighted avg"]["f1-score"],
            "rouge_l": ft_rouge_l
        },
        "per_intent_f1": per_intent_f1
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Saved metrics to results/metrics.json")
    
    # Plot confusion matrix for fine-tuned predictions
    save_confusion_matrix(test_df["intent"].tolist(), ft_predictions, intents)
    print("Saved confusion matrix plot to results/confusion_matrix.png")

if __name__ == "__main__":
    main()
