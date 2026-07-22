import os
import json
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Hugging Face Dataset path
DATASET_PATH = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

def load_and_split_dataset(train_size_limit=12000, val_pct=0.10, test_pct=0.10, seed=42):
    """
    Loads the Hugging Face dataset, performs stratified splits over the 'intent' field,
    and caps the training set size, saving counts to results/dataset_stats.json.
    """
    print(f"Loading dataset: {DATASET_PATH}...")
    dataset = load_dataset(DATASET_PATH, split="train")
    df = dataset.to_pandas()
    
    # Check shape and columns
    print(f"Loaded {len(df)} raw examples.")
    required_cols = ["instruction", "category", "intent", "response"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
            
    # Intent distribution
    intent_counts = df["intent"].value_counts().to_dict()
    intents = sorted(list(df["intent"].unique()))
    print(f"Found {len(intents)} unique intents.")

    # Stratified split: we first split off the test set
    # Then split the remainder into train/val
    temp_df, test_df = train_test_split(
        df,
        test_size=test_pct,
        random_state=seed,
        stratify=df["intent"]
    )
    
    val_relative_pct = val_pct / (1.0 - test_pct)
    train_df, val_df = train_test_split(
        temp_df,
        test_size=val_relative_pct,
        random_state=seed,
        stratify=temp_df["intent"]
    )
    
    # Cap train size via stratified sampling
    if len(train_df) > train_size_limit:
        print(f"Capping training set from {len(train_df)} to {train_size_limit} via stratified sampling...")
        train_df, _ = train_test_split(
            train_df,
            train_size=train_size_limit,
            random_state=seed,
            stratify=train_df["intent"]
        )
        
    print(f"Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Save dataset stats
    stats = {
        "total_raw": len(df),
        "num_intents": len(intents),
        "intents": intents,
        "split_sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df)
        },
        "intent_counts_raw": {k: int(v) for k, v in intent_counts.items()},
        "intent_counts_train": {k: int(v) for k, v in train_df["intent"].value_counts().to_dict().items()},
        "intent_counts_val": {k: int(v) for k, v in val_df["intent"].value_counts().to_dict().items()},
        "intent_counts_test": {k: int(v) for k, v in test_df["intent"].value_counts().to_dict().items()}
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=4)
    print("Saved dataset stats to results/dataset_stats.json")
    
    return train_df, val_df, test_df, intents

def format_instruction_prompt(instruction, intent_list=None):
    """
    Format each instruction into a prompt for classification.
    We constrain the model to output only the exact intent label.
    """
    if intent_list:
        intents_str = ", ".join(intent_list)
        system_prompt = f"You are a customer service assistant. Classify the user query into exactly one of these intents: {intents_str}. Output only the exact intent name and nothing else."
    else:
        system_prompt = "You are a customer service assistant. Classify the user query into the correct intent. Output only the exact intent name and nothing else."
        
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\nQuery: {instruction}<|im_end|>\n<|im_start|>assistant\nintent:"
    return prompt

def format_response_prompt(instruction, intent, response):
    """
    If we also do joint classification + response generation, or just response generation.
    """
    prompt = f"<|im_start|>system\nYou are a helpful customer support agent. Answer the user query based on the intent '{intent}'.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}"
    return prompt
