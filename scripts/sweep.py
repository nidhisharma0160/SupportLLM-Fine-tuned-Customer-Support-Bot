import os
import sys
import yaml
import optuna
import pandas as pd
import mlflow
from datetime import datetime

# Add src and scripts to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from train import run_training

def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True)
    lora_r = trial.suggest_categorical("lora_r", [8, 16, 32])
    lora_alpha = trial.suggest_categorical("lora_alpha", [16, 32, 64])
    lora_dropout = trial.suggest_float("lora_dropout", 0.01, 0.10)
    
    # Override for swift trials on CPU/MPS
    override = {
        "learning_rate": learning_rate,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "max_steps": 3,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 2,
        "eval_strategy": "no",
        "save_strategy": "no"
    }
    
    try:
        val_macro_f1 = run_training(config_override=override, is_sweep=True)
    except Exception as e:
        print(f"Trial failed with exception: {e}")
        val_macro_f1 = 0.0
        
    return val_macro_f1

def main():
    print("Starting hyperparameter sweep using Optuna (20 trials, 3 steps each)...")
    mlflow.set_experiment("SupportLLM-IntentClassification")
    
    study = optuna.create_study(direction="maximize")
    
    # Run 20 trials
    n_trials = 20
    study.optimize(objective, n_trials=n_trials)
    
    print("Sweep complete!")
    print(f"Best trial value (Val Macro-F1): {study.best_trial.value}")
    print("Best trial params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")
        
    # Write best config
    with open("configs/train.yaml", "r") as f:
        best_config = yaml.safe_load(f)
        
    best_config.update(study.best_trial.params)
    
    # Ensure best config has standard settings for final training run (10 steps)
    best_config["max_steps"] = 10
    best_config["eval_strategy"] = "no"
    best_config["save_strategy"] = "no"
    
    os.makedirs("configs", exist_ok=True)
    with open("configs/best.yaml", "w") as f:
        yaml.dump(best_config, f, default_flow_style=False)
    print("Saved best config to configs/best.yaml")
    
    # Write all trials to results/optuna_trials.csv
    trials_df = study.trials_dataframe()
    os.makedirs("results", exist_ok=True)
    trials_df.to_csv("results/optuna_trials.csv", index=False)
    print("Saved trial table to results/optuna_trials.csv")
    
    # Start a run just to log the final sweep findings
    with mlflow.start_run(run_name="sweep_summary") as run:
        mlflow.log_params(study.best_trial.params)
        mlflow.log_metric("best_val_macro_f1", study.best_trial.value)
        mlflow.log_artifact("results/optuna_trials.csv")
        mlflow.log_artifact("configs/best.yaml")

if __name__ == "__main__":
    main()
