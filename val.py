import os
import json
import torch
import argparse
import numpy as np
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, balanced_accuracy_score
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer
)

# --- Configuration for Batch Experiments ---
BATCH_EXPERIMENTS = [
    {
        "model_path": "~/scratch/faithful_raw_roberta-base_star_20250613-125332",
        "test_sets": [
            ("No drop", "~/data/test_no_cleaning_no_drop.json"),
            ("No Cleaning drop only extrinsic", "~/data/test_no_cleaning_only_drop_extrinsic.json"),
        ]
    },
    {
        "model_path": "~/scratch/faithful_raw_drop_only_extrinsic_roberta-base_star_20250626-141350",
        "test_sets": [
            ("No drop", "~/data/test_no_cleaning_no_drop.json"),
            ("No Cleaning drop only extrinsic", "~/data/test_no_cleaning_only_drop_extrinsic.json"),
        ]
    }
]

class FaithfulnessDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = {"yes": 0, "no": 1}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer(
            item["sentence"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": self.label_map.get(item["label"], -1) # Handle potential missing labels safely
        }

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    acc = accuracy_score(labels, preds)
    bal_acc = balanced_accuracy_score(labels, preds)

    # Print distribution to detect mode collapse (e.g., predicting all 0s)
    dist = Counter(preds)
    print(f"    Pred Distribution: {dict(dist)}")
    
    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def evaluate_single_pair(model_path, data_path, description="Custom Test"):
    """
    Loads one model and evaluates it on one dataset.
    """
    model_path = os.path.expanduser(model_path)
    data_path = os.path.expanduser(data_path)

    print(f"\n" + "="*50)
    print(f"EVALUATING: {description}")
    print(f"Model: {model_path}")
    print(f"Data:  {data_path}")
    print("="*50)

    # 1. Load Data
    if not os.path.exists(data_path):
        print(f"[Error] File not found: {data_path}")
        return

    with open(data_path) as f:
        raw_data = json.load(f)
    
    print(f"  Loaded {len(raw_data)} examples.")
    print(f"  Label Dist: {Counter(x['label'] for x in raw_data)}")

    # 2. Load Model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to("cuda")
    except Exception as e:
        print(f"[Error] Failed to load model: {e}")
        return

    # 3. Evaluate
    dataset = FaithfulnessDataset(raw_data, tokenizer)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    results = trainer.evaluate(dataset)
    
    print(f"\n  >>> RESULT [{description}]:")
    print(f"  Balanced Accuracy: {results['eval_balanced_accuracy']:.4f}")
    print(f"  F1 Score:          {results['eval_f1']:.4f}")
    return results

def run_batch_experiments():
    """Runs the predefined list of experiments from BATCH_EXPERIMENTS."""
    print(">>> Starting Batch Experiment Mode...")
    for exp_config in BATCH_EXPERIMENTS:
        model_path = exp_config["model_path"]
        for desc, test_file in exp_config["test_sets"]:
            evaluate_single_pair(model_path, test_file, description=desc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Faithfulness Classifiers")
    
    # Optional arguments for single-run mode
    parser.add_argument("--model", type=str, help="Path to the model checkpoint")
    parser.add_argument("--data", type=str, help="Path to the test JSON file")
    
    # Flag for batch mode
    parser.add_argument("--batch", action="store_true", help="Run all experiments defined in BATCH_EXPERIMENTS")

    args = parser.parse_args()

    if args.batch:
        run_batch_experiments()
    elif args.model and args.data:
        evaluate_single_pair(args.model, args.data)
    else:
        print("Usage Error: Please provide either (--model AND --data) for a single run, or (--batch) for bulk experiments.")
        print("\nExample Single Run:")
        print("  python evaluate_classifier.py --model ~/saved_models/roberta --data ~/data/test.json")
        print("\nExample Batch Run:")
        print("  python evaluate_classifier.py --batch")