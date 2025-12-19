import os
import json
import time
import random
import argparse
import numpy as np
import torch
import psutil
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoTokenizer, 
    AutoConfig, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments, 
    EarlyStoppingCallback
)

# --- Utils & Setup ---

def print_system_status():
    """Sanity check for memory usage."""
    mem_avail = round(psutil.virtual_memory().available / 1e9, 2)
    print(f"System Check | Available Memory: {mem_avail} GB")

def set_seed(seed=42):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- Data Handling ---

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
            "labels": self.label_map[item["label"]]
        }

def get_class_weights(data):
    """Compute inverse class weights to handle imbalance."""
    labels = [0 if x["label"] == "yes" else 1 for x in data]
    weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
    weights_tensor = torch.tensor(weights, dtype=torch.float)
    print(f"Computed Class Weights: {weights_tensor}")
    return weights_tensor

def oversample_minority(data, label_key="label", minority_val="no", ratio=1.0):
    """Simple random oversampling for the minority class."""
    majority = [x for x in data if x[label_key] != minority_val]
    minority = [x for x in data if x[label_key] == minority_val]
    
    n_target = int(len(majority) * ratio)
    if len(minority) >= n_target:
        return majority + minority

    # Randomly sample with replacement
    additional = [random.choice(minority) for _ in range(n_target - len(minority))]
    return majority + minority + additional

def load_and_clean_data(train_path, val_path):
    """
    Parses raw JSON, filters extrinsic info, and separates train/val based on doc IDs.
    Hardcoded logic: excludes 'Extrinsic Information' errors from unfaithful set.
    """
    with open(os.path.expanduser(train_path)) as f:
        full_raw_data = json.load(f)
    
    with open(os.path.expanduser(val_path)) as f:
        val_reference_data = json.load(f)

    # Map sentences to doc IDs (cleaning duplicates)
    faithful_map = {}
    unfaithful_map = {}

    for x in val_reference_data:
        label = x["faithfulness_label"].strip().lower()
        doc_id = x["document_id"]
        
        for sentence in x["docSentText"]:
            sent = sentence.strip()
            if label == "yes":
                faithful_map[sent] = doc_id
            else:
                # Specific logic: Only treat as unfaithful if NOT extrinsic info
                if x.get("type") != "Extrinsic Information":
                    unfaithful_map[sent] = doc_id

    # Dedup logic: remove sentences that appear in both maps (conflict resolution)
    unfaithful_sents = set(unfaithful_map.keys())
    filtered_faithful_sents = set(faithful_map.keys()) - unfaithful_sents

    clean_data = []
    for sent in filtered_faithful_sents:
        clean_data.append({"sentence": sent, "label": "yes", "document_id": faithful_map[sent]})
    for sent in unfaithful_sents:
        clean_data.append({"sentence": sent, "label": "no", "document_id": unfaithful_map[sent]})

    # Split based on specific val document IDs
    val_doc_ids = {'CNN-201245', 'AlamedaCC_09192017_2017-4642', 'CNN-229050', 'DenverCityCouncil_08292016_16-0553'}
    
    val_set = [x for x in clean_data if x["document_id"] in val_doc_ids]
    # Filter original training set to exclude val docs
    train_set = [x for x in full_raw_data if x["document_id"] not in val_doc_ids]

    return train_set, val_set

# --- Training Pipeline ---

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    bal_acc = balanced_accuracy_score(labels, preds)
    
    print(f"Prediction distribution: {Counter(preds)}")
    return {
        "eval_accuracy": acc,
        "eval_precision": precision,
        "eval_recall": recall,
        "eval_f1": f1,
        "eval_balanced_accuracy": bal_acc
    }

def train_model(train_data, eval_data, model_name, output_dir, epochs=6, lr=1e-5, wd=0.01):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = 2
    
    # ModernBERT/RoBERTa specific config tweaks if needed
    if "ModernBERT" in model_name:
        config._attn_implementation = "eager"

    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    # Freeze encoder, train classifier only
    for param in model.base_model.parameters():
        param.requires_grad = False

    weights = get_class_weights(train_data)
    train_ds = FaithfulnessDataset(train_data, tokenizer)
    eval_ds = FaithfulnessDataset(eval_data, tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        weight_decay=wd,
        load_best_model_at_end=True,
        metric_for_best_model="eval_balanced_accuracy",
        greater_is_better=True,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_total_limit=2,
    )

    # if want to use class weights in loss function (here since already balanced, no affect)
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = CrossEntropyLoss(weight=weights.to(model.device))
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    print(f"Starting training for {model_name}...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

# --- Main Execution ---

if __name__ == "__main__":
    print_system_status()
    set_seed(42)

    # Paths
    # TODO: Move these to CLI args if deployment needed
    # train: raw data with extrinsic info dropped, could be changed to include extrinsic if desired
    RAW_TRAIN_PATH = "~/faithfulness/data/sentence_labeled_no_cleaning_drop_extrinsic_only.json"
    # val data
    CLEAN_VAL_PATH = "~/faithfulness/data/cleaned.json"
    OUTPUT_BASE = os.path.expanduser("~/scratch/faithfulness_results/")

    # Load & Prep
    raw_train, val_data = load_and_clean_data(RAW_TRAIN_PATH, CLEAN_VAL_PATH)
    
    # Balance training set
    train_balanced = oversample_minority(raw_train, ratio=1.0)
    random.shuffle(train_balanced)

    print(f"Train size (balanced): {len(train_balanced)}")
    print(f"Val size: {len(val_data)}")
    print("Val Label Dist:", Counter(x["label"] for x in val_data))

    # Run Training
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    model_name = "roberta-base" # or "answerdotai/ModernBERT-base"
    save_path = os.path.join(OUTPUT_BASE, f"faithful_{model_name.split('/')[-1]}_{timestamp}")

    train_model(
        train_balanced, 
        val_data, 
        model_name=model_name, 
        output_dir=save_path,
        learning_rate=5e-5,
        weight_decay=0.001,
        epochs=8
    )