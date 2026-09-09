"""
Fine-tune DistilBERT-multilingual-cased on collapsed GoEmotions for multi-label
emotion classification (7 classes).

Optimized for GTX 1650 (4GB VRAM):
  - batch_size=16, max_seq_length=128, fp16=True
  - ~2.5-3GB VRAM usage

Usage:
  python train.py                          # Train with default settings
  python train.py --epochs 2 --batch_size 8  # Custom settings
  python train.py --use_augmented          # Use multilingual augmented data
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, classification_report

from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


# ─────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────

class EmotionDataset(Dataset):
    """PyTorch dataset for multi-label emotion classification."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

TARGET_CLASSES = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]


def compute_metrics(eval_pred):
    """Compute macro/micro F1 for multi-label classification."""
    logits, labels = eval_pred
    # Apply sigmoid to get probabilities
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)

    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)

    # Per-class F1
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
    metrics = {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
    }
    for i, cls in enumerate(TARGET_CLASSES):
        metrics[f"f1_{cls}"] = per_class_f1[i]

    return metrics


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

def load_data(data_dir: Path, use_augmented: bool = False):
    """Load collapsed (or augmented) training, validation, and test sets."""

    if use_augmented:
        train_path = data_dir / "train_multilingual_augmented.csv"
        if not train_path.exists():
            print(f"WARNING: Augmented data not found at {train_path}")
            print("Falling back to train_collapsed.csv")
            train_path = data_dir / "train_collapsed.csv"
    else:
        train_path = data_dir / "train_collapsed.csv"

    val_path = data_dir / "val_collapsed.csv"
    test_path = data_dir / "test_collapsed.csv"

    for path in [train_path, val_path, test_path]:
        if not path.exists():
            print(f"ERROR: {path} not found. Run data/scripts/build_splits.py first.")
            sys.exit(1)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print(f"Data loaded:")
    print(f"  Train: {len(train_df):,} rows")
    print(f"  Val:   {len(val_df):,} rows")
    print(f"  Test:  {len(test_df):,} rows")

    return train_df, val_df, test_df


def prepare_datasets(train_df, val_df, test_df, tokenizer, max_length=128):
    """Tokenize and prepare PyTorch datasets."""

    def tokenize(df):
        texts = df["text"].fillna("").tolist()
        labels = df[TARGET_CLASSES].values.tolist()
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return EmotionDataset(encodings, labels)

    return tokenize(train_df), tokenize(val_df), tokenize(test_df)


# ─────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────

def train(args):
    """Run the training pipeline."""

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "processed"
    model_dir = project_root / "models" / "emotion-distilbert-multi"

    print("=" * 60)
    print("MoodMax Emotion Model Training")
    print("=" * 60)

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("WARNING: No GPU detected. Training will be slow on CPU.")

    # Load data
    print("\nLoading data...")
    train_df, val_df, test_df = load_data(data_dir, args.use_augmented)

    # Load tokenizer and model
    model_name = "distilbert-base-multilingual-cased"
    print(f"\nLoading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(TARGET_CLASSES),
        problem_type="multi_label_classification",
    )

    # Prepare datasets
    print("Tokenizing...")
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        train_df, val_df, test_df, tokenizer, args.max_seq_length
    )

    # Training arguments (GTX 1650-optimized)
    training_args = TrainingArguments(
        output_dir=str(model_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        fp16=(device == "cuda"),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_dir=str(model_dir / "logs"),
        logging_steps=100,
        report_to="none",  # Set to "wandb" if you want W&B tracking
        dataloader_num_workers=0,  # Windows compatibility
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    print(f"\nStarting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max seq length: {args.max_seq_length}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  FP16: {device == 'cuda'}")
    print()

    trainer.train()

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    print("\nTest Results:")
    for key, value in sorted(test_results.items()):
        if key.startswith("eval_"):
            print(f"  {key}: {value:.4f}")

    # Save model
    print(f"\nSaving model to {model_dir}...")
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    # Save label map alongside model
    label_map = {cls: i for i, cls in enumerate(TARGET_CLASSES)}
    with open(model_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    # Save training config
    config = {
        "model_name": model_name,
        "target_classes": TARGET_CLASSES,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_seq_length": args.max_seq_length,
        "learning_rate": args.learning_rate,
        "use_augmented": args.use_augmented,
        "test_macro_f1": test_results.get("eval_macro_f1", 0),
        "test_micro_f1": test_results.get("eval_micro_f1", 0),
    }
    with open(model_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\n✓ Training complete!")
    print(f"  Model saved to: {model_dir}")
    print(f"  Macro-F1: {test_results.get('eval_macro_f1', 0):.4f}")
    print(f"  Micro-F1: {test_results.get('eval_micro_f1', 0):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train MoodMax emotion model")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--gradient_accumulation", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--use_augmented", action="store_true",
                        help="Use multilingual augmented training data")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
