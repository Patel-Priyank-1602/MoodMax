"""
Fine-tune DistilBERT-multilingual-cased on collapsed GoEmotions for multi-label
emotion classification (7 classes).

Optimized for GTX 1650 (4GB VRAM):
  - batch_size=16 (or 8 with gradient_accumulation=2), max_seq_length=128, fp16=True
  - Memory safe: ~2.2-2.8GB VRAM usage
  - Early stopping with patience=2 on validation Macro-F1
  - Class-weighted BCEWithLogitsLoss (inverse frequency or sqrt-inverse frequency)
  - Detailed per-class Precision / Recall / F1 logging per epoch

Usage:
  python train.py                                    # Default training (8 epochs, early stopping, class weights)
  python train.py --epochs 8 --batch_size 16         # Standard GTX 1650 config
  python train.py --batch_size 8 --gradient_accumulation 2  # Ultra-safe 4GB VRAM config
  python train.py --class_weighting sqrt_inverse     # Sqrt-smoothed inverse frequency weights
  python train.py --use_augmented                    # Use multilingual augmented data
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Mitigate CUDA memory fragmentation on 4GB GPUs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)


TARGET_CLASSES = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]


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
# Class Weights Computation
# ─────────────────────────────────────────────

def compute_class_weights(train_df: pd.DataFrame, mode: str = "sqrt_inverse", max_weight: float = 25.0) -> torch.Tensor:
    """
    Compute positive class weights for multi-label BCEWithLogitsLoss.
    pos_weight_c = (N_total - N_pos_c) / N_pos_c
    """
    total = len(train_df)
    weights = []
    print("\n" + "=" * 70)
    print(f"CLASS DISTRIBUTION & BCE POS_WEIGHTS (mode={mode})")
    print("=" * 70)
    print(f"  {'Class':<12} {'Positive':>10} {'Negative':>10} {'Pos Ratio':>12} {'Pos Weight':>12}")
    print("  " + "-" * 62)

    for cls in TARGET_CLASSES:
        pos = int(train_df[cls].sum())
        neg = total - pos
        ratio = neg / max(pos, 1)

        if mode == "inverse":
            w = ratio
        elif mode == "sqrt_inverse":
            w = np.sqrt(ratio)
        else:
            w = 1.0

        if max_weight is not None and max_weight > 0:
            w = min(w, max_weight)

        weights.append(w)
        pct = (pos / total) * 100
        print(f"  {cls:<12} {pos:>10,d} {neg:>10,d} {pct:>11.2f}% {w:>12.2f}")

    print("=" * 70 + "\n")
    return torch.tensor(weights, dtype=torch.float)


# ─────────────────────────────────────────────
# Custom Weighted Trainer
# ─────────────────────────────────────────────

class WeightedTrainer(Trainer):
    """Custom Trainer implementing per-class pos_weight in BCEWithLogitsLoss."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.class_weights.to(logits.device))
        else:
            loss_fct = torch.nn.BCEWithLogitsLoss()

        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ─────────────────────────────────────────────
# Metrics with Per-Class F1 / Precision / Recall
# ─────────────────────────────────────────────

def compute_metrics(eval_pred):
    """Compute per-class and global metrics using threshold=0.5."""
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)

    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    macro_prec = precision_score(labels, preds, average="macro", zero_division=0)
    macro_rec = recall_score(labels, preds, average="macro", zero_division=0)

    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
    per_class_prec = precision_score(labels, preds, average=None, zero_division=0)
    per_class_rec = recall_score(labels, preds, average=None, zero_division=0)

    print("\n" + "=" * 68)
    print("EPOCH EVALUATION BREAKDOWN (Threshold = 0.5)")
    print("=" * 68)
    print(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("  " + "-" * 64)

    metrics = {
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
    }

    for i, cls in enumerate(TARGET_CLASSES):
        support = int(labels[:, i].sum())
        print(f"  {cls:<12} {per_class_prec[i]:>10.4f} {per_class_rec[i]:>10.4f} "
              f"{per_class_f1[i]:>10.4f} {support:>10}")
        metrics[f"f1_{cls}"] = float(per_class_f1[i])
        metrics[f"precision_{cls}"] = float(per_class_prec[i])
        metrics[f"recall_{cls}"] = float(per_class_rec[i])

    print("=" * 68)
    print(f"  Macro-F1: {macro_f1:.4f} | Macro-Precision: {macro_prec:.4f} | Macro-Recall: {macro_rec:.4f}\n")

    return metrics


# ─────────────────────────────────────────────
# Data Loading & Tokenization
# ─────────────────────────────────────────────

def load_data(data_dir: Path, use_augmented: bool = False):
    """Load collapsed (or augmented) training, validation, and test sets."""
    if use_augmented:
        train_path = data_dir / "train_multilingual_augmented.csv"
        if not train_path.exists():
            print(f"WARNING: Augmented data not found at {train_path}. Falling back to train_collapsed.csv")
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
# Main Training Pipeline
# ─────────────────────────────────────────────

def train(args):
    """Run the training pipeline with early stopping and class weights."""
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "processed"
    model_dir = project_root / "models" / "emotion-distilbert-multi"

    print("=" * 70)
    print("MoodMax Emotion Model Training")
    print("=" * 70)

    # Hardware check & VRAM safety warnings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

        if gpu_mem <= 4.5:
            print("  [INFO] Detected 4GB VRAM GPU (e.g. GTX 1650). Applying safety limits:")
            if args.batch_size > 16:
                print(f"  [WARNING] batch_size={args.batch_size} may risk CUDA OOM on 4GB VRAM! Overriding to 16.")
                args.batch_size = 16
            if args.max_seq_length > 128:
                print(f"  [WARNING] max_seq_length={args.max_seq_length} may risk CUDA OOM on 4GB VRAM! Recommended: 128.")
    else:
        print("WARNING: No GPU detected. Training will run on CPU.")

    # Load data
    train_df, val_df, test_df = load_data(data_dir, args.use_augmented)

    # Compute class weights for multi-label loss
    class_weights = None
    if args.class_weighting != "none":
        class_weights = compute_class_weights(
            train_df,
            mode=args.class_weighting,
            max_weight=args.max_pos_weight
        )

    # Load tokenizer and model (prefer local cache to avoid Hugging Face network timeouts)
    model_name = "distilbert-base-multilingual-cased"
    print(f"Loading model: {model_name}")

    id2label = {i: cls for i, cls in enumerate(TARGET_CLASSES)}
    label2id = {cls: i for i, cls in enumerate(TARGET_CLASSES)}

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(TARGET_CLASSES),
            problem_type="multi_label_classification",
            id2label=id2label,
            label2id=label2id,
            local_files_only=True,
        )
        print("  ✓ Loaded model and tokenizer from local cache (offline mode).")
    except Exception as e:
        print(f"  Local cache load failed ({e}), attempting online download...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(TARGET_CLASSES),
            problem_type="multi_label_classification",
            id2label=id2label,
            label2id=label2id,
        )

    # Prepare datasets
    print("Tokenizing datasets...")
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        train_df, val_df, test_df, tokenizer, args.max_seq_length
    )

    # Training arguments
    # Keep eval batch size equal to train batch size to avoid OOM spikes on 4GB VRAM
    eval_batch_size = min(args.batch_size, 16)

    training_args = TrainingArguments(
        output_dir=str(model_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=eval_batch_size,
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
        report_to="none",
        dataloader_num_workers=0,  # Windows multi-threading compatibility
    )

    # Callbacks
    callbacks = []
    if args.patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))
        print(f"Early stopping enabled: patience={args.patience} epochs (monitoring validation macro_f1)")

    # Trainer instance
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        class_weights=class_weights,
    )

    # Train
    print(f"\nStarting training...")
    print(f"  Max Epochs: {args.epochs}")
    print(f"  Patience: {args.patience}")
    print(f"  Batch size: {args.batch_size} (effective: {args.batch_size * args.gradient_accumulation})")
    print(f"  Max seq length: {args.max_seq_length}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  FP16: {device == 'cuda'}")
    print(f"  Class weighting: {args.class_weighting}")
    print()

    train_result = trainer.train()

    # Evaluate on test set with best checkpoint
    print("\nEvaluating best checkpoint on test set...")
    test_results = trainer.evaluate(test_dataset)
    print("\nTest Results:")
    for key, value in sorted(test_results.items()):
        if key.startswith("eval_"):
            print(f"  {key}: {value:.4f}")

    # Save model and tokenizer
    print(f"\nSaving model to {model_dir}...")
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    # Save label map alongside model
    label_map = {cls: i for i, cls in enumerate(TARGET_CLASSES)}
    with open(model_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    # Save training config & metrics summary
    config = {
        "model_name": model_name,
        "target_classes": TARGET_CLASSES,
        "epochs_requested": args.epochs,
        "epochs_trained": trainer.state.epoch,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "max_seq_length": args.max_seq_length,
        "learning_rate": args.learning_rate,
        "class_weighting": args.class_weighting,
        "patience": args.patience,
        "use_augmented": args.use_augmented,
        "test_macro_f1": test_results.get("eval_macro_f1", 0),
        "test_micro_f1": test_results.get("eval_micro_f1", 0),
        "test_macro_precision": test_results.get("eval_macro_precision", 0),
        "test_macro_recall": test_results.get("eval_macro_recall", 0),
    }
    with open(model_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\n✓ Training complete!")
    print(f"  Model saved to: {model_dir}")
    print(f"  Best Validation Macro-F1: {trainer.state.best_metric:.4f}")
    print(f"  Test Macro-F1: {test_results.get('eval_macro_f1', 0):.4f}")
    print(f"  Test Micro-F1: {test_results.get('eval_micro_f1', 0):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train MoodMax emotion model with class weighting and early stopping")
    parser.add_argument("--epochs", type=int, default=8, help="Maximum number of training epochs (default: 8)")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience in epochs (default: 2, 0 to disable)")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size (default: 16, keep <=16 on 4GB VRAM)")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Max sequence length (default: 128)")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument("--gradient_accumulation", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--class_weighting", type=str, default="sqrt_inverse",
                        choices=["none", "inverse", "sqrt_inverse"],
                        help="Class weighting mode for BCE loss: 'sqrt_inverse' (recommended), 'inverse', or 'none'")
    parser.add_argument("--max_pos_weight", type=float, default=25.0,
                        help="Maximum pos_weight clamp to prevent extreme loss spikes on rare classes")
    parser.add_argument("--use_augmented", action="store_true", help="Use multilingual augmented training data")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
