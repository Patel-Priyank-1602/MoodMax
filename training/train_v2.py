"""
Phase 4 — Targeted Retraining with Rare-Class Oversampling.

Enhanced training script that adds:
  1. Targeted oversampling for fear/disgust/sadness (2-3x factor)
  2. WeightedRandomSampler for class-aware sampling
  3. Same hyperparameters as original (batch 32, LR 2e-5, FP16, early stopping)
  4. Saves new checkpoint to models/emotion-distilbert-multi-v2/ initially

⚠️ This script is designed to run on Google Colab with a Tesla T4 GPU.
   It will NOT run efficiently on CPU.

Usage (Colab):
  python train_v2.py                                    # Default oversampling
  python train_v2.py --oversample_factor 3              # 3x oversampling for rare classes
  python train_v2.py --batch_size 32 --epochs 8         # Full training run
  python train_v2.py --use_augmented                    # Include multilingual data
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Mitigate CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)


TARGET_CLASSES = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]
RARE_CLASSES = ["fear", "disgust", "sadness"]


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
    """Compute positive class weights for multi-label BCEWithLogitsLoss."""
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
# Sampling Weights for Rare-Class Oversampling
# ─────────────────────────────────────────────

def compute_sample_weights(train_df: pd.DataFrame, oversample_factor: float = 2.5) -> np.ndarray:
    """
    Compute per-sample weights for WeightedRandomSampler.

    Samples containing rare-class labels (fear, disgust, sadness) get
    a higher sampling weight, independent from BCE pos_weight.

    This changes HOW OFTEN the model sees diverse rare-class examples
    (pos_weight changes gradient magnitude, not example frequency).
    """
    weights = np.ones(len(train_df), dtype=np.float64)

    for cls in RARE_CLASSES:
        cls_idx = TARGET_CLASSES.index(cls)
        positive_mask = train_df[cls].values == 1
        weights[positive_mask] *= oversample_factor

    # Normalize so total weight equals dataset length
    weights = weights / weights.sum() * len(train_df)

    # Log the effective oversampling
    print("\n" + "=" * 70)
    print(f"TARGETED OVERSAMPLING (factor={oversample_factor}x for {RARE_CLASSES})")
    print("=" * 70)
    for cls in RARE_CLASSES:
        pos_count = int(train_df[cls].sum())
        effective = pos_count * oversample_factor
        print(f"  {cls:<12} original: {pos_count:>6,d}  effective: ~{int(effective):>6,d}")
    print("=" * 70 + "\n")

    return weights


# ─────────────────────────────────────────────
# Custom Weighted Trainer with Oversampling
# ─────────────────────────────────────────────

class OversamplingTrainer(Trainer):
    """Custom Trainer with per-class pos_weight and optional oversampling."""

    def __init__(self, *args, class_weights=None, sample_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.sample_weights = sample_weights

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

    def get_train_dataloader(self) -> DataLoader:
        """Override to use WeightedRandomSampler if sample weights are provided."""
        if self.sample_weights is not None:
            sampler = WeightedRandomSampler(
                weights=self.sample_weights,
                num_samples=len(self.train_dataset),
                replacement=True,
            )
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                sampler=sampler,
                num_workers=self.args.dataloader_num_workers,
                drop_last=self.args.dataloader_drop_last,
            )
        return super().get_train_dataloader()


# ─────────────────────────────────────────────
# Metrics
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
# Data Loading
# ─────────────────────────────────────────────

def load_data(data_dir: Path, use_augmented: bool = False):
    """Load training, validation, and test sets."""
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
            print(f"ERROR: {path} not found.")
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
            texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt",
        )
        return EmotionDataset(encodings, labels)

    return tokenize(train_df), tokenize(val_df), tokenize(test_df)


# ─────────────────────────────────────────────
# Main Training Pipeline
# ─────────────────────────────────────────────

def train(args):
    """Run the enhanced training pipeline with oversampling."""
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "processed"
    model_dir = project_root / "models" / "emotion-distilbert-multi-v2"

    print("=" * 70)
    print("MoodMax Phase 4 — Targeted Retraining with Oversampling")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

        if gpu_mem <= 4.5:
            print("  [INFO] 4GB VRAM GPU detected. Adjusting batch size.")
            if args.batch_size > 16:
                args.batch_size = 16
    else:
        print("WARNING: No GPU detected. Training will be extremely slow on CPU.")

    # Load data
    train_df, val_df, test_df = load_data(data_dir, args.use_augmented)

    # Compute class weights for BCE loss
    class_weights = compute_class_weights(train_df, mode=args.class_weighting, max_weight=args.max_pos_weight)

    # Compute sample weights for oversampling
    sample_weights = None
    if args.oversample_factor > 1.0:
        sample_weights = compute_sample_weights(train_df, args.oversample_factor)

    # Load tokenizer and model
    model_name = "distilbert-base-multilingual-cased"
    print(f"Loading base model: {model_name}")

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
        print("  ✓ Loaded model from local cache.")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(TARGET_CLASSES),
            problem_type="multi_label_classification",
            id2label=id2label,
            label2id=label2id,
        )
        print("  ✓ Downloaded model from Hugging Face Hub.")

    # Prepare datasets
    print("Tokenizing datasets...")
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        train_df, val_df, test_df, tokenizer, args.max_seq_length
    )

    # Training arguments
    eval_batch_size = min(args.batch_size, 32)

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
        logging_steps=100,
        report_to="none",
        dataloader_num_workers=0,
    )

    # Callbacks
    callbacks = []
    if args.patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))

    # Trainer with oversampling
    trainer = OversamplingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        class_weights=class_weights,
        sample_weights=sample_weights,
    )

    # Train
    print(f"\nStarting training...")
    print(f"  Max Epochs: {args.epochs}")
    print(f"  Patience: {args.patience}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Oversample factor: {args.oversample_factor}x for {RARE_CLASSES}")
    print(f"  Max seq length: {args.max_seq_length}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  FP16: {device == 'cuda'}")
    print(f"  Class weighting: {args.class_weighting}")
    print()

    train_result = trainer.train()

    # Evaluate on test set
    print("\nEvaluating best checkpoint on test set...")
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

    # Save label map
    label_map = {cls: i for i, cls in enumerate(TARGET_CLASSES)}
    with open(model_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    # Save training config
    config = {
        "model_name": model_name,
        "target_classes": TARGET_CLASSES,
        "phase": "Phase 4 — Targeted Retraining",
        "oversample_factor": args.oversample_factor,
        "rare_classes_oversampled": RARE_CLASSES,
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

    print("\n✓ Phase 4 Training complete!")
    print(f"  Model saved to: {model_dir}")
    print(f"  Best Validation Macro-F1: {trainer.state.best_metric:.4f}")
    print(f"  Test Macro-F1: {test_results.get('eval_macro_f1', 0):.4f}")
    print(f"  Test Micro-F1: {test_results.get('eval_micro_f1', 0):.4f}")
    print(f"\n  ⚠ To promote this model, copy contents of {model_dir}")
    print(f"    to models/emotion-distilbert-multi/ after verifying metrics.")


def main():
    parser = argparse.ArgumentParser(description="Phase 4 — Targeted retraining with rare-class oversampling")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size (32 for T4, 16 for 4GB VRAM)")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    parser.add_argument("--class_weighting", type=str, default="sqrt_inverse",
                        choices=["none", "inverse", "sqrt_inverse"])
    parser.add_argument("--max_pos_weight", type=float, default=25.0)
    parser.add_argument("--oversample_factor", type=float, default=2.5,
                        help="Oversampling factor for rare classes (fear/disgust/sadness). Default: 2.5x")
    parser.add_argument("--use_augmented", action="store_true")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
