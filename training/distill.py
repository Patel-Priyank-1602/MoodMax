"""
Phase 5 — Optional Knowledge Distillation from xlm-roberta Teacher.

Inherit accuracy from a larger multilingual model (xlm-roberta-base)
without inflating serving cost. The DistilBERT student stays as the
production model.

Steps:
  1. Fine-tune xlm-roberta-base as a teacher on the training set
  2. Generate soft labels (teacher's calibrated probability vectors)
  3. Retrain DistilBERT student with combined loss:
     L = α * BCE(student_logits, hard_labels) + (1-α) * KL_div(student_probs, teacher_probs)
  4. Compare against Phase 4 checkpoint — adopt only if macro-F1 improves >1pt

⚠️ Requires Google Colab with Tesla T4 GPU.

Usage (Colab):
  python distill.py                                # Full pipeline
  python distill.py --alpha 0.5                    # Specific alpha
  python distill.py --teacher_epochs 4             # Fewer teacher epochs
  python distill.py --skip_teacher_training        # Reuse existing teacher
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)


TARGET_CLASSES = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]


class EmotionDataset(Dataset):
    def __init__(self, encodings, labels, soft_labels=None):
        self.encodings = encodings
        self.labels = labels
        self.soft_labels = soft_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        if self.soft_labels is not None:
            item["soft_labels"] = torch.tensor(self.soft_labels[idx], dtype=torch.float)
        return item


class DistillationTrainer(Trainer):
    """Custom Trainer implementing combined hard-label + soft-label distillation loss."""

    def __init__(self, *args, class_weights=None, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        soft_labels = inputs.get("soft_labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Hard label loss (BCE with class weights)
        if self.class_weights is not None:
            bce_loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.class_weights.to(logits.device))
        else:
            bce_loss_fct = nn.BCEWithLogitsLoss()

        hard_loss = bce_loss_fct(logits, labels)

        if soft_labels is not None:
            # Soft label loss (KL divergence between student and teacher probabilities)
            student_probs = torch.sigmoid(logits / self.temperature)
            teacher_probs = soft_labels  # Already probabilities from teacher

            # Binary KL divergence per class (multi-label, not softmax)
            kl_loss = F.binary_cross_entropy(
                student_probs.clamp(1e-7, 1 - 1e-7),
                teacher_probs.clamp(1e-7, 1 - 1e-7),
                reduction="mean",
            ) * (self.temperature ** 2)

            loss = self.alpha * hard_loss + (1 - self.alpha) * kl_loss
        else:
            loss = hard_loss

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)

    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    macro_prec = precision_score(labels, preds, average="macro", zero_division=0)
    macro_rec = recall_score(labels, preds, average="macro", zero_division=0)

    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)

    print(f"\n  Macro-F1: {macro_f1:.4f} | Per-class: " +
          " ".join(f"{cls}={per_class_f1[i]:.3f}" for i, cls in enumerate(TARGET_CLASSES)))

    return {
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
    }


def compute_class_weights(train_df, mode="sqrt_inverse", max_weight=25.0):
    """Compute BCE pos_weights."""
    total = len(train_df)
    weights = []
    for cls in TARGET_CLASSES:
        pos = int(train_df[cls].sum())
        neg = total - pos
        ratio = neg / max(pos, 1)
        w = np.sqrt(ratio) if mode == "sqrt_inverse" else ratio
        weights.append(min(w, max_weight))
    return torch.tensor(weights, dtype=torch.float)


def generate_soft_labels(teacher_model, teacher_tokenizer, texts, device, temperature=1.0, batch_size=32):
    """Generate soft probability labels from the teacher model."""
    teacher_model.eval()
    all_probs = []

    print("Generating soft labels from teacher...")
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = teacher_tokenizer(
            batch, return_tensors="pt", truncation=True, padding=True, max_length=128
        ).to(device)

        with torch.no_grad():
            outputs = teacher_model(**inputs)
            probs = torch.sigmoid(outputs.logits / temperature).cpu().numpy()
            all_probs.append(probs)

        if (i // batch_size) % 50 == 0:
            print(f"  Progress: {min(i + batch_size, len(texts))}/{len(texts)}")

    return np.vstack(all_probs)


def main():
    parser = argparse.ArgumentParser(description="Phase 5 — Knowledge distillation from xlm-roberta teacher")
    parser.add_argument("--teacher_model", type=str, default="xlm-roberta-base")
    parser.add_argument("--teacher_epochs", type=int, default=4)
    parser.add_argument("--student_epochs", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for hard-label loss (0.3-0.7)")
    parser.add_argument("--distill_temperature", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--skip_teacher_training", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "processed"
    teacher_dir = project_root / "models" / "emotion-xlm-roberta-teacher"
    student_dir = project_root / "models" / "emotion-distilbert-distilled"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("MoodMax Phase 5 — Knowledge Distillation")
    print("=" * 70)
    print(f"Device: {device}")

    # Load data
    train_df = pd.read_csv(data_dir / "train_collapsed.csv")
    val_df = pd.read_csv(data_dir / "val_collapsed.csv")
    test_df = pd.read_csv(data_dir / "test_collapsed.csv")
    print(f"Data: Train={len(train_df):,}  Val={len(val_df):,}  Test={len(test_df):,}")

    train_texts = train_df["text"].fillna("").tolist()
    train_labels = train_df[TARGET_CLASSES].values.tolist()

    # ──────────────────────────────────────
    # Step 1: Train teacher (xlm-roberta-base)
    # ──────────────────────────────────────
    if not args.skip_teacher_training:
        print(f"\n{'='*70}\nStep 1: Training teacher model ({args.teacher_model})\n{'='*70}")

        id2label = {i: cls for i, cls in enumerate(TARGET_CLASSES)}
        label2id = {cls: i for i, cls in enumerate(TARGET_CLASSES)}

        teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
        teacher_model = AutoModelForSequenceClassification.from_pretrained(
            args.teacher_model,
            num_labels=len(TARGET_CLASSES),
            problem_type="multi_label_classification",
            id2label=id2label,
            label2id=label2id,
        )

        # Tokenize for teacher
        train_enc = teacher_tokenizer(train_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
        val_texts = val_df["text"].fillna("").tolist()
        val_enc = teacher_tokenizer(val_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")

        train_dataset = EmotionDataset(train_enc, train_labels)
        val_dataset = EmotionDataset(val_enc, val_df[TARGET_CLASSES].values.tolist())

        class_weights = compute_class_weights(train_df)

        # Simple weighted trainer for teacher
        class TeacherTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.get("labels")
                outputs = model(**inputs)
                loss_fct = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(outputs.logits.device))
                loss = loss_fct(outputs.logits, labels)
                return (loss, outputs) if return_outputs else loss

        teacher_args = TrainingArguments(
            output_dir=str(teacher_dir / "checkpoints"),
            num_train_epochs=args.teacher_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=0.01,
            fp16=(device == "cuda"),
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            logging_steps=100,
            report_to="none",
            dataloader_num_workers=0,
        )

        teacher_trainer = TeacherTrainer(
            model=teacher_model,
            args=teacher_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
        )

        teacher_trainer.train()
        teacher_dir.mkdir(parents=True, exist_ok=True)
        teacher_trainer.save_model(str(teacher_dir))
        teacher_tokenizer.save_pretrained(str(teacher_dir))
        print(f"Teacher model saved to {teacher_dir}")
    else:
        print(f"\nSkipping teacher training — loading from {teacher_dir}")
        teacher_tokenizer = AutoTokenizer.from_pretrained(str(teacher_dir))
        teacher_model = AutoModelForSequenceClassification.from_pretrained(str(teacher_dir))

    # ──────────────────────────────────────
    # Step 2: Generate soft labels
    # ──────────────────────────────────────
    print(f"\n{'='*70}\nStep 2: Generating soft labels from teacher\n{'='*70}")
    teacher_model = AutoModelForSequenceClassification.from_pretrained(str(teacher_dir)).to(device)
    teacher_tokenizer = AutoTokenizer.from_pretrained(str(teacher_dir))

    soft_labels = generate_soft_labels(
        teacher_model, teacher_tokenizer, train_texts, device,
        temperature=1.0, batch_size=args.batch_size
    )
    np.save(teacher_dir / "soft_labels.npy", soft_labels)
    print(f"Soft labels saved: {soft_labels.shape}")

    # Free teacher memory
    del teacher_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    import gc; gc.collect()

    # ──────────────────────────────────────
    # Step 3: Train distilled student
    # ──────────────────────────────────────
    print(f"\n{'='*70}\nStep 3: Training distilled DistilBERT student (α={args.alpha})\n{'='*70}")

    student_model_name = "distilbert-base-multilingual-cased"
    student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)

    id2label = {i: cls for i, cls in enumerate(TARGET_CLASSES)}
    label2id = {cls: i for i, cls in enumerate(TARGET_CLASSES)}

    student_model = AutoModelForSequenceClassification.from_pretrained(
        student_model_name,
        num_labels=len(TARGET_CLASSES),
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
    )

    # Tokenize for student
    train_enc = student_tokenizer(train_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    val_texts = val_df["text"].fillna("").tolist()
    val_enc = student_tokenizer(val_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    test_texts = test_df["text"].fillna("").tolist()
    test_enc = student_tokenizer(test_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")

    train_dataset = EmotionDataset(train_enc, train_labels, soft_labels=soft_labels.tolist())
    val_dataset = EmotionDataset(val_enc, val_df[TARGET_CLASSES].values.tolist())
    test_dataset = EmotionDataset(test_enc, test_df[TARGET_CLASSES].values.tolist())

    class_weights = compute_class_weights(train_df)

    student_args = TrainingArguments(
        output_dir=str(student_dir / "checkpoints"),
        num_train_epochs=args.student_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
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

    distill_trainer = DistillationTrainer(
        model=student_model,
        args=student_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
        class_weights=class_weights,
        alpha=args.alpha,
        temperature=args.distill_temperature,
    )

    distill_trainer.train()

    # Evaluate
    test_results = distill_trainer.evaluate(test_dataset)
    print(f"\nDistilled Student Test Results:")
    for key, value in sorted(test_results.items()):
        if key.startswith("eval_"):
            print(f"  {key}: {value:.4f}")

    # Save
    student_dir.mkdir(parents=True, exist_ok=True)
    distill_trainer.save_model(str(student_dir))
    student_tokenizer.save_pretrained(str(student_dir))

    label_map = {cls: i for i, cls in enumerate(TARGET_CLASSES)}
    with open(student_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    config = {
        "phase": "Phase 5 — Knowledge Distillation",
        "teacher_model": args.teacher_model,
        "student_model": student_model_name,
        "alpha": args.alpha,
        "distill_temperature": args.distill_temperature,
        "test_macro_f1": test_results.get("eval_macro_f1", 0),
        "test_micro_f1": test_results.get("eval_micro_f1", 0),
    }
    with open(student_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ Phase 5 complete!")
    print(f"  Distilled model saved to: {student_dir}")
    print(f"  Test Macro-F1: {test_results.get('eval_macro_f1', 0):.4f}")
    print(f"\n  ⚠ Only adopt if macro-F1 improves by >1 point over Phase 4 checkpoint.")


if __name__ == "__main__":
    main()
