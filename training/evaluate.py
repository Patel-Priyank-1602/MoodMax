"""
Evaluate the fine-tuned emotion model on the test set.

Outputs:
  - Macro-F1, Micro-F1
  - Per-label Precision, Recall, F1
  - Per-language F1 breakdown (if augmented data used)
  - Confusion-style analysis
  - Jaccard similarity

Usage:
  python evaluate.py
  python evaluate.py --model_dir ../models/emotion-distilbert-multi
  python evaluate.py --per_language   # If test data has language tags
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    jaccard_score,
    multilabel_confusion_matrix,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification


TARGET_CLASSES = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]


def evaluate_model(args):
    """Run full evaluation of the emotion model."""

    project_root = Path(__file__).resolve().parent.parent
    model_dir = Path(args.model_dir) if args.model_dir else project_root / "models" / "emotion-distilbert-multi"
    data_dir = project_root / "data" / "processed"

    print("=" * 60)
    print("MoodMax Emotion Model Evaluation")
    print("=" * 60)

    # Check model exists
    if not model_dir.exists():
        print(f"ERROR: Model not found at {model_dir}")
        print("Train the model first with: python training/train.py")
        sys.exit(1)

    # Load model
    print(f"\nLoading model from {model_dir}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()
    print(f"  Device: {device}")

    # Load test data
    test_path = data_dir / "test_collapsed.csv"
    if not test_path.exists():
        print(f"ERROR: Test data not found at {test_path}")
        sys.exit(1)

    test_df = pd.read_csv(test_path)
    print(f"  Test set: {len(test_df):,} rows")

    # Run inference
    print("\nRunning inference...")
    texts = test_df["text"].fillna("").tolist()
    true_labels = test_df[TARGET_CLASSES].values

    # Check calibration
    temperature = args.temperature
    if temperature is None and (model_dir / "calibration.json").exists():
        with open(model_dir / "calibration.json") as f:
            calib = json.load(f)
            temperature = float(calib.get("temperature", 1.0))
            print(f"  Loaded calibration temperature T = {temperature:.4f} from calibration.json")
    elif temperature is not None:
        print(f"  Using specified calibration temperature T = {temperature:.4f}")
    else:
        temperature = 1.0

    all_probs = []
    batch_size = min(args.batch_size, 32)

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            scaled_logits = outputs.logits / temperature
            probs = torch.sigmoid(scaled_logits).cpu().numpy()
            all_probs.append(probs)

        if (i // batch_size) % 20 == 0:
            print(f"  Progress: {min(i + batch_size, len(texts))}/{len(texts)}")

    all_probs = np.vstack(all_probs)
    pred_labels = (all_probs > 0.5).astype(int)

    # ─── Overall Metrics ───
    print("\n" + "=" * 60)
    print("OVERALL METRICS")
    print("=" * 60)

    macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    micro_f1 = f1_score(true_labels, pred_labels, average="micro", zero_division=0)
    macro_precision = precision_score(true_labels, pred_labels, average="macro", zero_division=0)
    macro_recall = recall_score(true_labels, pred_labels, average="macro", zero_division=0)
    jaccard = jaccard_score(true_labels, pred_labels, average="samples", zero_division=0)

    print(f"  Macro-F1:       {macro_f1:.4f}")
    print(f"  Micro-F1:       {micro_f1:.4f}")
    print(f"  Macro-Precision: {macro_precision:.4f}")
    print(f"  Macro-Recall:   {macro_recall:.4f}")
    print(f"  Jaccard (samples): {jaccard:.4f}")

    # ─── Per-Class Metrics ───
    print("\n" + "=" * 60)
    print("PER-CLASS METRICS")
    print("=" * 60)

    per_class_f1 = f1_score(true_labels, pred_labels, average=None, zero_division=0)
    per_class_prec = precision_score(true_labels, pred_labels, average=None, zero_division=0)
    per_class_recall = recall_score(true_labels, pred_labels, average=None, zero_division=0)

    print(f"\n  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("  " + "-" * 52)
    for i, cls in enumerate(TARGET_CLASSES):
        support = true_labels[:, i].sum()
        print(f"  {cls:<12} {per_class_prec[i]:>10.4f} {per_class_recall[i]:>10.4f} "
              f"{per_class_f1[i]:>10.4f} {support:>10}")

    # ─── Per-Language Metrics (if applicable) ───
    if args.per_language and "lang" in test_df.columns:
        print("\n" + "=" * 60)
        print("PER-LANGUAGE METRICS")
        print("=" * 60)

        for lang in sorted(test_df["lang"].unique()):
            mask = test_df["lang"] == lang
            lang_true = true_labels[mask]
            lang_pred = pred_labels[mask]

            if len(lang_true) == 0:
                continue

            lang_f1 = f1_score(lang_true, lang_pred, average="macro", zero_division=0)
            print(f"\n  Language: {lang} ({mask.sum():,} rows)")
            print(f"    Macro-F1: {lang_f1:.4f}")

    # ─── Multilabel Confusion Matrix ───
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX (per class)")
    print("=" * 60)

    mcm = multilabel_confusion_matrix(true_labels, pred_labels)
    for i, cls in enumerate(TARGET_CLASSES):
        tn, fp, fn, tp = mcm[i].ravel()
        print(f"\n  {cls}:")
        print(f"    TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")

    # ─── Save results ───
    results = {
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "jaccard_samples": float(jaccard),
        "per_class": {
            cls: {
                "f1": float(per_class_f1[i]),
                "precision": float(per_class_prec[i]),
                "recall": float(per_class_recall[i]),
                "support": int(true_labels[:, i].sum()),
            }
            for i, cls in enumerate(TARGET_CLASSES)
        },
    }

    results_path = model_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MoodMax emotion model")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Path to the trained model directory")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Post-hoc temperature scaling parameter T (defaults to calibration.json if present)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Evaluation batch size (default: 16)")
    parser.add_argument("--per_language", action="store_true",
                        help="Show per-language F1 breakdown")
    args = parser.parse_args()
    evaluate_model(args)


if __name__ == "__main__":
    main()
