"""
Phase 1 -- Per-Class Threshold Optimization.

Instead of a flat t=0.5 for all 7 emotion classes, sweep per-class thresholds
on the validation set to find the optimal decision boundary for each class.

Supports two modes:
  - Plain F1 maximization (--no-recall_weighted)
  - F-beta (beta=1.5) maximization (--recall_weighted, default) which weights recall
    higher, consistent with the project's stated priority: "missing a fearful or
    angry customer is more dangerous than routing an ambiguous comment for review."

Usage:
  python optimize_thresholds.py                         # Default: recall-weighted
  python optimize_thresholds.py --no-recall_weighted    # Plain F1
  python optimize_thresholds.py --eval_test             # Also evaluate on test set
  python optimize_thresholds.py --eval_test --no-recall_weighted
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
    fbeta_score,
    precision_score,
    recall_score,
    jaccard_score,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification


TARGET_CLASSES = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]


def extract_probabilities(model, tokenizer, texts, device, temperature=1.0, batch_size=16):
    """Run inference and return calibrated sigmoid probabilities."""
    model.eval()
    all_probs = []

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

        if (i // batch_size) % 50 == 0:
            print(f"  Inference progress: {min(i + batch_size, len(texts))}/{len(texts)}")

    return np.vstack(all_probs)


def sweep_thresholds(probs, true_labels, recall_weighted=True, beta=1.5):
    """
    Sweep thresholds per class and find the optimal threshold.

    Args:
        probs: (N, 7) array of predicted probabilities
        true_labels: (N, 7) array of true binary labels
        recall_weighted: if True, maximize F-beta (b=1.5); otherwise plain F1
        beta: beta parameter for F-beta score

    Returns:
        dict mapping class name to optimal threshold
        dict mapping class name to sweep results
    """
    thresholds_range = np.arange(0.05, 0.96, 0.02)
    optimal_thresholds = {}
    sweep_details = {}

    print("\n" + "=" * 80)
    print(f"THRESHOLD SWEEP ({'F-beta b=' + str(beta) + ' (recall-weighted)' if recall_weighted else 'Plain F1'})")
    print("=" * 80)

    for cls_idx, cls_name in enumerate(TARGET_CLASSES):
        cls_probs = probs[:, cls_idx]
        cls_true = true_labels[:, cls_idx]

        best_threshold = 0.5
        best_score = -1.0
        results = []

        for thresh in thresholds_range:
            cls_pred = (cls_probs >= thresh).astype(int)

            if cls_pred.sum() == 0 or cls_true.sum() == 0:
                results.append({
                    "threshold": round(float(thresh), 2),
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "fbeta": 0.0,
                })
                continue

            prec = precision_score(cls_true, cls_pred, zero_division=0)
            rec = recall_score(cls_true, cls_pred, zero_division=0)
            f1 = f1_score(cls_true, cls_pred, zero_division=0)
            fb = fbeta_score(cls_true, cls_pred, beta=beta, zero_division=0)

            score = fb if recall_weighted else f1
            results.append({
                "threshold": round(float(thresh), 2),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1": round(float(f1), 4),
                "fbeta": round(float(fb), 4),
            })

            if score > best_score:
                best_score = score
                best_threshold = round(float(thresh), 2)

        optimal_thresholds[cls_name] = best_threshold
        sweep_details[cls_name] = results

        # Find the metrics at the optimal threshold
        opt_result = next((r for r in results if r["threshold"] == best_threshold), None)
        metric_name = f"F-beta(beta={beta})" if recall_weighted else "F1"
        print(f"  {cls_name:<12} optimal t = {best_threshold:.2f}  "
              f"(P={opt_result['precision']:.4f}  R={opt_result['recall']:.4f}  "
              f"F1={opt_result['f1']:.4f}  {metric_name}={best_score:.4f})")

    print("=" * 80)
    return optimal_thresholds, sweep_details


def evaluate_with_thresholds(probs, true_labels, thresholds, label="Custom Thresholds"):
    """Evaluate predictions using per-class thresholds."""
    pred_labels = np.zeros_like(probs, dtype=int)
    for cls_idx, cls_name in enumerate(TARGET_CLASSES):
        thresh = thresholds.get(cls_name, 0.5)
        pred_labels[:, cls_idx] = (probs[:, cls_idx] >= thresh).astype(int)

    macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    micro_f1 = f1_score(true_labels, pred_labels, average="micro", zero_division=0)
    macro_prec = precision_score(true_labels, pred_labels, average="macro", zero_division=0)
    macro_rec = recall_score(true_labels, pred_labels, average="macro", zero_division=0)
    jaccard = jaccard_score(true_labels, pred_labels, average="samples", zero_division=0)

    per_class_f1 = f1_score(true_labels, pred_labels, average=None, zero_division=0)
    per_class_prec = precision_score(true_labels, pred_labels, average=None, zero_division=0)
    per_class_rec = recall_score(true_labels, pred_labels, average=None, zero_division=0)

    print(f"\n{'-' * 70}")
    print(f"  EVALUATION: {label}")
    print(f"{'-' * 70}")
    print(f"  Macro-F1:        {macro_f1:.4f}")
    print(f"  Micro-F1:        {micro_f1:.4f}")
    print(f"  Macro-Precision: {macro_prec:.4f}")
    print(f"  Macro-Recall:    {macro_rec:.4f}")
    print(f"  Jaccard:         {jaccard:.4f}")
    print()
    print(f"  {'Class':<12} {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'-' * 62}")

    for cls_idx, cls_name in enumerate(TARGET_CLASSES):
        support = int(true_labels[:, cls_idx].sum())
        thresh = thresholds.get(cls_name, 0.5)
        print(f"  {cls_name:<12} {thresh:>10.2f} {per_class_prec[cls_idx]:>10.4f} "
              f"{per_class_rec[cls_idx]:>10.4f} {per_class_f1[cls_idx]:>10.4f} {support:>10}")

    print(f"{'-' * 70}")

    return {
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "jaccard_samples": float(jaccard),
        "per_class": {
            cls: {
                "f1": float(per_class_f1[i]),
                "precision": float(per_class_prec[i]),
                "recall": float(per_class_rec[i]),
                "support": int(true_labels[:, i].sum()),
                "threshold": thresholds.get(cls, 0.5),
            }
            for i, cls in enumerate(TARGET_CLASSES)
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Per-class threshold optimization for MoodMax emotion model")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Path to the trained model directory")
    parser.add_argument("--recall_weighted", action=argparse.BooleanOptionalAction, default=True,
                        help="Maximize F-beta (b=1.5) instead of plain F1 (default: True)")
    parser.add_argument("--beta", type=float, default=1.5,
                        help="Beta parameter for F-beta score (default: 1.5)")
    parser.add_argument("--eval_test", action="store_true",
                        help="Also evaluate on the test set with optimized thresholds")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Inference batch size (default: 16)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    model_dir = Path(args.model_dir) if args.model_dir else project_root / "models" / "emotion-distilbert-multi"
    data_dir = project_root / "data" / "processed"

    print("=" * 70)
    print("MoodMax Phase 1 -- Per-Class Threshold Optimization")
    print("=" * 70)

    # Validate paths
    if not model_dir.exists():
        print(f"ERROR: Model not found at {model_dir}")
        sys.exit(1)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"Loading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()

    # Load calibration temperature
    temperature = 1.0
    calib_path = model_dir / "calibration.json"
    if calib_path.exists():
        with open(calib_path) as f:
            calib = json.load(f)
            temperature = float(calib.get("temperature", 1.0))
            print(f"  Loaded calibration temperature T = {temperature:.4f}")

    # --------------------------------------
    # Step 1: Extract validation probabilities
    # --------------------------------------
    val_path = data_dir / "val_collapsed.csv"
    if not val_path.exists():
        print(f"ERROR: Validation data not found at {val_path}")
        sys.exit(1)

    val_df = pd.read_csv(val_path)
    val_texts = val_df["text"].fillna("").tolist()
    val_labels = val_df[TARGET_CLASSES].values
    print(f"\nValidation set: {len(val_df):,} rows")

    print("\nExtracting validation probabilities...")
    val_probs = extract_probabilities(model, tokenizer, val_texts, device, temperature, args.batch_size)

    # Save validation probabilities
    val_probs_df = pd.DataFrame(val_probs, columns=TARGET_CLASSES)
    val_probs_path = model_dir / "val_probs.csv"
    val_probs_df.to_csv(val_probs_path, index=False)
    print(f"  Saved validation probabilities to {val_probs_path}")

    # Also save as numpy
    np.save(model_dir / "val_probs.npy", val_probs)

    # --------------------------------------
    # Step 2: Baseline evaluation (flat t=0.5)
    # --------------------------------------
    flat_thresholds = {cls: 0.5 for cls in TARGET_CLASSES}
    print("\n" + "=" * 70)
    print("BASELINE EVALUATION (Flat t = 0.5)")
    print("=" * 70)
    baseline_val = evaluate_with_thresholds(val_probs, val_labels, flat_thresholds, label="Validation -- Flat t=0.5")

    # --------------------------------------
    # Step 3: Sweep thresholds
    # --------------------------------------
    optimal_thresholds, sweep_details = sweep_thresholds(
        val_probs, val_labels,
        recall_weighted=args.recall_weighted,
        beta=args.beta,
    )

    # --------------------------------------
    # Step 4: Evaluate with optimal thresholds on validation
    # --------------------------------------
    mode_label = f"F-beta (b={args.beta})" if args.recall_weighted else "Plain F1"
    optimized_val = evaluate_with_thresholds(
        val_probs, val_labels, optimal_thresholds,
        label=f"Validation -- Optimized ({mode_label})"
    )

    # --------------------------------------
    # Step 5: Save thresholds.json
    # --------------------------------------
    thresholds_output = {
        "mode": "recall_weighted" if args.recall_weighted else "f1",
        "beta": args.beta if args.recall_weighted else 1.0,
        "thresholds": optimal_thresholds,
        "validation_macro_f1_before": baseline_val["macro_f1"],
        "validation_macro_f1_after": optimized_val["macro_f1"],
    }

    thresholds_path = model_dir / "thresholds.json"
    with open(thresholds_path, "w") as f:
        json.dump(thresholds_output, f, indent=2)
    print(f"\n[OK] Thresholds saved to {thresholds_path}")
    print(f"     {json.dumps(optimal_thresholds)}")

    # --------------------------------------
    # Step 6: Evaluate on test set (optional)
    # --------------------------------------
    if args.eval_test:
        test_path = data_dir / "test_collapsed.csv"
        if not test_path.exists():
            print(f"WARNING: Test data not found at {test_path}, skipping test evaluation.")
        else:
            test_df = pd.read_csv(test_path)
            test_texts = test_df["text"].fillna("").tolist()
            test_labels = test_df[TARGET_CLASSES].values
            print(f"\nTest set: {len(test_df):,} rows")

            print("\nExtracting test probabilities...")
            test_probs = extract_probabilities(model, tokenizer, test_texts, device, temperature, args.batch_size)

            # Save test probabilities for Phase 2
            np.save(model_dir / "test_probs.npy", test_probs)
            test_probs_df = pd.DataFrame(test_probs, columns=TARGET_CLASSES)
            test_probs_df.to_csv(model_dir / "test_probs.csv", index=False)

            # Baseline (flat 0.5)
            print("\n" + "=" * 70)
            print("TEST SET COMPARISON")
            print("=" * 70)
            baseline_test = evaluate_with_thresholds(test_probs, test_labels, flat_thresholds,
                                                     label="Test -- Flat t=0.5 (BEFORE)")
            optimized_test = evaluate_with_thresholds(test_probs, test_labels, optimal_thresholds,
                                                      label=f"Test -- Optimized ({mode_label}) (AFTER)")

            # Print delta summary
            print("\n" + "=" * 70)
            print("BEFORE / AFTER COMPARISON (Test Set)")
            print("=" * 70)
            print(f"  {'Metric':<20} {'Before (t=0.5)':>15} {'After (Optimized)':>18} {'Delta':>10}")
            print(f"  {'-' * 65}")

            for metric in ["macro_f1", "micro_f1", "macro_precision", "macro_recall", "jaccard_samples"]:
                before = baseline_test[metric]
                after = optimized_test[metric]
                delta = after - before
                sign = "+" if delta >= 0 else ""
                print(f"  {metric:<20} {before:>15.4f} {after:>18.4f} {sign}{delta:>9.4f}")

            print(f"\n  {'Class':<12} {'F1 Before':>10} {'F1 After':>10} {'dF1':>8} {'t':>6}")
            print(f"  {'-' * 48}")
            for cls in TARGET_CLASSES:
                f1_before = baseline_test["per_class"][cls]["f1"]
                f1_after = optimized_test["per_class"][cls]["f1"]
                delta = f1_after - f1_before
                thresh = optimal_thresholds[cls]
                sign = "+" if delta >= 0 else ""
                print(f"  {cls:<12} {f1_before:>10.4f} {f1_after:>10.4f} {sign}{delta:>7.4f} {thresh:>6.2f}")

            print("=" * 70)

            # Check acceptance criteria
            print("\n" + "=" * 70)
            print("ACCEPTANCE CRITERIA CHECK")
            print("=" * 70)
            f1_improved = optimized_test["macro_f1"] > baseline_test["macro_f1"]
            recall_ok = optimized_test["macro_recall"] >= 0.6695  # Current baseline

            print(f"  macro-F1 improved:     {'[PASS]' if f1_improved else '[FAIL]'} "
                  f"({baseline_test['macro_f1']:.4f} -> {optimized_test['macro_f1']:.4f})")
            print(f"  macro-recall >= 66.95%: {'[PASS]' if recall_ok else '[FAIL]'} "
                  f"({optimized_test['macro_recall']:.4f})")

            if f1_improved and recall_ok:
                print("\n  [PASS] Phase 1 ACCEPTED -- thresholds improve metrics without breaking recall.")
            elif f1_improved and not recall_ok:
                print(f"\n  [WARN] Phase 1 PARTIAL -- F1 improved but recall dropped below baseline.")
                print(f"    Consider re-running with --recall_weighted to prioritize recall.")
            else:
                print(f"\n  [FAIL] Phase 1 REJECTED -- thresholds did not improve macro-F1 on test set.")
                print(f"    The flat t=0.5 threshold may already be near-optimal for this model.")
            print("=" * 70)


if __name__ == "__main__":
    main()

