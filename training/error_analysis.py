"""
Phase 2 — Rare-Class Error Analysis.

Diagnose whether the disgust/fear/sadness weakness is a data-volume problem,
label-ambiguity problem, or model-capacity problem by analyzing false positives
and false negatives for those classes.

This determines whether Phase 4 (retraining) should even be attempted.

Usage:
  python error_analysis.py                          # Uses saved test_probs.npy from Phase 1
  python error_analysis.py --max_examples 30        # Limit examples per class
  python error_analysis.py --model_dir path/to/model
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


TARGET_CLASSES = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]
RARE_CLASSES = ["disgust", "fear", "sadness"]


def classify_error_bucket(text, true_labels, pred_labels, cls_name, cls_prob, confused_cls, confused_prob):
    """
    Heuristically classify an error into one of four buckets.

    Buckets:
    1. ambiguous_multilabel - text genuinely expresses two emotions
    2. annotation_noise - GoEmotions crowd label looks wrong
    3. clear_model_miss - unambiguous text, model got it wrong
    4. too_short - under ~5 tokens, insufficient context
    """
    tokens = text.split()
    num_tokens = len(tokens)

    # Bucket 4: Too short
    if num_tokens < 5:
        return "too_short"

    # Count how many true labels are active
    true_active = sum(true_labels)

    # Bucket 1: Ambiguous / multi-label boundary
    # If text has multiple true labels OR the confused class is also close in probability
    if true_active >= 2:
        return "ambiguous_multilabel"

    # If the confused class probability is within 0.15 of the target class probability
    if abs(cls_prob - confused_prob) < 0.15 and confused_prob > 0.2:
        return "ambiguous_multilabel"

    # Bucket 2: Likely annotation noise
    # Heuristic: if model is very confident (>0.6) on a different class and the text
    # doesn't obviously match the labeled class
    # This is a rough heuristic — errors where the model's top prediction makes more
    # sense than the label
    if confused_prob > 0.6 and cls_prob < 0.3:
        return "annotation_noise"

    # Bucket 3: Clear model miss (everything else)
    return "clear_model_miss"


def get_confused_class(probs_row, cls_idx):
    """Find the class the model most likely confused this with."""
    # Get probabilities for all classes except the target
    confused_idx = -1
    confused_prob = -1.0
    for i, cls in enumerate(TARGET_CLASSES):
        if i != cls_idx and probs_row[i] > confused_prob:
            confused_prob = probs_row[i]
            confused_idx = i

    return TARGET_CLASSES[confused_idx], confused_prob


def analyze_errors(test_df, test_probs, test_labels, thresholds, max_examples=50):
    """Analyze false positives and false negatives for rare classes."""
    results = {}

    for cls_name in RARE_CLASSES:
        cls_idx = TARGET_CLASSES.index(cls_name)
        thresh = thresholds.get(cls_name, 0.5)

        cls_probs = test_probs[:, cls_idx]
        cls_true = test_labels[:, cls_idx]
        cls_pred = (cls_probs >= thresh).astype(int)

        # Find FP and FN indices
        fp_mask = (cls_pred == 1) & (cls_true == 0)
        fn_mask = (cls_pred == 0) & (cls_true == 1)

        fp_indices = np.where(fp_mask)[0]
        fn_indices = np.where(fn_mask)[0]

        # Limit examples
        fp_indices = fp_indices[:max_examples]
        fn_indices = fn_indices[:max_examples]

        fp_examples = []
        fn_examples = []

        for idx in fp_indices:
            text = test_df.iloc[idx]["text"]
            true_row = test_labels[idx]
            pred_row = (test_probs[idx] >= np.array([thresholds.get(c, 0.5) for c in TARGET_CLASSES])).astype(int)
            confused_cls, confused_prob = get_confused_class(test_probs[idx], cls_idx)

            bucket = classify_error_bucket(
                text, true_row, pred_row, cls_name,
                cls_probs[idx], confused_cls, confused_prob
            )

            true_labels_list = [TARGET_CLASSES[i] for i in range(7) if true_row[i] == 1]
            pred_labels_list = [TARGET_CLASSES[i] for i in range(7) if pred_row[i] == 1]

            fp_examples.append({
                "text": text,
                "true_labels": true_labels_list,
                "pred_labels": pred_labels_list,
                "cls_prob": round(float(cls_probs[idx]), 4),
                "confused_class": confused_cls,
                "confused_prob": round(float(confused_prob), 4),
                "bucket": bucket,
                "error_type": "false_positive",
            })

        for idx in fn_indices:
            text = test_df.iloc[idx]["text"]
            true_row = test_labels[idx]
            pred_row = (test_probs[idx] >= np.array([thresholds.get(c, 0.5) for c in TARGET_CLASSES])).astype(int)
            confused_cls, confused_prob = get_confused_class(test_probs[idx], cls_idx)

            bucket = classify_error_bucket(
                text, true_row, pred_row, cls_name,
                cls_probs[idx], confused_cls, confused_prob
            )

            true_labels_list = [TARGET_CLASSES[i] for i in range(7) if true_row[i] == 1]
            pred_labels_list = [TARGET_CLASSES[i] for i in range(7) if pred_row[i] == 1]

            fn_examples.append({
                "text": text,
                "true_labels": true_labels_list,
                "pred_labels": pred_labels_list,
                "cls_prob": round(float(cls_probs[idx]), 4),
                "confused_class": confused_cls,
                "confused_prob": round(float(confused_prob), 4),
                "bucket": bucket,
                "error_type": "false_negative",
            })

        results[cls_name] = {
            "total_fp": int(fp_mask.sum()),
            "total_fn": int(fn_mask.sum()),
            "total_tp": int(((cls_pred == 1) & (cls_true == 1)).sum()),
            "total_tn": int(((cls_pred == 0) & (cls_true == 0)).sum()),
            "support": int(cls_true.sum()),
            "threshold": thresh,
            "fp_examples": fp_examples,
            "fn_examples": fn_examples,
        }

    return results


def generate_markdown_report(results, output_path):
    """Generate the rare_class_error_report.md."""
    lines = []
    lines.append("# Rare-Class Error Analysis Report\n")
    lines.append("**Phase 2 — MoodMax Model Improvement**\n")
    lines.append("This report analyzes false positives (FP) and false negatives (FN) for the three")
    lines.append("weakest emotion classes: **disgust**, **fear**, and **sadness**.\n")
    lines.append("The goal is to determine whether the weakness is caused by:\n")
    lines.append("1. **Data volume** — not enough examples to learn the pattern")
    lines.append("2. **Label ambiguity** — GoEmotions crowd-worker labels are inconsistent")
    lines.append("3. **Model capacity** — the model fails on clear, unambiguous examples\n")
    lines.append("---\n")

    # Summary table
    lines.append("## Summary\n")
    lines.append("| Class | Support | TP | FP | FN | Threshold |")
    lines.append("|---|---|---|---|---|---|")
    for cls_name in RARE_CLASSES:
        r = results[cls_name]
        lines.append(f"| **{cls_name}** | {r['support']} | {r['total_tp']} | {r['total_fp']} | {r['total_fn']} | {r['threshold']:.2f} |")
    lines.append("")

    # Bucket distribution
    lines.append("## Error Bucket Distribution\n")
    lines.append("| Class | Error Type | Ambiguous/Multi-label | Annotation Noise | Clear Model Miss | Too Short |")
    lines.append("|---|---|---|---|---|---|")

    overall_ambiguous = 0
    overall_total = 0

    for cls_name in RARE_CLASSES:
        r = results[cls_name]
        for error_type, examples in [("FP", r["fp_examples"]), ("FN", r["fn_examples"])]:
            buckets = Counter(ex["bucket"] for ex in examples)
            total = len(examples)
            overall_total += total
            overall_ambiguous += buckets.get("ambiguous_multilabel", 0) + buckets.get("annotation_noise", 0)

            lines.append(
                f"| **{cls_name}** | {error_type} (n={total}) | "
                f"{buckets.get('ambiguous_multilabel', 0)} ({buckets.get('ambiguous_multilabel', 0)/max(total,1)*100:.0f}%) | "
                f"{buckets.get('annotation_noise', 0)} ({buckets.get('annotation_noise', 0)/max(total,1)*100:.0f}%) | "
                f"{buckets.get('clear_model_miss', 0)} ({buckets.get('clear_model_miss', 0)/max(total,1)*100:.0f}%) | "
                f"{buckets.get('too_short', 0)} ({buckets.get('too_short', 0)/max(total,1)*100:.0f}%) |"
            )
    lines.append("")

    # Per-class detailed examples
    for cls_name in RARE_CLASSES:
        r = results[cls_name]
        lines.append(f"---\n")
        lines.append(f"## {cls_name.title()} — Detailed Error Examples\n")
        lines.append(f"**Support:** {r['support']} | **Threshold:** {r['threshold']:.2f} | "
                     f"**TP:** {r['total_tp']} | **FP:** {r['total_fp']} | **FN:** {r['total_fn']}\n")

        for error_type, examples in [("False Positives", r["fp_examples"]), ("False Negatives", r["fn_examples"])]:
            if not examples:
                continue

            lines.append(f"### {error_type} (showing {len(examples)} of {r['total_fp'] if 'Positive' in error_type else r['total_fn']})\n")
            lines.append(f"| # | Bucket | Text | True Labels | Pred Labels | {cls_name} Prob | Confused With |")
            lines.append(f"|---|---|---|---|---|---|---|")

            for i, ex in enumerate(examples[:25], 1):  # Limit to 25 in the table for readability
                text_short = ex["text"][:80].replace("|", "\\|").replace("\n", " ")
                if len(ex["text"]) > 80:
                    text_short += "..."
                lines.append(
                    f"| {i} | {ex['bucket']} | {text_short} | "
                    f"{', '.join(ex['true_labels'])} | {', '.join(ex['pred_labels'])} | "
                    f"{ex['cls_prob']:.4f} | {ex['confused_class']} ({ex['confused_prob']:.4f}) |"
                )
            lines.append("")

    # Conclusion
    lines.append("---\n")
    lines.append("## Conclusion & Phase 4 Recommendation\n")

    ambiguous_pct = (overall_ambiguous / max(overall_total, 1)) * 100

    if ambiguous_pct > 40:
        lines.append(f"> **{ambiguous_pct:.0f}%** of errors fall into the ambiguous/annotation-noise buckets.\n")
        lines.append("> ⚠️ **Recommendation:** More training data of the *same kind* is unlikely to help significantly.")
        lines.append("> The primary bottleneck is label ambiguity in the GoEmotions dataset, not model capacity.")
        lines.append("> Phase 4 (retraining) may yield diminishing returns. Focus on Phase 1 + 3 optimizations")
        lines.append("> and consider curating higher-quality external data if Phase 4 is attempted.\n")
        lines.append(f"**Phase 4 go/no-go: PROCEED WITH CAUTION** — retraining should focus on curated external data,")
        lines.append(f"not just oversampling the existing noisy examples.")
    else:
        lines.append(f"> **{ambiguous_pct:.0f}%** of errors fall into the ambiguous/annotation-noise buckets.\n")
        lines.append(f"> ✓ **Recommendation:** A meaningful share of errors are genuine model misses.")
        lines.append(f"> Phase 4 (retraining with targeted oversampling and hard negatives) is justified.")
        lines.append(f"> The model has room to improve on clear, unambiguous examples.\n")
        lines.append(f"**Phase 4 go/no-go: PROCEED** — retraining with oversampling and targeted hard negatives is warranted.")

    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[OK] Error analysis report saved to {output_path}")
    return ambiguous_pct


def main():
    parser = argparse.ArgumentParser(description="Phase 2 — Rare-class error analysis for MoodMax")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Path to the trained model directory")
    parser.add_argument("--max_examples", type=int, default=50,
                        help="Maximum error examples to analyze per class per error type (default: 50)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Inference batch size if re-extracting probabilities (default: 16)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    model_dir = Path(args.model_dir) if args.model_dir else project_root / "models" / "emotion-distilbert-multi"
    data_dir = project_root / "data" / "processed"
    output_path = project_root / "analysis" / "rare_class_error_report.md"

    print("=" * 70)
    print("MoodMax Phase 2 — Rare-Class Error Analysis")
    print("=" * 70)

    # Load test data
    test_path = data_dir / "test_collapsed.csv"
    if not test_path.exists():
        print(f"ERROR: Test data not found at {test_path}")
        sys.exit(1)

    test_df = pd.read_csv(test_path)
    test_labels = test_df[TARGET_CLASSES].values
    print(f"Test set: {len(test_df):,} rows")

    # Load thresholds (from Phase 1)
    thresholds_path = model_dir / "thresholds.json"
    if thresholds_path.exists():
        with open(thresholds_path) as f:
            thresholds_data = json.load(f)
            thresholds = thresholds_data.get("thresholds", {cls: 0.5 for cls in TARGET_CLASSES})
        print(f"Loaded per-class thresholds from {thresholds_path}")
    else:
        thresholds = {cls: 0.5 for cls in TARGET_CLASSES}
        print("No thresholds.json found — using flat τ=0.5")

    # Load or extract test probabilities
    test_probs_path = model_dir / "test_probs.npy"
    if test_probs_path.exists():
        test_probs = np.load(test_probs_path)
        print(f"Loaded pre-computed test probabilities from {test_probs_path}")
    else:
        print("No pre-computed test probabilities found. Running inference...")
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
        model.eval()

        temperature = 1.0
        calib_path = model_dir / "calibration.json"
        if calib_path.exists():
            with open(calib_path) as f:
                calib = json.load(f)
                temperature = float(calib.get("temperature", 1.0))

        texts = test_df["text"].fillna("").tolist()
        all_probs = []
        for i in range(0, len(texts), args.batch_size):
            batch = texts[i:i + args.batch_size]
            inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                scaled = outputs.logits / temperature
                probs = torch.sigmoid(scaled).cpu().numpy()
                all_probs.append(probs)
            if (i // args.batch_size) % 50 == 0:
                print(f"  Progress: {min(i + args.batch_size, len(texts))}/{len(texts)}")

        test_probs = np.vstack(all_probs)
        np.save(test_probs_path, test_probs)
        print(f"  Saved test probabilities to {test_probs_path}")

    # Run error analysis
    print("\nAnalyzing errors for rare classes...")
    results = analyze_errors(test_df, test_probs, test_labels, thresholds, args.max_examples)

    # Print summary
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS SUMMARY")
    print("=" * 70)
    for cls_name in RARE_CLASSES:
        r = results[cls_name]
        all_examples = r["fp_examples"] + r["fn_examples"]
        buckets = Counter(ex["bucket"] for ex in all_examples)
        total = len(all_examples)
        print(f"\n  {cls_name.upper()} (support={r['support']}, τ={r['threshold']:.2f}):")
        print(f"    FP={r['total_fp']}  FN={r['total_fn']}  TP={r['total_tp']}  TN={r['total_tn']}")
        for bucket_name in ["ambiguous_multilabel", "annotation_noise", "clear_model_miss", "too_short"]:
            count = buckets.get(bucket_name, 0)
            pct = count / max(total, 1) * 100
            print(f"    {bucket_name:<25} {count:>4} ({pct:>5.1f}%)")

    # Generate markdown report
    ambiguous_pct = generate_markdown_report(results, output_path)

    print("\n" + "=" * 70)
    if ambiguous_pct > 40:
        print(f"⚠ {ambiguous_pct:.0f}% of errors are ambiguous/noise — retraining may have diminishing returns.")
    else:
        print(f"✓ {ambiguous_pct:.0f}% of errors are ambiguous/noise — Phase 4 retraining is warranted.")
    print("=" * 70)


if __name__ == "__main__":
    main()
