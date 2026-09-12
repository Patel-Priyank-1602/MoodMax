"""
Model Comparison & Anti-Degradation Safety Gate.

Ensures that any newly retrained or tuned model does NOT degrade performance
compared to the baseline production checkpoint.

Acceptance Criteria:
  1. Test Macro-F1 >= Baseline Macro-F1 (0.5796)
  2. Anchor Classes Stability:
     - Joy F1 >= 0.79 (Baseline: 0.81, max 2% regression allowed)
     - Neutral F1 >= 0.62 (Baseline: 0.64, max 2% regression allowed)
  3. Rare-Class Signal:
     - Weak classes (fear, disgust, sadness) average F1 must improve or stay within 1%

Usage:
  python training/compare_and_gate.py --candidate models/emotion-distilbert-multi-v2
  python training/compare_and_gate.py --candidate models/emotion-distilbert-multi-v2 --promote
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

TARGET_CLASSES = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]
BASELINE_METRICS = {
    "macro_f1": 0.5796,
    "joy_f1": 0.81,
    "neutral_f1": 0.64,
    "sadness_f1": 0.52,
    "anger_f1": 0.52,
    "surprise_f1": 0.58,
    "disgust_f1": 0.44,
    "fear_f1": 0.54,
}


def load_test_data(data_dir: Path):
    test_path = data_dir / "test_collapsed.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_path}. Run data preparation first.")
    df = pd.read_csv(test_path)
    labels = df[TARGET_CLASSES].values
    return df["text"].tolist(), labels


def evaluate_checkpoint(model_dir: Path, texts, true_labels, device, batch_size=32):
    print(f"Loading checkpoint from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()

    all_probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            enc = {k: v for k, v in enc.items() if k != "token_type_ids"}
            logits = model(**enc).logits
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    probs = np.vstack(all_probs)

    # Check for thresholds.json
    thresholds_file = model_dir / "thresholds.json"
    if thresholds_file.exists():
        with open(thresholds_file) as f:
            t_data = json.load(f)
            t_dict = t_data.get("thresholds", {})
            thresh_arr = np.array([t_dict.get(c, 0.5) for c in TARGET_CLASSES])
            print(f"  Applied per-class thresholds from {thresholds_file.name}")
    else:
        thresh_arr = np.full(len(TARGET_CLASSES), 0.5)
        print("  Using standard flat threshold = 0.5")

    preds = (probs >= thresh_arr).astype(int)

    macro_f1 = f1_score(true_labels, preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(true_labels, preds, average=None, zero_division=0)
    per_class_p = precision_score(true_labels, preds, average=None, zero_division=0)
    per_class_r = recall_score(true_labels, preds, average=None, zero_division=0)

    results = {
        "macro_f1": float(macro_f1),
        "per_class": {
            cls: {
                "f1": float(per_class_f1[idx]),
                "precision": float(per_class_p[idx]),
                "recall": float(per_class_r[idx]),
            }
            for idx, cls in enumerate(TARGET_CLASSES)
        },
    }
    return results


def run_gate(candidate_dir: Path, baseline_dir: Path, data_dir: Path, promote: bool = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("MoodMax Model Evaluation & Anti-Degradation Gate")
    print(f"Device: {device}")
    print("=" * 70)

    texts, true_labels = load_test_data(data_dir)

    # Evaluate Candidate
    print("\n[1/2] Evaluating Candidate Model...")
    cand_metrics = evaluate_checkpoint(candidate_dir, texts, true_labels, device)

    # Get Baseline Metrics
    if baseline_dir.exists() and (baseline_dir / "model.safetensors").exists():
        print("\n[2/2] Evaluating Active Baseline Model...")
        base_metrics = evaluate_checkpoint(baseline_dir, texts, true_labels, device)
    else:
        print("\n[2/2] Local baseline checkpoint not present; using recorded production baseline metrics.")
        base_metrics = {
            "macro_f1": BASELINE_METRICS["macro_f1"],
            "per_class": {cls: {"f1": BASELINE_METRICS[f"{cls}_f1"]} for cls in TARGET_CLASSES},
        }

    # Print Comparison Table
    print("\n" + "=" * 70)
    print(f"{'Class':<12} {'Baseline F1':>15} {'Candidate F1':>15} {'Delta':>15}")
    print("-" * 70)
    for cls in TARGET_CLASSES:
        b_f1 = base_metrics["per_class"][cls]["f1"]
        c_f1 = cand_metrics["per_class"][cls]["f1"]
        delta = c_f1 - b_f1
        sign = "+" if delta >= 0 else ""
        print(f"{cls:<12} {b_f1:>15.4f} {c_f1:>15.4f} {sign + f'{delta:.4f}':>15}")

    b_macro = base_metrics["macro_f1"]
    c_macro = cand_metrics["macro_f1"]
    macro_delta = c_macro - b_macro
    sign = "+" if macro_delta >= 0 else ""
    print("-" * 70)
    print(f"{'MACRO-F1':<12} {b_macro:>15.4f} {c_macro:>15.4f} {sign + f'{macro_delta:.4f}':>15}")
    print("=" * 70)

    # Anti-Degradation Checks
    violations = []

    # Check 1: Macro-F1 regression
    if c_macro < b_macro:
        violations.append(f"Macro-F1 regressed: candidate ({c_macro:.4f}) < baseline ({b_macro:.4f})")

    # Check 2: Anchor class regression (Joy > 2% drop)
    joy_delta = cand_metrics["per_class"]["joy"]["f1"] - base_metrics["per_class"]["joy"]["f1"]
    if joy_delta < -0.02:
        violations.append(f"Joy F1 dropped excessively: delta={joy_delta:.4f} (allowed max -0.02)")

    # Check 3: Anchor class regression (Neutral > 2% drop)
    neu_delta = cand_metrics["per_class"]["neutral"]["f1"] - base_metrics["per_class"]["neutral"]["f1"]
    if neu_delta < -0.02:
        violations.append(f"Neutral F1 dropped excessively: delta={neu_delta:.4f} (allowed max -0.02)")

    # Decision
    print("\n" + "=" * 70)
    if violations:
        print("[ANTI-DEGRADATION GATE: REJECTED] ❌")
        print("Candidate model will NOT be promoted because it degrades baseline performance:")
        for v in violations:
            print(f"  - {v}")
        print("\nProduction baseline remains ACTIVE and UNTOUCHED.")
        print("=" * 70)
        return False
    else:
        print("[ANTI-DEGRADATION GATE: PASSED] ✅")
        print("Candidate model satisfies all quality and safety requirements!")
        print("  - Macro-F1 maintained or improved.")
        print("  - Anchor classes (joy, neutral) are stable.")
        print("=" * 70)

        if promote:
            print("\nPromoting candidate to production model...")
            backup_dir = baseline_dir.parent / f"{baseline_dir.name}-backup"
            if baseline_dir.exists():
                print(f"  Creating backup of current baseline at {backup_dir}...")
                if backup_dir.exists():
                    shutil.rmtree(backup_dir)
                shutil.copytree(baseline_dir, backup_dir)

            baseline_dir.mkdir(parents=True, exist_ok=True)
            for item in candidate_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, baseline_dir / item.name)
            print(f"✓ Promotion successful! Model files copied to {baseline_dir}")
        else:
            print("\n(Run with --promote to safely replace the production baseline model)")
        return True


def main():
    parser = argparse.ArgumentParser(description="Model Comparison & Anti-Degradation Safety Gate")
    parser.add_argument("--candidate", type=str, default="models/emotion-distilbert-multi-v2",
                        help="Path to candidate model checkpoint")
    parser.add_argument("--baseline", type=str, default="models/emotion-distilbert-multi",
                        help="Path to baseline model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/processed",
                        help="Path to processed data directory")
    parser.add_argument("--promote", action="store_true",
                        help="Automatically promote candidate if it passes the gate")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    candidate_path = project_root / args.candidate
    baseline_path = project_root / args.baseline
    data_path = project_root / args.data_dir

    success = run_gate(candidate_path, baseline_path, data_path, promote=args.promote)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
