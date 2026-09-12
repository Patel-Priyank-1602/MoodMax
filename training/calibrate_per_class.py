"""
Phase 3 — Per-Class Temperature Calibration.

Replace the single global temperature T=1.1724 with per-class temperatures,
since fear/disgust/sadness have very different base rates than joy/neutral
and a shared T likely under- or over-corrects them.

Usage:
  python calibrate_per_class.py
  python calibrate_per_class.py --model_dir ../models/emotion-distilbert-multi
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


TARGET_CLASSES = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]


class EvalDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


def compute_per_class_ece(probs, labels, n_bins=10):
    """Compute ECE per class (binary calibration)."""
    per_class_ece = {}
    for cls_idx, cls_name in enumerate(TARGET_CLASSES):
        cls_probs = probs[:, cls_idx]
        cls_labels = labels[:, cls_idx]

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (cls_probs > bin_lower) & (cls_probs <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if np.sum(in_bin) > 0:
                accuracy_in_bin = np.mean(cls_labels[in_bin])
                avg_confidence_in_bin = np.mean(cls_probs[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        per_class_ece[cls_name] = round(float(ece), 6)

    return per_class_ece


def extract_logits(model, tokenizer, val_df, device, batch_size=16):
    """Extract raw logits from the validation set."""
    texts = val_df["text"].fillna("").tolist()
    labels = val_df[TARGET_CLASSES].values

    encodings = tokenizer(
        texts, truncation=True, padding=True, max_length=128, return_tensors="pt"
    )
    dataset = EvalDataset(encodings, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_logits = []
    all_labels = []

    print("Extracting validation logits...")
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.cpu())
            all_labels.append(target)

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def fit_per_class_temperature(logits, labels, device):
    """
    Fit a separate temperature parameter per class by minimizing
    per-class BCE loss independently.
    """
    per_class_temps = {}

    for cls_idx, cls_name in enumerate(TARGET_CLASSES):
        cls_logits = logits[:, cls_idx:cls_idx+1].to(device)
        cls_labels = labels[:, cls_idx:cls_idx+1].to(device)

        # Initialize temperature at 1.0
        temperature = nn.Parameter(torch.ones(1, device=device))
        optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=100)
        loss_fct = nn.BCEWithLogitsLoss()

        def closure():
            optimizer.zero_grad()
            scaled = cls_logits / temperature
            loss = loss_fct(scaled, cls_labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        # Clamp to reasonable range
        with torch.no_grad():
            temperature.clamp_(min=0.1, max=5.0)

        per_class_temps[cls_name] = round(float(temperature.item()), 4)

    return per_class_temps


def main():
    parser = argparse.ArgumentParser(description="Phase 3 — Per-class temperature calibration")
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    model_dir = Path(args.model_dir) if args.model_dir else project_root / "models" / "emotion-distilbert-multi"
    val_path = project_root / "data" / "processed" / "val_collapsed.csv"

    print("=" * 70)
    print("MoodMax Phase 3 — Per-Class Temperature Calibration")
    print("=" * 70)

    if not model_dir.exists():
        print(f"ERROR: Model not found at {model_dir}")
        sys.exit(1)
    if not val_path.exists():
        print(f"ERROR: Validation data not found at {val_path}")
        sys.exit(1)

    val_df = pd.read_csv(val_path)
    val_labels_np = val_df[TARGET_CLASSES].values
    print(f"Validation set: {len(val_df):,} rows")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)

    # ──────────────────────────────────────
    # Step 1: Extract raw logits
    # ──────────────────────────────────────
    logits, labels = extract_logits(model, tokenizer, val_df, device, args.batch_size)

    # ──────────────────────────────────────
    # Step 2: Compute uncalibrated ECE (T=1.0)
    # ──────────────────────────────────────
    uncalib_probs = torch.sigmoid(logits).numpy()
    ece_uncalibrated = compute_per_class_ece(uncalib_probs, val_labels_np)

    print("\n" + "-" * 70)
    print("UNCALIBRATED PER-CLASS ECE (T = 1.0):")
    for cls_name in TARGET_CLASSES:
        print(f"  {cls_name:<12} ECE = {ece_uncalibrated[cls_name]:.6f}")
    print("-" * 70)

    # ──────────────────────────────────────
    # Step 3: Compute global T=1.1724 ECE (current baseline)
    # ──────────────────────────────────────
    global_temp = 1.1724
    calib_path = model_dir / "calibration.json"
    if calib_path.exists():
        with open(calib_path) as f:
            calib = json.load(f)
            global_temp = float(calib.get("temperature", 1.1724))

    global_probs = torch.sigmoid(logits / global_temp).numpy()
    ece_global = compute_per_class_ece(global_probs, val_labels_np)

    print(f"\nGLOBAL CALIBRATION PER-CLASS ECE (T = {global_temp:.4f}):")
    for cls_name in TARGET_CLASSES:
        print(f"  {cls_name:<12} ECE = {ece_global[cls_name]:.6f}")
    print("-" * 70)

    # ──────────────────────────────────────
    # Step 4: Fit per-class temperatures
    # ──────────────────────────────────────
    print("\nFitting per-class temperatures...")
    per_class_temps = fit_per_class_temperature(logits, labels, device)

    print("\nOptimal per-class temperatures:")
    for cls_name in TARGET_CLASSES:
        t = per_class_temps[cls_name]
        direction = "sharpens" if t < 1.0 else "softens"
        print(f"  {cls_name:<12} T = {t:.4f}  ({direction} probabilities)")

    # ──────────────────────────────────────
    # Step 5: Compute per-class calibrated ECE
    # ──────────────────────────────────────
    per_class_probs = np.zeros_like(uncalib_probs)
    for cls_idx, cls_name in enumerate(TARGET_CLASSES):
        t = per_class_temps[cls_name]
        per_class_probs[:, cls_idx] = torch.sigmoid(logits[:, cls_idx] / t).numpy()

    ece_per_class = compute_per_class_ece(per_class_probs, val_labels_np)

    # ──────────────────────────────────────
    # Step 6: Comparison table
    # ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("PER-CLASS CALIBRATION COMPARISON")
    print("=" * 70)
    print(f"  {'Class':<12} {'ECE (T=1.0)':>12} {'ECE (Global)':>14} {'ECE (Per-Class)':>16} {'Per-Class T':>12} {'Improvement':>12}")
    print(f"  {'─' * 80}")

    improvements = {}
    regressions = {}
    rare_classes = ["disgust", "fear", "sadness"]
    strong_classes = ["joy", "neutral"]

    for cls_name in TARGET_CLASSES:
        ece_raw = ece_uncalibrated[cls_name]
        ece_g = ece_global[cls_name]
        ece_pc = ece_per_class[cls_name]
        t = per_class_temps[cls_name]

        # Improvement vs global calibration
        if ece_g > 0:
            improvement = ((ece_g - ece_pc) / ece_g) * 100
        else:
            improvement = 0.0

        improvements[cls_name] = improvement
        if improvement < 0:
            regressions[cls_name] = improvement

        sign = "+" if improvement >= 0 else ""
        marker = " 🟢" if improvement > 0 else (" 🔴" if improvement < 0 else "")
        print(f"  {cls_name:<12} {ece_raw:>12.6f} {ece_g:>14.6f} {ece_pc:>16.6f} {t:>12.4f} {sign}{improvement:>10.1f}%{marker}")

    # Overall sigmoid ECE
    overall_global_ece = np.mean([ece_global[c] for c in TARGET_CLASSES])
    overall_perclass_ece = np.mean([ece_per_class[c] for c in TARGET_CLASSES])
    print(f"\n  {'MEAN ECE':<12} {'':>12} {overall_global_ece:>14.6f} {overall_perclass_ece:>16.6f}")
    print("=" * 70)

    # ──────────────────────────────────────
    # Step 7: Accept/reject decision
    # ──────────────────────────────────────
    rare_improved = all(improvements.get(c, 0) > 0 for c in rare_classes)
    strong_regressed = any(improvements.get(c, 0) < -5 for c in strong_classes)  # Allow small regression

    print("\n" + "=" * 70)
    print("ACCEPTANCE CHECK")
    print("=" * 70)
    print(f"  Rare classes (disgust/fear/sadness) ECE improved: {'✓ YES' if rare_improved else '✗ NO'}")
    print(f"  Strong classes (joy/neutral) ECE regressed >5%:   {'✗ YES' if strong_regressed else '✓ NO'}")

    if rare_improved and not strong_regressed:
        decision = "ACCEPTED"
        print(f"\n  ✓ Phase 3 ACCEPTED — per-class calibration improves weak classes without regressing strong ones.")
    elif not rare_improved:
        decision = "REJECTED"
        print(f"\n  ✗ Phase 3 REJECTED — per-class calibration did NOT improve ECE for all rare classes.")
        print(f"    Reverting to global T={global_temp}.")
    else:
        decision = "PARTIAL"
        print(f"\n  ⚠ Phase 3 PARTIAL — rare classes improved but strong classes regressed significantly.")
        print(f"    Consider using a hybrid approach (per-class T for rare, global T for strong).")

    # ──────────────────────────────────────
    # Step 8: Save results
    # ──────────────────────────────────────
    result = {
        "decision": decision,
        "temperatures": per_class_temps,
        "ece_comparison": {
            cls: {
                "ece_uncalibrated": ece_uncalibrated[cls],
                "ece_global": ece_global[cls],
                "ece_per_class": ece_per_class[cls],
                "improvement_vs_global_pct": round(improvements[cls], 2),
            }
            for cls in TARGET_CLASSES
        },
    }

    output_path = model_dir / "calibration_per_class.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[OK] Per-class calibration saved to {output_path}")

    if decision == "REJECTED":
        print("\n[INFO] Since per-class calibration was rejected, the inference pipeline")
        print("       should continue using the global temperature from calibration.json.")
    print("=" * 70)


if __name__ == "__main__":
    main()
