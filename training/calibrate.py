"""
Post-hoc Temperature Scaling Calibration for Emotion Classification Model.

Based on Guo et al. (2017) "On Calibration of Modern Neural Networks".

Learns a single scalar temperature T > 0 on the validation set to calibrate model logits:
  calibrated_logits = logits / T

Benefits:
- If T < 1.0: Sharpens diffuse 40-60% probabilities so dominant emotions (like 'joy')
  peak clearly instead of remaining tied with background noise.
- If T > 1.0: Softens overconfident false alarms (e.g. false fear/sadness predictions).
- Preserves the argmax (dominant emotion rank order does not flip), but aligns
  confidence values with true empirical probabilities.

Usage:
  python calibrate.py
  python calibrate.py --model_dir ../models/emotion-distilbert-multi
  python calibrate.py --objective bce     # Optimize for multi-label sigmoid probabilities
  python calibrate.py --objective softmax # Optimize for dominant emotion softmax confidence
"""

import argparse
import json
import os
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


class ModelWithTemperature(nn.Module):
    """A thin wrapper that applies temperature scaling to raw logits."""

    def __init__(self):
        super().__init__()
        # Initialize temperature at 1.0
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits):
        return logits / self.temperature


def compute_binary_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) for binary/multi-label probabilities."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = probs.size

    flat_probs = probs.flatten()
    flat_labels = labels.flatten()

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (flat_probs > bin_lower) & (flat_probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if np.sum(in_bin) > 0:
            accuracy_in_bin = np.mean(flat_labels[in_bin])
            avg_confidence_in_bin = np.mean(flat_probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def compute_multiclass_ece(probs: np.ndarray, dominant_labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) for dominant class softmax."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == dominant_labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            ece += (bin_size / n) * np.abs(conf_in_bin - acc_in_bin)

    return float(ece)


def extract_validation_logits(model, tokenizer, val_df, device, batch_size=16):
    """Run model over validation set and return raw logits and labels as tensors."""
    texts = val_df["text"].fillna("").tolist()
    labels = val_df[TARGET_CLASSES].values

    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )
    dataset = EvalDataset(encodings, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_logits = []
    all_labels = []

    print("Extracting validation set logits...")
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.cpu())
            all_labels.append(target)

    logits_tensor = torch.cat(all_logits, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    return logits_tensor, labels_tensor


def calibrate(args):
    project_root = Path(__file__).resolve().parent.parent
    model_dir = Path(args.model_dir) if args.model_dir else project_root / "models" / "emotion-distilbert-multi"
    val_path = project_root / "data" / "processed" / "val_collapsed.csv"

    print("=" * 70)
    print("Post-Hoc Temperature Scaling Calibration")
    print("=" * 70)

    if not model_dir.exists():
        print(f"ERROR: Model directory {model_dir} not found.")
        print("Please train the model first with: python training/train.py")
        sys.exit(1)

    if not val_path.exists():
        print(f"ERROR: Validation file {val_path} not found.")
        sys.exit(1)

    val_df = pd.read_csv(val_path)
    print(f"Loaded validation set: {len(val_df):,} examples")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {model_dir} onto {device}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)

    # 1. Extract raw logits on validation set
    logits, labels = extract_validation_logits(model, tokenizer, val_df, device, batch_size=args.batch_size)

    # Move tensors to working device
    logits = logits.to(device)
    labels = labels.to(device)

    # 2. Compute uncalibrated metrics
    uncalibrated_probs_sigmoid = torch.sigmoid(logits).cpu().numpy()
    ece_before_sigmoid = compute_binary_ece(uncalibrated_probs_sigmoid, labels.cpu().numpy())

    uncalibrated_probs_softmax = torch.softmax(logits, dim=-1).cpu().numpy()
    dominant_true = torch.argmax(labels, dim=-1).cpu().numpy()
    ece_before_softmax = compute_multiclass_ece(uncalibrated_probs_softmax, dominant_true)

    print("\n" + "-" * 70)
    print("UNCALIBRATED METRICS (T = 1.000):")
    print(f"  Multi-label Sigmoid ECE: {ece_before_sigmoid:.4f}")
    print(f"  Dominant Emotion Softmax ECE: {ece_before_softmax:.4f}")
    print("-" * 70)

    # 3. Optimize Temperature
    scaler = ModelWithTemperature().to(device)

    if args.objective == "bce":
        loss_fct = nn.BCEWithLogitsLoss()
        target = labels
    else:
        loss_fct = nn.CrossEntropyLoss()
        target = torch.argmax(labels, dim=-1)

    # Use L-BFGS for exact convex 1D optimization
    optimizer = optim.LBFGS([scaler.temperature], lr=0.01, max_iter=100)

    def eval_loss():
        optimizer.zero_grad()
        scaled_logits = scaler(logits)
        if args.objective == "bce":
            loss = loss_fct(scaled_logits, target)
        else:
            loss = loss_fct(scaled_logits, target)
        loss.backward()
        return loss

    optimizer.step(eval_loss)

    # Clamp temperature to avoid numerical extremes (e.g. 0.1 <= T <= 5.0)
    with torch.no_grad():
        scaler.temperature.clamp_(min=0.1, max=5.0)

    optimal_temperature = float(scaler.temperature.item())

    # 4. Compute calibrated metrics
    calibrated_logits = logits / optimal_temperature

    calibrated_probs_sigmoid = torch.sigmoid(calibrated_logits).cpu().numpy()
    ece_after_sigmoid = compute_binary_ece(calibrated_probs_sigmoid, labels.cpu().numpy())

    calibrated_probs_softmax = torch.softmax(calibrated_logits, dim=-1).cpu().numpy()
    ece_after_softmax = compute_multiclass_ece(calibrated_probs_softmax, dominant_true)

    print("\n" + "=" * 70)
    print("CALIBRATION COMPLETE!")
    print("=" * 70)
    print(f"  Learned Temperature (T): {optimal_temperature:.4f}")
    if optimal_temperature < 1.0:
        print("  [Note] T < 1.0: Model was UNDERCONFIDENT. Scaling will sharpen probabilities and reduce 40-60% ties.")
    else:
        print("  [Note] T > 1.0: Model was OVERCONFIDENT. Scaling will temper overconfidence.")

    print(f"\n  ECE Improvement (Objective: {args.objective.upper()}):")
    print(f"    Sigmoid ECE:  {ece_before_sigmoid:.4f} -> {ece_after_sigmoid:.4f} "
          f"({((ece_before_sigmoid - ece_after_sigmoid)/ece_before_sigmoid)*100:+.1f}%)")
    print(f"    Softmax ECE:  {ece_before_softmax:.4f} -> {ece_after_softmax:.4f} "
          f"({((ece_before_softmax - ece_after_softmax)/ece_before_softmax)*100:+.1f}%)")
    print("=" * 70)

    # 5. Save calibration config
    calib_result = {
        "temperature": round(optimal_temperature, 4),
        "objective": args.objective,
        "sigmoid_ece_before": round(ece_before_sigmoid, 4),
        "sigmoid_ece_after": round(ece_after_sigmoid, 4),
        "softmax_ece_before": round(ece_before_softmax, 4),
        "softmax_ece_after": round(ece_after_softmax, 4),
    }

    calib_path = model_dir / "calibration.json"
    with open(calib_path, "w") as f:
        json.dump(calib_result, f, indent=2)

    print(f"\nSaved calibration parameters to: {calib_path}")
    print("\nHow to use in inference pipeline:")
    print("  calibrated_logits = raw_logits / temperature")
    print("  probs = torch.sigmoid(calibrated_logits)")


def main():
    parser = argparse.ArgumentParser(description="Calibrate emotion classification model with temperature scaling")
    parser.add_argument("--model_dir", type=str, default=None, help="Directory containing the fine-tuned model")
    parser.add_argument("--batch_size", type=int, default=16, help="Validation inference batch size")
    parser.add_argument("--objective", type=str, default="bce", choices=["bce", "softmax"],
                        help="Calibration loss objective: 'bce' (multi-label sigmoid) or 'softmax' (dominant emotion)")
    args = parser.parse_args()
    calibrate(args)


if __name__ == "__main__":
    main()
