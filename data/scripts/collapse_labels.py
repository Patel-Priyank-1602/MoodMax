"""
Collapse GoEmotions 27 fine-grained labels → 7 target emotion classes.
Also derives sentiment labels from emotion labels.

Target classes:
  joy, sadness, anger, fear, surprise, disgust, neutral

Sentiment mapping:
  joy → Positive
  sadness, anger, fear, disgust → Negative
  surprise, neutral → Neutral
"""

import os
import sys
import json
import ast
from pathlib import Path

import pandas as pd
import numpy as np


# GoEmotions simplified label list (28 labels, 0-indexed)
GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]

# Mapping: target class → list of GoEmotions label indices
COLLAPSE_MAP = {
    "joy": ["admiration", "amusement", "excitement", "approval",
            "gratitude", "joy", "love", "optimism", "pride", "relief"],
    "sadness": ["sadness", "disappointment", "grief", "remorse"],
    "anger": ["anger", "annoyance", "disapproval"],
    "fear": ["fear", "nervousness"],
    "surprise": ["surprise", "realization", "curiosity", "confusion"],
    "disgust": ["disgust", "embarrassment"],
    "neutral": ["neutral", "desire", "caring"],
}

# Sentiment derived from dominant emotion
EMOTION_TO_SENTIMENT = {
    "joy": "Positive",
    "sadness": "Negative",
    "anger": "Negative",
    "fear": "Negative",
    "disgust": "Negative",
    "surprise": "Neutral",
    "neutral": "Neutral",
}

TARGET_CLASSES = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]


def parse_labels(label_field):
    """Parse the labels field from GoEmotions CSV.
    
    Can be a string representation of a list like '[0, 17]' or already a list.
    """
    if isinstance(label_field, list):
        return label_field
    if isinstance(label_field, str):
        try:
            return ast.literal_eval(label_field)
        except (ValueError, SyntaxError):
            return []
    if isinstance(label_field, (int, float)):
        return [int(label_field)]
    return []


def collapse_labels(raw_dir: str = None, output_dir: str = None):
    """Collapse GoEmotions labels and save processed CSVs."""

    base_path = Path(__file__).resolve().parent.parent
    if raw_dir is None:
        raw_dir = base_path / "raw" / "go_emotions"
    else:
        raw_dir = Path(raw_dir)

    if output_dir is None:
        output_dir = base_path / "processed"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build index lookup: GoEmotions label name → index
    label_to_idx = {name: i for i, name in enumerate(GOEMOTIONS_LABELS)}

    # Build collapse index map: target class → set of GoEmotions indices
    collapse_idx_map = {}
    for target, source_labels in COLLAPSE_MAP.items():
        collapse_idx_map[target] = {label_to_idx[lbl] for lbl in source_labels}

    for split_name, filename in [("train", "train.csv"),
                                  ("val", "validation.csv"),
                                  ("test", "test.csv")]:
        csv_path = raw_dir / filename
        if not csv_path.exists():
            print(f"  ✗ {csv_path} not found — run download_goemotions.py first")
            continue

        print(f"Processing {split_name}...")
        df = pd.read_csv(csv_path)

        rows = []
        for _, row in df.iterrows():
            original_labels = parse_labels(row["labels"])
            original_label_set = set(original_labels)

            # Compute collapsed multi-label vector
            collapsed = {}
            for target in TARGET_CLASSES:
                # 1 if any of the source labels are present
                collapsed[target] = 1 if original_label_set & collapse_idx_map[target] else 0

            # Derive sentiment from dominant emotion
            # Priority: if any negative emotion → Negative; if joy → Positive; else Neutral
            if any(collapsed[e] for e in ["sadness", "anger", "fear", "disgust"]):
                sentiment = "Negative"
            elif collapsed["joy"]:
                sentiment = "Positive"
            else:
                sentiment = "Neutral"

            rows.append({
                "id": row.get("id", row.name),
                "text": row["text"],
                "lang": "en",
                **collapsed,
                "sentiment": sentiment,
            })

        out_df = pd.DataFrame(rows)
        out_path = output_dir / f"{split_name}_collapsed.csv"
        out_df.to_csv(out_path, index=False)
        print(f"  [OK] {split_name}: {len(out_df)} rows -> {out_path}")

        # Print class distribution
        print(f"    Emotion distribution:")
        for cls in TARGET_CLASSES:
            count = out_df[cls].sum()
            pct = count / len(out_df) * 100
            print(f"      {cls}: {count:,} ({pct:.1f}%)")
        print(f"    Sentiment distribution:")
        for s in ["Positive", "Negative", "Neutral"]:
            count = (out_df["sentiment"] == s).sum()
            pct = count / len(out_df) * 100
            print(f"      {s}: {count:,} ({pct:.1f}%)")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # Save label map
    label_map = {cls: i for i, cls in enumerate(TARGET_CLASSES)}
    label_map_path = output_dir / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"\n[OK] Label map saved to {label_map_path}")

    print("\nDone! Collapsed datasets saved to:", output_dir)


if __name__ == "__main__":
    collapse_labels()
