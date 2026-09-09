"""
Download GoEmotions dataset from HuggingFace.
Uses the 'simplified' config which has 28 emotion labels (27 + neutral).
Saves raw data to data/raw/go_emotions/
"""

import os
import sys
from pathlib import Path

def download_goemotions(output_dir: str = None):
    """Download GoEmotions simplified dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library not installed.")
        print("Install with: pip install datasets")
        sys.exit(1)

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "raw" / "go_emotions"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading GoEmotions (simplified) from HuggingFace...")
    dataset = load_dataset("google-research-datasets/go_emotions", "simplified")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # Save each split as CSV
    for split_name in ["train", "validation", "test"]:
        split = dataset[split_name]
        csv_path = output_dir / f"{split_name}.csv"
        split.to_csv(str(csv_path), index=False)
        print(f"  [OK] {split_name}: {len(split)} rows -> {csv_path}")

    print(f"\nDone! Raw data saved to: {output_dir}")
    print(f"  Train:      {len(dataset['train']):,} rows")
    print(f"  Validation: {len(dataset['validation']):,} rows")
    print(f"  Test:       {len(dataset['test']):,} rows")

    return dataset


if __name__ == "__main__":
    download_goemotions()
