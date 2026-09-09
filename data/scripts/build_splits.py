"""
Master pipeline script: orchestrates the full data preparation workflow.

Steps:
  1. Download GoEmotions dataset
  2. Collapse labels (27 → 7)
  3. (Optional) Translate and augment for multilingual support

Usage:
  python build_splits.py                    # Steps 1-2 only (fast)
  python build_splits.py --with-translation # Steps 1-3 (slower, downloads ~900MB of models)
"""

import argparse
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from download_goemotions import download_goemotions
from collapse_labels import collapse_labels


def main():
    parser = argparse.ArgumentParser(
        description="Build data splits for MoodMax emotion model training"
    )
    parser.add_argument(
        "--with-translation",
        action="store_true",
        help="Include multilingual augmentation (downloads ~900MB of translation models)"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip dataset download (use existing raw data)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MoodMax Data Pipeline")
    print("=" * 60)

    # Step 1: Download
    if not args.skip_download:
        print("\n[Step 1/3] Downloading GoEmotions dataset...")
        print("-" * 40)
        download_goemotions()
    else:
        print("\n[Step 1/3] Skipping download (--skip-download)")

    # Step 2: Collapse labels
    print("\n[Step 2/3] Collapsing labels (27 → 7 classes)...")
    print("-" * 40)
    collapse_labels()

    # Step 3: Translation augmentation
    if args.with_translation:
        print("\n[Step 3/3] Translating for multilingual augmentation...")
        print("-" * 40)
        from translate_augment import augment_with_translations
        augment_with_translations()
    else:
        print("\n[Step 3/3] Skipping translation (use --with-translation to enable)")

    print("\n" + "=" * 60)
    print("✓ Data pipeline complete!")
    print("=" * 60)

    processed_dir = Path(__file__).resolve().parent.parent / "processed"
    print(f"\nOutput files in: {processed_dir}")
    for f in sorted(processed_dir.glob("*")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
