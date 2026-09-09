"""
Multilingual augmentation via machine translation.
Takes a stratified sample from the collapsed training set and translates
to Hindi, Spanish, and French using Helsinki-NLP MarianMT models.

Models used (auto-downloaded from HuggingFace):
  - Helsinki-NLP/opus-mt-en-hi  (~300MB)
  - Helsinki-NLP/opus-mt-en-es  (~300MB)
  - Helsinki-NLP/opus-mt-en-fr  (~300MB)

Output: train_multilingual_augmented.csv
"""

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

TARGET_CLASSES = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]

# Translation targets: (language code, HuggingFace model name)
TRANSLATION_TARGETS = [
    ("hi", "Helsinki-NLP/opus-mt-en-hi"),
    ("es", "Helsinki-NLP/opus-mt-en-es"),
    ("fr", "Helsinki-NLP/opus-mt-en-fr"),
]

SAMPLE_SIZE = 10000  # Total rows to sample from English training set
BATCH_SIZE = 32      # Translation batch size


def stratified_sample(df: pd.DataFrame, n: int, label_cols: list) -> pd.DataFrame:
    """Take a stratified sample balanced across label columns."""
    # For multi-label, we use the dominant label for stratification
    df = df.copy()
    df["_dominant"] = df[label_cols].idxmax(axis=1)

    sampled = df.groupby("_dominant", group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), n // len(label_cols)), random_state=42)
    )

    # If we didn't get enough, sample more randomly
    if len(sampled) < n:
        remaining = df.drop(sampled.index)
        extra = remaining.sample(n=min(n - len(sampled), len(remaining)), random_state=42)
        sampled = pd.concat([sampled, extra])

    sampled = sampled.drop(columns=["_dominant"])
    return sampled.head(n)


def translate_batch(texts: list, model, tokenizer, device="cpu") -> list:
    """Translate a batch of texts."""
    import torch

    inputs = tokenizer(texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=128).to(device)

    with torch.no_grad():
        translated = model.generate(**inputs, max_length=128)

    return tokenizer.batch_decode(translated, skip_special_tokens=True)


def augment_with_translations(processed_dir: str = None):
    """Create multilingual augmented training set."""
    try:
        from transformers import MarianMTModel, MarianTokenizer
        import torch
    except ImportError:
        print("ERROR: 'transformers' and 'torch' are required.")
        print("Install with: pip install transformers torch")
        sys.exit(1)

    base_path = Path(__file__).resolve().parent.parent
    if processed_dir is None:
        processed_dir = base_path / "processed"
    else:
        processed_dir = Path(processed_dir)

    train_path = processed_dir / "train_collapsed.csv"
    if not train_path.exists():
        print(f"ERROR: {train_path} not found. Run collapse_labels.py first.")
        sys.exit(1)

    print("Loading collapsed training data...")
    df = pd.read_csv(train_path)
    print(f"  Total rows: {len(df):,}")

    # Take stratified sample
    sample = stratified_sample(df, SAMPLE_SIZE, TARGET_CLASSES)
    print(f"  Stratified sample: {len(sample):,} rows")

    device = "cuda" if "torch" in sys.modules and __import__("torch").cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    all_translated = []

    for lang_code, model_name in TRANSLATION_TARGETS:
        print(f"\nTranslating to {lang_code} using {model_name}...")

        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(device)
        model.eval()

        texts = sample["text"].tolist()
        translated_texts = []

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            try:
                translated = translate_batch(batch, model, tokenizer, device)
                translated_texts.extend(translated)
            except Exception as e:
                print(f"  Warning: Translation failed for batch {i}: {e}")
                translated_texts.extend(batch)  # Keep original on failure

            if (i // BATCH_SIZE) % 10 == 0:
                progress = min(i + BATCH_SIZE, len(texts))
                print(f"  Progress: {progress}/{len(texts)}")

        # Create translated dataframe
        translated_df = sample.copy()
        translated_df["text"] = translated_texts
        translated_df["lang"] = lang_code
        all_translated.append(translated_df)

        print(f"  ✓ {lang_code}: {len(translated_df):,} rows translated")

        # Free memory
        del model, tokenizer
        if device == "cuda":
            __import__("torch").cuda.empty_cache()

    # Combine original + all translations
    augmented = pd.concat([df] + all_translated, ignore_index=True)

    output_path = processed_dir / "train_multilingual_augmented.csv"
    augmented.to_csv(output_path, index=False)

    print(f"\n✓ Augmented dataset saved to: {output_path}")
    print(f"  Total rows: {len(augmented):,}")
    print(f"  Language distribution:")
    for lang, count in augmented["lang"].value_counts().items():
        print(f"    {lang}: {count:,}")


if __name__ == "__main__":
    augment_with_translations()
