#!/bin/bash
# ============================================================
# MoodMax Model Pipeline on Google Colab (Tesla T4 GPU)
# Safe, end-to-end retraining & optimization with Anti-Degradation Gate
# ============================================================
set -e

echo "============================================================"
echo "MoodMax Model Improvement & Training Pipeline"
echo "============================================================"

# 1. GPU Check
echo -e "\n>>> [1/6] Checking GPU hardware..."
nvidia-smi

# 2. Dependencies
echo -e "\n>>> [2/6] Installing dependencies..."
pip install -q -r training/requirements-training.txt

# 3. Data Preparation
echo -e "\n>>> [3/6] Checking training data..."
if [ ! -f "data/processed/train_collapsed.csv" ]; then
    echo "Data not found. Downloading GoEmotions and building collapsed splits..."
    python data/scripts/download_goemotions.py
    python data/scripts/collapse_labels.py
else
    echo "✓ Processed datasets found in data/processed/"
fi

# 4. Phase 1: Threshold Sweep on current baseline
if [ -d "models/emotion-distilbert-multi" ] && [ -f "models/emotion-distilbert-multi/model.safetensors" ]; then
    echo -e "\n>>> [4/6] Optimizing per-class decision thresholds (Phase 1)..."
    python training/optimize_thresholds.py --eval_test
else
    echo -e "\n>>> [4/6] No local baseline weights found; proceeding directly to Phase 4 Retraining."
fi

# 5. Phase 4: Targeted Retraining with Rare-Class Oversampling
echo -e "\n>>> [5/6] Training Candidate Model (v2) with Rare-Class Oversampling..."
python training/train_v2.py --batch_size 32 --epochs 8 --oversample_factor 2.5

# 6. Safety Gate & Anti-Degradation Check
echo -e "\n>>> [6/6] Verifying Candidate against Anti-Degradation Gate..."
python training/compare_and_gate.py --candidate models/emotion-distilbert-multi-v2 --promote

echo -e "\n============================================================"
echo "Pipeline execution finished successfully!"
echo "============================================================"
