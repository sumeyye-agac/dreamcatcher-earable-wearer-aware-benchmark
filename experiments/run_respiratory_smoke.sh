#!/usr/bin/env bash
set -e

# Activate virtual environment with all dependencies
source dream-env/bin/activate

echo "========================================"
echo "Respiratory Subset Smoke Test"
echo "Classes: breathe, cough, snore (3-class)"
echo "Quick test with smoke dataset"
echo "========================================"
echo ""

echo "== Respiratory Smoke Test: CRNN =="
python3 -m src.training.train_baseline_respiratory \
  --model crnn \
  --epochs 2 \
  --batch_size 16 \
  --lr 1e-3 \
  --early_stop_patience 5 \
  --early_stop_min_delta 0.001 \
  --rnn_hidden 64 \
  --rnn_layers 1 \
  --dataset_mode full \
  --max_samples 512 \
  --run_name crnn_respiratory_smoke

echo ""
echo "=========================================="
echo "Smoke Test Complete!"
echo "=========================================="
