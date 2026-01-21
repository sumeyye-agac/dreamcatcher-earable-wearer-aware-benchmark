#!/usr/bin/env bash
set -e

# Activate virtual environment with all dependencies
source dream-env/bin/activate

echo "========================================"
echo "Respiratory Subset Baseline Experiments"
echo "Classes: breathe, cough, snore (3-class)"
echo "========================================"
echo ""

echo "== Respiratory Subset: CRNN Baseline =="
python3 -m src.training.train_baseline_respiratory \
  --model crnn \
  --epochs 30 \
  --batch_size 16 \
  --lr 1e-3 \
  --early_stop_patience 5 \
  --early_stop_min_delta 0.001 \
  --rnn_hidden 64 \
  --rnn_layers 1 \
  --run_name crnn_respiratory_baseline

echo ""
echo "== Respiratory Subset: CRNN+CBAM Baseline =="
python3 -m src.training.train_baseline_respiratory \
  --model crnn_cbam \
  --epochs 30 \
  --batch_size 16 \
  --lr 1e-3 \
  --early_stop_patience 5 \
  --early_stop_min_delta 0.001 \
  --rnn_hidden 64 \
  --rnn_layers 1 \
  --cbam_reduction 8 \
  --cbam_sa_kernel 7 \
  --run_name crnn_cbam_respiratory_baseline

echo ""
echo "=========================================="
echo "Respiratory Baseline Experiments Complete!"
echo "=========================================="
