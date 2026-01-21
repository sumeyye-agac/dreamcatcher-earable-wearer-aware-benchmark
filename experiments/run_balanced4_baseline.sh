#!/usr/bin/env bash
set -e

source dream-env/bin/activate

echo "========================================"
echo "Balanced 4-Class Baseline Experiments"
echo "Classes: quiet, breathe, non_wearer, snore"
echo "========================================"
echo ""

echo "== Balanced 4-Class: CRNN Baseline =="
python3 -m src.training.train_baseline_balanced4 \
  --model crnn \
  --epochs 30 \
  --batch_size 16 \
  --lr 1e-3 \
  --early_stop_patience 5 \
  --early_stop_min_delta 0.001 \
  --rnn_hidden 64 \
  --rnn_layers 1 \
  --run_name crnn_balanced4_baseline

echo ""
echo "== Balanced 4-Class: CRNN+CBAM Baseline =="
python3 -m src.training.train_baseline_balanced4 \
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
  --run_name crnn_cbam_balanced4_baseline

echo ""
echo "=========================================="
echo "Balanced 4-Class Baseline Complete!"
echo "=========================================="
