#!/usr/bin/env bash
set -e

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  DreamCatcher Audio Benchmark - Baseline Models"
echo "════════════════════════════════════════════════════════════"
echo ""

echo "== TinyCNN baseline =="
python -m src.training.train_baseline \
  --model tinycnn \
  --epochs 5 \
  --batch_size 16 \
  --lr 1e-3 \
  --run_name tinycnn_baseline

echo ""
echo "== CRNN baseline =="
python -m src.training.train_baseline \
  --model crnn \
  --epochs 5 \
  --batch_size 16 \
  --lr 1e-3 \
  --rnn_hidden 64 \
  --rnn_layers 1 \
  --run_name crnn_baseline

echo ""
echo "== CRNN + CBAM (attention) =="
python -m src.training.train_baseline \
  --model crnn_cbam \
  --epochs 5 \
  --batch_size 16 \
  --lr 1e-3 \
  --rnn_hidden 64 \
  --rnn_layers 1 \
  --cbam_reduction 8 \
  --cbam_sa_kernel 7 \
  --run_name crnn_cbam_att
