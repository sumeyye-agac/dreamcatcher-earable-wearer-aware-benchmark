#!/usr/bin/env bash
set -e

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  DreamCatcher Audio Benchmark - SMOKE (fast sanity check)"
echo "════════════════════════════════════════════════════════════"
echo ""

echo "Tip: make sure you're logged in first:"
echo "  hf auth login"
echo ""

echo "== TinyCNN (smoke) =="
python -m src.training.train_baseline \
  --model tinycnn \
  --dataset_mode smoke \
  --epochs 1 \
  --batch_size 8 \
  --lr 1e-3 \
  --run_name tinycnn_smoke

echo ""
echo "== CRNN (smoke) =="
python -m src.training.train_baseline \
  --model crnn \
  --dataset_mode smoke \
  --epochs 1 \
  --batch_size 8 \
  --lr 1e-3 \
  --rnn_hidden 64 \
  --rnn_layers 1 \
  --run_name crnn_smoke

echo ""
echo "== CRNN + CBAM (smoke) =="
python -m src.training.train_baseline \
  --model crnn_cbam \
  --dataset_mode smoke \
  --epochs 1 \
  --batch_size 8 \
  --lr 1e-3 \
  --rnn_hidden 64 \
  --rnn_layers 1 \
  --cbam_reduction 8 \
  --cbam_sa_kernel 7 \
  --run_name crnn_cbam_smoke

