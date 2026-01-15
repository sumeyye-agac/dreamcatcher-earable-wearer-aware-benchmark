#!/usr/bin/env bash
set -e

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  DreamCatcher KD Benchmark - SMOKE (fast sanity check)"
echo "════════════════════════════════════════════════════════════"
echo ""

echo "Tip: make sure you're logged in first:"
echo "  hf auth login"
echo ""

echo "== RB-KD (smoke): student=CRNN =="
python -m src.training.train_kd \
  --student crnn \
  --dataset_mode smoke \
  --epochs 1 \
  --batch_size 4 \
  --lr 1e-3 \
  --alpha 0.7 \
  --tau 5 \
  --rnn_hidden 64 \
  --rnn_layers 1 \
  --teacher_name facebook/wav2vec2-base \
  --run_name crnn_rbkd_smoke

echo "== RB-KD-Att (smoke): student=CRNN+CBAM =="
python -m src.training.train_kd \
  --student crnn_cbam \
  --dataset_mode smoke \
  --epochs 1 \
  --batch_size 4 \
  --lr 1e-3 \
  --alpha 0.7 \
  --tau 5 \
  --rnn_hidden 64 \
  --rnn_layers 1 \
  --cbam_reduction 8 \
  --cbam_sa_kernel 7 \
  --teacher_name facebook/wav2vec2-base \
  --run_name crnn_cbam_rbkdatt_smoke

