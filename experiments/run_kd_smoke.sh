#!/usr/bin/env bash
set -e

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  DreamCatcher KD Benchmark - SMOKE (fast sanity check)"
echo "════════════════════════════════════════════════════════════"
echo ""

if ! python3 -c "from huggingface_hub import get_token; import sys; sys.exit(0 if get_token() else 1)"; then
  echo "ERROR: HuggingFace token not found. DreamCatcher is a gated dataset."
  echo "Run: hf auth login"
  echo "Or set: export HF_TOKEN=\"hf_...\"; export HUGGINGFACE_HUB_TOKEN=\"$HF_TOKEN\""
  exit 1
fi

echo "Tip: make sure you're logged in first:"
echo "  hf auth login"
echo ""

echo "== RB-KD (smoke): student=CRNN =="
python3 -m src.training.train_kd \
  --student crnn \
  --dataset_mode smoke \
  --epochs 1 \
  --batch_size 4 \
  --max_samples 512 \
  --lr 1e-3 \
  --alpha 0.7 \
  --tau 5 \
  --rnn_hidden 64 \
  --rnn_layers 1 \
  --teacher_name facebook/wav2vec2-base \
  --run_name crnn_rbkd_smoke

echo "== RB-KD-Att (smoke): student=CRNN+CBAM =="
python3 -m src.training.train_kd \
  --student crnn_cbam \
  --dataset_mode smoke \
  --epochs 1 \
  --batch_size 4 \
  --max_samples 512 \
  --lr 1e-3 \
  --alpha 0.7 \
  --tau 5 \
  --rnn_hidden 64 \
  --rnn_layers 1 \
  --cbam_reduction 8 \
  --cbam_sa_kernel 7 \
  --teacher_name facebook/wav2vec2-base \
  --run_name crnn_cbam_rbkdatt_smoke

