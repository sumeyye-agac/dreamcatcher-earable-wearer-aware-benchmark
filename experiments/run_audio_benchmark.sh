#!/usr/bin/env bash
set -e

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  DreamCatcher Audio Benchmark - Baseline Models"
echo "════════════════════════════════════════════════════════════"
echo ""

if ! python3 -c "from huggingface_hub import get_token; import sys; sys.exit(0 if get_token() else 1)"; then
  echo "ERROR: HuggingFace token not found. DreamCatcher is a gated dataset."
  echo "Run: hf auth login"
  echo "Or set: export HF_TOKEN=\"hf_...\"; export HUGGINGFACE_HUB_TOKEN=\"$HF_TOKEN\""
  exit 1
fi

echo "== TinyCNN baseline =="
python3 -m src.training.train_baseline \
  --model tinycnn \
  --epochs 5 \
  --batch_size 16 \
  --lr 1e-3 \
  --run_name tinycnn_baseline

echo ""
echo "== CRNN baseline =="
python3 -m src.training.train_baseline \
  --model crnn \
  --epochs 5 \
  --batch_size 16 \
  --lr 1e-3 \
  --rnn_hidden 64 \
  --rnn_layers 1 \
  --run_name crnn_baseline

echo ""
echo "== CRNN + CBAM (attention) =="
python3 -m src.training.train_baseline \
  --model crnn_cbam \
  --epochs 5 \
  --batch_size 16 \
  --lr 1e-3 \
  --rnn_hidden 64 \
  --rnn_layers 1 \
  --cbam_reduction 8 \
  --cbam_sa_kernel 7 \
  --run_name crnn_cbam_att
