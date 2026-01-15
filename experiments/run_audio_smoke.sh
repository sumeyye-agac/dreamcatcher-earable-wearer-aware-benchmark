#!/usr/bin/env bash
set -e

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  DreamCatcher Audio Benchmark - SMOKE (fast sanity check)"
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

echo "== TinyCNN (smoke) =="
python3 -m src.training.train_baseline \
  --model tinycnn \
  --dataset_mode smoke \
  --invalid_audio_policy skip \
  --epochs 1 \
  --batch_size 8 \
  --lr 1e-3 \
  --run_name tinycnn_smoke

echo ""
echo "== CRNN (smoke) =="
python3 -m src.training.train_baseline \
  --model crnn \
  --dataset_mode smoke \
  --invalid_audio_policy skip \
  --epochs 1 \
  --batch_size 8 \
  --lr 1e-3 \
  --rnn_hidden 64 \
  --rnn_layers 1 \
  --run_name crnn_smoke

echo ""
echo "== CRNN + CBAM (smoke) =="
python3 -m src.training.train_baseline \
  --model crnn_cbam \
  --dataset_mode smoke \
  --invalid_audio_policy skip \
  --epochs 1 \
  --batch_size 8 \
  --lr 1e-3 \
  --rnn_hidden 64 \
  --rnn_layers 1 \
  --cbam_reduction 8 \
  --cbam_sa_kernel 7 \
  --run_name crnn_cbam_smoke

