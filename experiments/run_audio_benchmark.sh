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

earable suite audio-benchmark \
  --epochs 50 \
  --batch_size 16 \
  --lr 1e-3 \
  --early_stop_patience 10 \
  --early_stop_min_delta 0.0001
