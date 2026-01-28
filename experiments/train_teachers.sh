#!/usr/bin/env bash
set -e

# Activate virtual environment
source dream-env/bin/activate

# Create logs directory
mkdir -p logs

echo "========================================"
echo "Training CRNN_CBAM Teacher (3-class)"
echo "========================================"
echo ""

python3 -m src.training.train \
  --model crnn_cbam \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-3 \
  --early_stop_patience 5 \
  --run_name crnn_cbam_teacher \
  --class_weights 1.0,1.5,5.5 \
  --rnn_hidden 64 \
  --rnn_layers 2 \
  --att_mode cbam \
  --cbam_reduction 16 \
  --cbam_sa_kernel 7 \
  2>&1 | tee logs/crnn_cbam_teacher.log

echo ""
echo "========================================"
echo "Teacher training complete"
echo "========================================"
echo ""
echo "Checkpoint: results/runs/crnn_cbam_teacher/best_model.pth"
echo ""
echo "Use this checkpoint for knowledge distillation experiments."
