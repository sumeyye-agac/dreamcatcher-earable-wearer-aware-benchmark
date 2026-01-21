#!/usr/bin/env bash
set -e

# Activate virtual environment
source dream-env/bin/activate

# Create logs directory
mkdir -p logs

echo "========================================"
echo "Training EfficientNet Teacher"
echo "========================================"
echo ""

echo "== Training EfficientNet Teacher (fine-tuned) =="
python3 -m src.training.train_teacher \
  --teacher_type efficientnet \
  --epochs 10 \
  --batch_size 8 \
  --lr 1e-4 \
  --early_stop_patience 3 \
  --early_stop_min_delta 0.001 \
  --run_name efficientnet_teacher_finetuned \
  2>&1 | tee logs/efficientnet_teacher_train.log

echo ""
echo "========================================"
echo "Teacher training complete"
echo "========================================"
echo ""
echo "Checkpoint saved: results/runs/efficientnet_teacher_finetuned/teacher_checkpoint.pth"
echo ""
echo "Use this checkpoint for knowledge distillation experiments."
