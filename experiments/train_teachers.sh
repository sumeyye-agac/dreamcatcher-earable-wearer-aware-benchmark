#!/usr/bin/env bash
set -e

# Activate virtual environment
source dream-env/bin/activate

# Create logs directory
mkdir -p logs

echo "========================================"
echo "Training EfficientNet Teacher (4-class)"
echo "========================================"
echo ""

python3 -m src.training.train_teacher \
  --teacher_type efficientnet \
  --epochs 30 \
  --batch_size 32 \
  --lr 1e-4 \
  --early_stop_patience 5 \
  --early_stop_min_delta 0.0005 \
  --run_name efficientnet_teacher_4class \
  2>&1 | tee logs/efficientnet_teacher.log

echo ""
echo "========================================"
echo "Teacher training complete"
echo "========================================"
echo ""
echo "Checkpoint: results/runs/efficientnet_teacher_4class/teacher_checkpoint.pth"
echo ""
echo "Use this checkpoint for knowledge distillation experiments."
