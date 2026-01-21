#!/usr/bin/env bash
set -e

# Activate virtual environment with all dependencies
source dream-env/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

echo "=========================================="
echo "Evaluating Teacher Model: ViT-base"
echo "=========================================="

python3 -m src.training.eval_teacher \
  --teacher_type vit \
  --teacher_name google/vit-base-patch16-224 \
  --batch_size 4 \
  --sr 16000 \
  --dataset_mode full \
  --device cpu \
  --run_name vit_teacher \
  --out_csv results/leaderboard.csv \
  --steps_csv results/run_steps.csv \
  2>&1 | tee logs/vit_teacher_eval.log

echo ""
echo "=========================================="
echo "Evaluating Teacher Model: EfficientNet-b0"
echo "=========================================="

python3 -m src.training.eval_teacher \
  --teacher_type efficientnet \
  --teacher_name google/efficientnet-b0 \
  --batch_size 4 \
  --sr 16000 \
  --dataset_mode full \
  --device cpu \
  --run_name efficientnet_teacher \
  --out_csv results/leaderboard.csv \
  --steps_csv results/run_steps.csv \
  2>&1 | tee logs/efficientnet_teacher_eval.log

echo ""
echo "=========================================="
echo "Teacher evaluation complete!"
echo "=========================================="
