#!/usr/bin/env bash
set -e

source dream-env/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

echo "=========================================="
echo "Balanced 4-Class Teacher Evaluation"
echo "Classes: quiet, breathe, non_wearer, snore"
echo "Teachers: ViT-base, EfficientNet-b0"
echo "=========================================="
echo ""

echo "== Evaluating ViT Teacher on Balanced 4-Class Test Set =="
python3 -m src.training.eval_teacher_balanced4 \
  --teacher_type vit \
  --teacher_name google/vit-base-patch16-224 \
  --batch_size 4 \
  --dataset_mode full \
  --device cpu \
  --run_name vit_teacher_balanced4 \
  2>&1 | tee logs/vit_teacher_balanced4_eval.log

echo ""
echo "== Evaluating EfficientNet Teacher on Balanced 4-Class Test Set =="
python3 -m src.training.eval_teacher_balanced4 \
  --teacher_type efficientnet \
  --teacher_name google/efficientnet-b0 \
  --batch_size 4 \
  --dataset_mode full \
  --device cpu \
  --run_name efficientnet_teacher_balanced4 \
  2>&1 | tee logs/efficientnet_teacher_balanced4_eval.log

echo ""
echo "=========================================="
echo "Balanced 4-Class Teacher Evaluation Complete!"
echo "=========================================="
