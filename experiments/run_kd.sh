#!/usr/bin/env bash
set -e

# Activate virtual environment with all dependencies
source earable-env/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Define EfficientNet teacher checkpoint
TEACHER_CHECKPOINT="results/runs/efficientnet_teacher_4class/teacher_checkpoint.pth"

if [ ! -f "$TEACHER_CHECKPOINT" ]; then
  echo "ERROR: Teacher checkpoint not found: $TEACHER_CHECKPOINT"
  echo "Please train the teacher first using: bash experiments/train_teachers.sh"
  exit 1
fi

echo "========================================"
echo "Knowledge Distillation Experiments (4-class)"
echo "Using EfficientNet Teacher"
echo "========================================"
echo ""

echo "== KD: student=CRNN, teacher=EfficientNet =="
python3 -m src.training.train_kd \
  --student crnn \
  --epochs 50 \
  --batch_size 8 \
  --lr 1e-3 \
  --early_stop_patience 10 \
  --early_stop_min_delta 0.0001 \
  --alpha 0.7 \
  --tau 5 \
  --rnn_hidden 64 \
  --rnn_layers 1 \
  --teacher_checkpoint "$TEACHER_CHECKPOINT" \
  --teacher_name google/efficientnet-b0 \
  --run_name crnn_efficientnet_kd_4class \
  2>&1 | tee logs/crnn_efficientnet_kd.log

echo ""
echo "== KD: student=CRNN+CBAM, teacher=EfficientNet =="
python3 -m src.training.train_kd \
  --student crnn_cbam \
  --epochs 50 \
  --batch_size 8 \
  --lr 1e-3 \
  --early_stop_patience 10 \
  --early_stop_min_delta 0.0001 \
  --alpha 0.7 \
  --tau 5 \
  --rnn_hidden 64 \
  --rnn_layers 1 \
  --cbam_reduction 8 \
  --cbam_sa_kernel 7 \
  --att_mode cbam \
  --teacher_checkpoint "$TEACHER_CHECKPOINT" \
  --teacher_name google/efficientnet-b0 \
  --run_name crnn_cbam_efficientnet_kd_4class \
  2>&1 | tee logs/crnn_cbam_efficientnet_kd.log

echo ""
echo "== KD: student=TinyCNN, teacher=EfficientNet =="
python3 -m src.training.train_kd \
  --student tinycnn \
  --epochs 50 \
  --batch_size 8 \
  --lr 1e-3 \
  --early_stop_patience 10 \
  --early_stop_min_delta 0.0001 \
  --alpha 0.7 \
  --tau 5 \
  --teacher_checkpoint "$TEACHER_CHECKPOINT" \
  --teacher_name google/efficientnet-b0 \
  --run_name tinycnn_efficientnet_kd_4class \
  2>&1 | tee logs/tinycnn_efficientnet_kd.log

echo ""
echo "========================================"
echo "All KD experiments complete"
echo "========================================"
