#!/bin/bash
#
# Knowledge Distillation: Train TinyCNN student with CRNN_CBAM teacher
#
# Usage:
#   bash experiments/train_kd.sh
#
# Requirements:
#   - Teacher model checkpoint must exist at: results/runs/crnn_cbam/best_model.pth
#   - Pre-computed spectrograms in results/cache/spectrograms/
#

set -e

TEACHER_CHECKPOINT="results/runs/crnn_cbam/best_model.pth"

# Check if teacher checkpoint exists
if [ ! -f "$TEACHER_CHECKPOINT" ]; then
    echo "ERROR: Teacher checkpoint not found at: $TEACHER_CHECKPOINT"
    echo "Please train CRNN_CBAM first using: bash experiments/train_teachers.sh"
    exit 1
fi

echo "=== Knowledge Distillation Training ==="
echo "Teacher: CRNN_CBAM (74K params)"
echo "Student: TinyCNN (23K params)"
echo "Temperature: 5.0"
echo "Alpha: 0.7"
echo ""

python3 -m src.training.train_kd \
  --student_model tinycnn \
  --teacher_model crnn_cbam \
  --teacher_checkpoint "$TEACHER_CHECKPOINT" \
  --temperature 5.0 \
  --alpha 0.7 \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-3 \
  --early_stop_patience 5 \
  --run_name tinycnn_kd \
  --class_weights 1.0,1.5,5.5 \
  --rnn_hidden 64 \
  --rnn_layers 2 \
  --att_mode cbam \
  --cbam_reduction 16 \
  --cbam_sa_kernel 7

echo ""
echo "Training complete! Results saved to: results/runs/tinycnn_kd/"
