#!/usr/bin/env bash
set -e

# Activate virtual environment with all dependencies
source dream-env/bin/activate

echo "=========================================="
echo "Evaluating Teacher Model: Wav2Vec2-base"
echo "=========================================="

python3 -m src.training.eval_teacher \
  --teacher_name facebook/wav2vec2-base \
  --batch_size 4 \
  --sr 16000 \
  --dataset_mode full \
  --device cpu \
  --run_name wav2vec2_teacher \
  --out_csv results/leaderboard.csv \
  --steps_csv results/run_steps.csv

echo ""
echo "=========================================="
echo "Teacher evaluation complete!"
echo "=========================================="
