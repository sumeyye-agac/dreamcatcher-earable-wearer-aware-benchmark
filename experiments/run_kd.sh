#!/usr/bin/env bash
set -e

# Activate virtual environment with all dependencies
source dream-env/bin/activate

echo "== RB-KD: student=CRNN =="
python3 -m src.training.train_kd \
  --student crnn \
  --epochs 30 \
  --batch_size 8 \
  --lr 1e-3 \
  --early_stop_patience 5 \
  --early_stop_min_delta 0.001 \
  --alpha 0.7 \
  --tau 5 \
  --rnn_hidden 64 \
  --rnn_layers 1 \
  --teacher_name facebook/wav2vec2-base \
  --run_name crnn_rbkd

echo "== RB-KD-Att: student=CRNN+CBAM =="
python3 -m src.training.train_kd \
  --student crnn_cbam \
  --epochs 30 \
  --batch_size 8 \
  --lr 1e-3 \
  --early_stop_patience 5 \
  --early_stop_min_delta 0.001 \
  --alpha 0.7 \
  --tau 5 \
  --rnn_hidden 64 \
  --rnn_layers 1 \
  --cbam_reduction 8 \
  --cbam_sa_kernel 7 \
  --teacher_name facebook/wav2vec2-base \
  --run_name crnn_cbam_rbkdatt

echo "== RB-KD: student=TinyCNN =="
python3 -m src.training.train_kd \
  --student tinycnn \
  --epochs 30 \
  --batch_size 8 \
  --lr 1e-3 \
  --early_stop_patience 5 \
  --early_stop_min_delta 0.001 \
  --alpha 0.7 \
  --tau 5 \
  --teacher_name facebook/wav2vec2-base \
  --run_name tinycnn_rbkd
