#!/usr/bin/env bash
set -e

echo "== RB-KD: student=CRNN =="
python -m src.training.train_kd \
  --student crnn \
  --epochs 5 \
  --batch_size 8 \
  --lr 1e-3 \
  --alpha 0.7 \
  --tau 5 \
  --rnn_hidden 64 \
  --rnn_layers 1 \
  --teacher_name facebook/wav2vec2-base \
  --run_name crnn_rbkd

echo "== RB-KD-Att: student=CRNN+CBAM =="
python -m src.training.train_kd \
  --student crnn_cbam \
  --epochs 5 \
  --batch_size 8 \
  --lr 1e-3 \
  --alpha 0.7 \
  --tau 5 \
  --rnn_hidden 64 \
  --rnn_layers 1 \
  --cbam_reduction 8 \
  --cbam_sa_kernel 7 \
  --teacher_name facebook/wav2vec2-base \
  --run_name crnn_cbam_rbkdatt
