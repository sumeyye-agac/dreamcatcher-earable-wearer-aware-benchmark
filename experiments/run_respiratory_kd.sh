#!/usr/bin/env bash
set -e

# Activate virtual environment with all dependencies
source dream-env/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

echo "========================================"
echo "Respiratory Subset KD Experiments"
echo "Classes: breathe, cough, snore (3-class)"
echo "Teachers: ViT-base, EfficientNet-b0"
echo "========================================"
echo ""

echo "== Respiratory KD: CRNN + ViT =="
python3 -m src.training.train_kd_respiratory \
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
  --teacher_type vit \
  --teacher_name google/vit-base-patch16-224 \
  --run_name crnn_respiratory_vit_kd \
  2>&1 | tee logs/crnn_respiratory_vit_kd.log

echo ""
echo "== Respiratory KD: CRNN+CBAM + ViT =="
python3 -m src.training.train_kd_respiratory \
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
  --teacher_type vit \
  --teacher_name google/vit-base-patch16-224 \
  --run_name crnn_cbam_respiratory_vit_kd \
  2>&1 | tee logs/crnn_cbam_respiratory_vit_kd.log

echo ""
echo "== Respiratory KD: TinyCNN + ViT =="
python3 -m src.training.train_kd_respiratory \
  --student tinycnn \
  --epochs 30 \
  --batch_size 8 \
  --lr 1e-3 \
  --early_stop_patience 5 \
  --early_stop_min_delta 0.001 \
  --alpha 0.7 \
  --tau 5 \
  --teacher_type vit \
  --teacher_name google/vit-base-patch16-224 \
  --run_name tinycnn_respiratory_vit_kd \
  2>&1 | tee logs/tinycnn_respiratory_vit_kd.log

echo ""
echo "== Respiratory KD: CRNN + EfficientNet =="
python3 -m src.training.train_kd_respiratory \
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
  --teacher_type efficientnet \
  --teacher_name google/efficientnet-b0 \
  --run_name crnn_respiratory_efficientnet_kd \
  2>&1 | tee logs/crnn_respiratory_efficientnet_kd.log

echo ""
echo "== Respiratory KD: CRNN+CBAM + EfficientNet =="
python3 -m src.training.train_kd_respiratory \
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
  --teacher_type efficientnet \
  --teacher_name google/efficientnet-b0 \
  --run_name crnn_cbam_respiratory_efficientnet_kd \
  2>&1 | tee logs/crnn_cbam_respiratory_efficientnet_kd.log

echo ""
echo "== Respiratory KD: TinyCNN + EfficientNet =="
python3 -m src.training.train_kd_respiratory \
  --student tinycnn \
  --epochs 30 \
  --batch_size 8 \
  --lr 1e-3 \
  --early_stop_patience 5 \
  --early_stop_min_delta 0.001 \
  --alpha 0.7 \
  --tau 5 \
  --teacher_type efficientnet \
  --teacher_name google/efficientnet-b0 \
  --run_name tinycnn_respiratory_efficientnet_kd \
  2>&1 | tee logs/tinycnn_respiratory_efficientnet_kd.log

echo ""
echo "=========================================="
echo "Respiratory KD Experiments Complete!"
echo "=========================================="
