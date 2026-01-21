#!/usr/bin/env bash
set -e

source dream-env/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

echo "========================================"
echo "Balanced 4-Class KD Experiments"
echo "Classes: quiet, breathe, non_wearer, snore"
echo "Teachers: ViT-base, EfficientNet-b0"
echo "========================================"
echo ""

echo "== Balanced 4-Class KD: CRNN + ViT =="
python3 -m src.training.train_kd_balanced4 \
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
  --run_name crnn_balanced4_vit_kd \
  2>&1 | tee logs/crnn_balanced4_vit_kd.log

echo ""
echo "== Balanced 4-Class KD: CRNN+CBAM + ViT =="
python3 -m src.training.train_kd_balanced4 \
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
  --run_name crnn_cbam_balanced4_vit_kd \
  2>&1 | tee logs/crnn_cbam_balanced4_vit_kd.log

echo ""
echo "== Balanced 4-Class KD: TinyCNN + ViT =="
python3 -m src.training.train_kd_balanced4 \
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
  --run_name tinycnn_balanced4_vit_kd \
  2>&1 | tee logs/tinycnn_balanced4_vit_kd.log

echo ""
echo "== Balanced 4-Class KD: CRNN + EfficientNet =="
python3 -m src.training.train_kd_balanced4 \
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
  --run_name crnn_balanced4_efficientnet_kd \
  2>&1 | tee logs/crnn_balanced4_efficientnet_kd.log

echo ""
echo "== Balanced 4-Class KD: CRNN+CBAM + EfficientNet =="
python3 -m src.training.train_kd_balanced4 \
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
  --run_name crnn_cbam_balanced4_efficientnet_kd \
  2>&1 | tee logs/crnn_cbam_balanced4_efficientnet_kd.log

echo ""
echo "== Balanced 4-Class KD: TinyCNN + EfficientNet =="
python3 -m src.training.train_kd_balanced4 \
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
  --run_name tinycnn_balanced4_efficientnet_kd \
  2>&1 | tee logs/tinycnn_balanced4_efficientnet_kd.log

echo ""
echo "=========================================="
echo "Balanced 4-Class KD Complete!"
echo "=========================================="
