# Teacher Model Training Guide

This guide explains how to train and use teacher models for knowledge distillation experiments.

## Overview

The repository supports two vision-based teacher models that treat audio spectrograms as images:

1. **ViT (Vision Transformer)** - ~86M parameters, powerful but large
2. **EfficientNet-b0** - ~5M parameters, lightweight and efficient

Both models:
- Use pre-trained ImageNet weights as initialization
- Convert log-mel spectrograms to RGB images (224x224)
- Fine-tune on DreamCatcher audio classification task
- Serve as teachers for knowledge distillation to smaller student models

## Quick Start

### Step 1: Train Teacher Models

```bash
# Train both teachers (recommended)
bash experiments/train_teachers.sh

# Or train individually
python3 -m src.training.train_teacher --teacher_type vit --epochs 10
python3 -m src.training.train_teacher --teacher_type efficientnet --epochs 10
```

### Step 2: Use Trained Teachers for Knowledge Distillation

After training, checkpoints are saved in `results/runs/`:
- `results/runs/vit_teacher_finetuned/teacher_checkpoint.pth`
- `results/runs/efficientnet_teacher_finetuned/teacher_checkpoint.pth`

Use these checkpoints in KD experiments:

```bash
# Example: Train student with ViT teacher
python3 -m src.training.train_kd \
  --student crnn \
  --teacher_type vit \
  --teacher_checkpoint results/runs/vit_teacher_finetuned/teacher_checkpoint.pth \
  --alpha 0.7 \
  --tau 5 \
  --epochs 30
```

## Training Options

### Basic Usage

```bash
python3 -m src.training.train_teacher \
  --teacher_type {vit|efficientnet} \
  --epochs 10 \
  --batch_size 8 \
  --lr 1e-4
```

### Advanced Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--teacher_type` | **required** | Teacher model type: `vit` or `efficientnet` |
| `--teacher_name` | auto | HuggingFace model name (optional override) |
| `--epochs` | 10 | Number of training epochs |
| `--batch_size` | 8 | Training batch size |
| `--lr` | 1e-4 | Learning rate |
| `--n_mels` | 64 | Number of mel-frequency bins |
| `--sr` | 16000 | Audio sample rate |
| `--seed` | 42 | Random seed for reproducibility |
| `--early_stop_patience` | 3 | Early stopping patience |
| `--early_stop_min_delta` | 0.001 | Min improvement for early stopping |
| `--device` | auto | Device: `auto`, `cpu`, `cuda`, or `mps` |
| `--freeze_encoder` | False | Keep encoder frozen (only train head) |
| `--dataset_mode` | full | `full` or `smoke` for quick testing |
| `--run_name` | auto | Custom run name for logging |

### Example: Smoke Test

Quick test on small subset:

```bash
python3 -m src.training.train_teacher \
  --teacher_type vit \
  --dataset_mode smoke \
  --epochs 2
```

### Example: Frozen Encoder (Not Recommended)

Train only the classification head with frozen encoder:

```bash
python3 -m src.training.train_teacher \
  --teacher_type efficientnet \
  --freeze_encoder \
  --epochs 5
```

**⚠️ Warning:** This approach gives poor results. Always fine-tune the encoder.

## Using Teachers for Knowledge Distillation

### Option 1: With Trained Checkpoint (Recommended)

```bash
python3 -m src.training.train_kd \
  --student crnn \
  --teacher_type vit \
  --teacher_checkpoint results/runs/vit_teacher_finetuned/teacher_checkpoint.pth \
  --alpha 0.7 \
  --tau 5
```

### Option 2: Without Checkpoint (Not Recommended)

```bash
python3 -m src.training.train_kd \
  --student crnn \
  --teacher_type vit \
  --alpha 0.7 \
  --tau 5
```

**⚠️ Warning:** Without a trained checkpoint, the teacher uses raw pre-trained weights (ImageNet) which have never seen audio spectrograms. This will give poor KD results.

## Expected Results

### Training Time

- **ViT**: ~45-60 minutes for 10 epochs (GPU)
- **EfficientNet**: ~30-45 minutes for 10 epochs (GPU)
- **CPU**: 5-10x slower

### Expected Performance

On DreamCatcher 9-class audio event classification:

| Model | Params | Test F1 | Test Acc |
|-------|--------|---------|----------|
| ViT-base (fine-tuned) | 86M | ~0.80-0.85 | ~0.80-0.85 |
| EfficientNet-b0 (fine-tuned) | 5M | ~0.75-0.80 | ~0.75-0.80 |

*Note: Actual results depend on hyperparameters and training runs*

## Output Files

After training, each run creates:

```
results/runs/<run_name>/
├── teacher_checkpoint.pth      # Model checkpoint (use for KD)
├── metrics.json                # Full training metrics
├── test_confusion_matrix.csv   # Per-class predictions
├── args.json                   # Training arguments
└── env.json                    # Environment snapshot
```

Additionally, results are logged to:
- `results/leaderboard.csv` - All experiment results
- `logs/<run_name>.log` - Training logs

## Troubleshooting

### Out of Memory

Reduce batch size:
```bash
python3 -m src.training.train_teacher --teacher_type vit --batch_size 4
```

### Slow Training on CPU

Use MPS (Mac) or CUDA (GPU) if available:
```bash
python3 -m src.training.train_teacher --teacher_type efficientnet --device mps
```

### Poor Teacher Performance

1. Check if encoder was unfrozen (don't use `--freeze_encoder`)
2. Increase training epochs: `--epochs 15`
3. Try lower learning rate: `--lr 5e-5`
4. Check for class imbalance in dataset

## Architecture Details

### ViT Teacher

```python
ViTTeacher(
  encoder: ViTModel (frozen pre-trained from ImageNet)
  head: Sequential(
    Linear(768, 256),
    ReLU(),
    Dropout(0.1),
    Linear(256, n_classes)
  )
)
```

### EfficientNet Teacher

```python
EfficientNetTeacher(
  encoder: EfficientNetModel (frozen pre-trained from ImageNet)
  head: Sequential(
    Linear(1280, 256),
    ReLU(),
    Dropout(0.1),
    Linear(256, n_classes)
  )
)
```

## Notes

- **Fine-tuning is essential**: Pre-trained weights from ImageNet need adaptation to audio spectrograms
- **Encoder must be unfrozen**: The default behavior unfreezes the encoder for fine-tuning
- **Checkpoints are portable**: Save and share trained teachers across experiments
- **Reproducibility**: Set `--seed` for deterministic results

## References

- ViT: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
- EfficientNet: "Rethinking Model Scaling for CNNs" (Tan & Le, 2019)
- Knowledge Distillation: "Distilling the Knowledge in a Neural Network" (Hinton et al., 2015)
