# Experiment Workflow

This document outlines the complete workflow for running experiments in this repository.

## Overview

This repository implements knowledge distillation (KD) for audio event classification on the DreamCatcher dataset. The workflow consists of three main phases:

1. **Teacher Training** - Train large teacher models
2. **Baseline Training** - Train student models without KD
3. **Knowledge Distillation** - Train students with KD from teachers

## Complete Workflow

### Phase 1: Train Teacher Models (REQUIRED FIRST)

Before running any KD experiments, you **must** train the teacher models:

```bash
# Train both ViT and EfficientNet teachers
bash experiments/train_teachers.sh
```

This creates trained checkpoints:
- `results/runs/vit_teacher_finetuned/teacher_checkpoint.pth`
- `results/runs/efficientnet_teacher_finetuned/teacher_checkpoint.pth`

**Time estimate**: ~2-3 hours total (both teachers)

See [TEACHER_TRAINING.md](TEACHER_TRAINING.md) for detailed documentation.

### Phase 2: Train Baseline Student Models

Train student models without knowledge distillation to establish baselines:

```bash
# Train all baseline models
bash experiments/run_audio_benchmark.sh
```

This trains:
- TinyCNN (23.9K params)
- CRNN (48.8K params)
- CRNN+CBAM (49.2K params)

**Time estimate**: ~1-2 hours total

### Phase 3: Knowledge Distillation Experiments

Now you can run KD experiments using the trained teachers:

```bash
# Option A: Update shell scripts to use trained checkpoints
# Edit experiments/run_kd.sh and add --teacher_checkpoint arguments

# Option B: Run individual experiments
python3 -m src.training.train_kd \
  --student crnn \
  --teacher_type vit \
  --teacher_checkpoint results/runs/vit_teacher_finetuned/teacher_checkpoint.pth \
  --alpha 0.7 \
  --tau 5 \
  --epochs 30
```

**⚠️ Important**: The KD scripts will warn you if you don't provide a checkpoint. Running without a trained teacher will give poor results.

## Quick Start (Smoke Test)

Test the entire pipeline quickly:

```bash
# 1. Train teachers (smoke test)
python3 -m src.training.train_teacher --teacher_type vit --dataset_mode smoke --epochs 2
python3 -m src.training.train_teacher --teacher_type efficientnet --dataset_mode smoke --epochs 2

# 2. Train baseline student (smoke test)
bash experiments/run_audio_smoke.sh

# 3. Run KD (smoke test)
python3 -m src.training.train_kd \
  --student crnn \
  --teacher_type vit \
  --teacher_checkpoint results/runs/vit_teacher_smoke/teacher_checkpoint.pth \
  --dataset_mode smoke \
  --epochs 2
```

## Dataset Modes

### Full Dataset (Production)

```bash
--dataset_mode full
```

Uses the complete DreamCatcher dataset.

### Smoke Test (Development)

```bash
--dataset_mode smoke
```

Uses a small subset for quick testing (~50 samples per split).

## Subset Experiments

The repository supports three dataset subsets:

### 1. Full 9-Class (Default)

```bash
python3 -m src.training.train_kd --student crnn --teacher_type vit
```

Classes: quiet, breathe, cough, snore, speech, chew, laugh, non_wearer, noise

### 2. Balanced 4-Class Subset

```bash
python3 -m src.training.train_kd_balanced4 --student crnn --teacher_type vit
```

Classes: quiet, breathe, non_wearer, snore (most balanced subset)

### 3. Respiratory 3-Class Subset

```bash
python3 -m src.training.train_kd_respiratory --student crnn --teacher_type vit
```

Classes: breathe, cough, snore (respiratory events only)

## Available Scripts

### Training Scripts

| Script | Purpose | Time |
|--------|---------|------|
| `experiments/train_teachers.sh` | Train ViT + EfficientNet teachers | ~2-3h |
| `experiments/run_audio_benchmark.sh` | Train baseline students | ~1-2h |
| `experiments/run_kd.sh` | KD experiments (needs update for checkpoints) | ~3-4h |
| `experiments/run_balanced4_kd.sh` | KD on balanced 4-class | ~2-3h |
| `experiments/run_respiratory_kd.sh` | KD on respiratory subset | ~1-2h |

### Evaluation Scripts

| Script | Purpose |
|--------|---------|
| `experiments/eval_teacher.sh` | Evaluate trained teachers |
| `experiments/run_balanced4_teacher.sh` | Evaluate teachers on balanced4 |

### Smoke Test Scripts

| Script | Purpose |
|--------|---------|
| `experiments/run_audio_smoke.sh` | Quick baseline test |
| `experiments/run_kd_smoke.sh` | Quick KD test |
| `experiments/run_balanced4_baseline.sh` | Balanced4 baseline test |

## Results and Logs

### Output Locations

- **Checkpoints**: `results/runs/<run_name>/`
- **Logs**: `logs/<run_name>.log`
- **Leaderboard**: `results/leaderboard.csv`
- **Run steps**: `results/run_steps.csv`

### Monitoring Progress

```bash
# Watch leaderboard updates
watch -n 5 "tail -20 results/leaderboard.csv"

# Follow log file
tail -f logs/<run_name>.log

# Check run steps
tail -f results/run_steps.csv
```

## Common Issues

### Issue: KD Warning about Untrained Teacher

```
⚠️  WARNING: Initializing teacher from pre-trained google/vit-base-patch16-224 without fine-tuning!
```

**Solution**: Train the teacher first using Phase 1 above.

### Issue: Out of Memory

**Solutions**:
- Reduce batch size: `--batch_size 4`
- Use gradient accumulation (not currently implemented)
- Use smaller teacher: `--teacher_type efficientnet`

### Issue: Slow Training

**Solutions**:
- Use GPU: Ensure CUDA or MPS is available
- Use smoke mode for testing: `--dataset_mode smoke`
- Reduce epochs: `--epochs 10`

## Reproducibility

For reproducible results:

1. Set random seed: `--seed 42`
2. Use same dataset mode
3. Use same hyperparameters
4. Use same teacher checkpoint

## Next Steps

After completing the workflow:

1. **Analyze results**: Check `results/leaderboard.csv`
2. **Compare models**: Look at test F1 scores and model sizes
3. **Evaluate KD effectiveness**: Compare KD students vs baselines
4. **Experiment with hyperparameters**: Try different alpha/tau values

## References

- Main README: [README.md](README.md)
- Teacher training guide: [TEACHER_TRAINING.md](TEACHER_TRAINING.md)
- Dataset info: [src/data/dreamcatcher_hf.py](src/data/dreamcatcher_hf.py)
