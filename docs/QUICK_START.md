# Quick Start Guide

Get started with the DreamCatcher benchmark in 3 steps.

## Prerequisites

- Python 3.9+
- HuggingFace account with DreamCatcher dataset access
- ~50GB disk space

## Step 1: Setup (5 minutes)

```bash
# Clone and setup
git clone <repository-url>
cd dreamcatcher-earable-wearer-aware-benchmark

# Create virtual environment
python3 -m venv dream-env
source dream-env/bin/activate  # Windows: dream-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Authenticate with HuggingFace
huggingface-cli login
# Enter your token when prompted
```

## Step 2: Train Teacher (2-3 hours)

Teacher must be trained before KD experiments:

```bash
# Train EfficientNet teacher for 4-class balanced subset
bash experiments/train_teachers.sh
```

This creates checkpoint:
- `results/runs/efficientnet_teacher_4class/teacher_checkpoint.pth`

## Step 3: Run Experiments

### Option A: Baseline Models

Train student models without KD:

```bash
bash experiments/run_audio_benchmark.sh
```

### Option B: Knowledge Distillation

Train students with KD from trained teacher:

```bash
bash experiments/run_kd.sh
```

Or run individual experiments:

```bash
python3 -m src.training.train_kd \
  --student crnn \
  --teacher_checkpoint results/runs/efficientnet_teacher_4class/teacher_checkpoint.pth \
  --teacher_name google/efficientnet-b0 \
  --alpha 0.7 \
  --tau 5 \
  --epochs 50
```

## Quick Test (Smoke Mode)

Test the pipeline quickly with small data:

```bash
# Step 1: Train teachers (smoke)
python3 -m src.training.train_teacher \
  --teacher_type vit \
  --dataset_mode smoke \
  --epochs 2

# Step 2: Train student (smoke)
python3 -m src.training.train_baseline \
  --model crnn \
  --dataset_mode smoke \
  --epochs 2

# Step 3: KD (smoke)
python3 -m src.training.train_kd \
  --student crnn \
  --teacher_type vit \
  --teacher_checkpoint results/runs/vit_teacher_smoke/teacher_checkpoint.pth \
  --dataset_mode smoke \
  --epochs 2
```

## View Results

```bash
# Check leaderboard
cat results/leaderboard.csv

# Watch training logs
tail -f logs/<run_name>.log

# Monitor experiments
bash scripts/watch_experiments.sh
```

## Common Commands

### Dataset Operations

```bash
# Visualize class distribution
python scripts/plot_distribution.py

# Dataset info
python scripts/earable_cli.py --help
```

### Training

```bash
# Train teacher
python3 -m src.training.train_teacher --teacher_type vit --epochs 10

# Train baseline student
python3 -m src.training.train_baseline --model crnn --epochs 30

# Train with KD
python3 -m src.training.train_kd \
  --student crnn \
  --teacher_checkpoint <path> \
  --alpha 0.7 \
  --tau 5
```

### Evaluation

```bash
# Evaluate teacher
python3 -m src.training.eval_teacher \
  --teacher_type vit \
  --teacher_checkpoint <path>

# Check results
cat results/leaderboard.csv | column -t -s,
```

## Next Steps

- Read [WORKFLOW.md](WORKFLOW.md) for complete workflow
- Read [TEACHER_TRAINING.md](TEACHER_TRAINING.md) for teacher training details
- Check [README.md](../README.md) for full documentation

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
--batch_size 4
```

### No Space Left
```bash
# Set HuggingFace cache location
export HF_DATASETS_CACHE="/path/with/space"
```

### Slow Training
```bash
# Use GPU if available (auto-detected)
# Or explicitly set device
--device cuda  # or mps for Mac
```

### Teacher Warning
```
⚠️  WARNING: Initializing teacher from pre-trained ... without fine-tuning!
```
**Solution**: Train teachers first using `experiments/train_teachers.sh`
