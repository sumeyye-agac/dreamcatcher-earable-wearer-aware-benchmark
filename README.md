# DreamCatcher Sleep Event Classification

[![Status](https://img.shields.io/badge/status-under%20development-yellow.svg)](#)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Datasets-yellow.svg)](https://huggingface.co/)

**Note:** This repository is under active development. Performance benchmarks are being validated.

Resource-efficient sleep event classification on earables using the DreamCatcher dataset.

This repository provides a benchmarking framework for sleep event recognition
from in-ear audio under real-world conditions. The focus is on building
lightweight models using attention mechanisms and knowledge distillation
for resource-constrained wearable devices.

**Dataset:** 3-class classification (quiet, breathe, snore) from in-ear audio recordings.

---

## Problem

Sleep event monitoring using earables introduces unique challenges:

- Sleep events (quiet, breathing, snoring) are captured by in-ear microphones
- Audio quality varies significantly across different sleep stages and positions
- Models must operate under strict resource constraints for wearable deployment
- Real-time inference requires low-latency and low-power processing

This repository addresses these challenges through:
- Lightweight CNN and RNN architectures
- Attention mechanisms (CBAM) for improved feature learning
- Knowledge distillation for model compression
- Weighted loss for handling class imbalance

---

## Installation

### Prerequisites

- Python 3.9+ (tested with Python 3.12)
- ~50GB free disk space (for dataset download and cache)
- For full (non-smoke) runs, you may need **significantly more** free disk due to HuggingFace caching.
- HuggingFace account with access to the DreamCatcher dataset

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd dreamcatcher-earable-wearer-aware-benchmark
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv dream-env
   source dream-env/bin/activate  # On Windows: dream-env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   python3 -m pip install -r requirements.txt
   python3 -m pip install -e .
   ```

### If you hit "No space left on device"

Full dataset preparation can be large. You can move the HuggingFace datasets cache to another disk by either:

- Setting an environment variable:

```bash
export HF_DATASETS_CACHE="/path/with/space/hf_datasets_cache"
```

- Or passing `--cache_dir` to training scripts:

```bash
python3 -m src.training.train --cache_dir "/path/with/space/hf_datasets_cache" ...
```

4. **Authenticate with HuggingFace:**
   The DreamCatcher dataset is gated and requires authentication.
   
   - Request access on the [HuggingFace dataset page](https://huggingface.co/datasets/THU-PI-Sensing/DreamCatcher)
   - After approval, authenticate:
     ```bash
     hf auth login
     ```
   - Enter your HuggingFace token when prompted

   **Non-interactive / CI login (env var):** If `hf auth login` is not convenient (e.g., CI or terminals that can’t prompt),
   you can provide your token via environment variables:
   ```bash
   export HF_TOKEN="hf_..."
   export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
   ```

   **Rare invalid audio buffers:** Some samples may contain empty/invalid audio (NaN/inf). The loader can skip/resample
   these cases via `--invalid_audio_policy {skip,resample,zero}` (recommended: `skip`).

### Troubleshooting

**Dependency conflicts:** If you encounter issues with `datasets`, `pyarrow`, or `numpy`, ensure you're using the pinned versions in `requirements.txt`. These versions are tested and compatible.

**Disk space:** The dataset cache can grow large (~70-80GB). If you encounter "No space left on device" errors:

1. Check available disk space:
   ```bash
   df -h ~
   ```

2. Check HuggingFace cache size:
   ```bash
   du -sh ~/.cache/huggingface/datasets
   ```

3. Clear the HuggingFace cache if needed:
   ```bash
   rm -rf ~/.cache/huggingface/datasets/*
   ```
   
   **Warning:** This will delete all cached datasets. You'll need to re-download them on next use.

4. If disk space is still limited, consider using streaming mode (modify `src/data/dreamcatcher_hf.py` to set `streaming=True`). Note that streaming may be slower for training.

**Dataset loading:** The dataset uses the `sleep_event_classification` config. This is automatically handled by the code.

---

## Repository Structure

```
dreamcatcher-earable-wearer-aware-benchmark/
├── README.md                    # Main documentation
├── scripts/                     # Utility scripts
│   └── preprocess.py           # Cache spectrogram preprocessing
├── experiments/                 # Training scripts
│   └── train_teachers.sh       # Train CRNN_CBAM teacher
├── src/                        # Source code
│   ├── data/                  # Dataset loading
│   │   ├── dreamcatcher_hf.py        # HuggingFace dataset loader
│   │   ├── dreamcatcher_dataset.py   # 3-class dataset
│   │   ├── cached_dataset.py         # Fast HDF5 loader
│   │   └── audio_features.py         # Spectrogram extraction
│   ├── models/                # Model implementations
│   │   ├── tinycnn.py        # Lightweight CNN (23K params)
│   │   ├── crnn.py           # CNN + RNN
│   │   └── crnn_cbam.py      # CRNN + CBAM attention (74K params)
│   ├── training/              # Training scripts
│   │   └── train.py          # Main training script
│   ├── evaluation/            # Metrics and evaluation
│   └── utils/                 # Utilities
├── results/                    # Experiment outputs
│   ├── runs/                 # Per-run checkpoints and metrics
│   │   ├── tinycnn/
│   │   └── crnn_cbam_teacher/
│   ├── cache/                # Pre-computed spectrograms (HDF5)
│   └── leaderboard.csv       # All results
└── logs/                      # Training logs
```

---

## Dataset

DreamCatcher dataset (NeurIPS 2024 Datasets & Benchmarks Track)

- **Source:** [THU-PI-Sensing/DreamCatcher](https://huggingface.co/datasets/THU-PI-Sensing/DreamCatcher)
- **Modality:** In-ear audio recordings
- **Classes:** 3 sleep events (quiet, breathe, snore)
- **Total samples:** 380,362 (after filtering)
  - Train: 250,050 samples
  - Validation: 63,056 samples
  - Test: 67,256 samples
- **Features:** 128-band log-mel spectrograms (5-second windows)
- **Sample rate:** 16 kHz

**Preprocessing:**
```bash
python3 scripts/preprocess.py
```
Generates HDF5 cache (~30 GB) for fast training.

---

## Scope

This repository focuses on:

- **Audio-only 3-class classification** (quiet, breathe, snore)
- **Lightweight architectures** for wearable deployment
- **Attention mechanisms** (CBAM) for improved performance
- **Knowledge distillation** from larger teacher models
- **Class imbalance handling** via weighted loss
- **Resource-aware evaluation** (accuracy, parameters, latency)

**Note:** This is a sleep event classification task, not speech recognition.

---

## Models

### Implemented Architectures

| Model | Parameters | Description |
|-------|-----------|-------------|
| **TinyCNN** | 23K | Lightweight baseline CNN |
| **CRNN** | ~50K | CNN + Bidirectional GRU |
| **CRNN_CBAM** | 74K | CRNN + CBAM attention (teacher) |

**Key Features:**
- All models use 128-mel spectrograms as input
- CBAM: Convolutional Block Attention Module (channel + spatial attention)
- Weighted CrossEntropyLoss for class imbalance (weights: 1.0, 1.5, 5.5)
- Early stopping with patience=5 on validation F1

---

## Training

### Quick Start

**1. Preprocess dataset (one-time, ~30-40 min):**
```bash
python3 scripts/preprocess.py
```

**2. Train models:**

**TinyCNN (lightweight baseline):**
```bash
python3 -m src.training.train \
  --model tinycnn \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-3 \
  --early_stop_patience 5 \
  --run_name tinycnn \
  --class_weights 1.0,1.5,5.5
```

**CRNN_CBAM (teacher with attention):**
```bash
bash experiments/train_teachers.sh
```

**Monitor training:**
```bash
tail -f logs/tinycnn.log
tail -f logs/crnn_cbam.log
```

### Training Arguments

**Common arguments:**
- `--model`: Model architecture (tinycnn, crnn, crnn_cbam)
- `--epochs`: Maximum epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-3)
- `--early_stop_patience`: Early stopping patience (default: 5)
- `--class_weights`: Class weights for imbalanced data (e.g., 1.0,1.5,5.5)
- `--run_name`: Experiment name for logging

**CRNN_CBAM specific:**
- `--rnn_hidden`: RNN hidden size (default: 64)
- `--rnn_layers`: Number of RNN layers (default: 2)
- `--att_mode`: Attention mechanism (cbam)
- `--cbam_reduction`: CBAM reduction ratio (default: 16)
- `--cbam_sa_kernel`: CBAM spatial attention kernel (default: 7)

### Results

Results are saved to:
- `results/runs/<run_name>/metrics.json` - Per-run metrics
- `results/runs/<run_name>/best_model.pth` - Best checkpoint
- `results/runs/<run_name>/test_confusion_matrix.csv` - Confusion matrix
- `results/leaderboard.csv` - All experiments summary

## Expected Performance

Baseline performance on 3-class DreamCatcher test set:

| Model | Accuracy | F1-Macro | Parameters |
|-------|----------|----------|------------|
| TinyCNN | ~78-80% | ~74-76% | 23K |
| CRNN_CBAM | ~80-82% | ~76-78% | 74K |

*Note: Results may vary based on random initialization and data splits.*

## Key Features

- ✅ **Lightweight models** optimized for wearable deployment
- ✅ **Attention mechanisms** (CBAM) for improved feature learning
- ✅ **Class imbalance handling** via weighted loss
- ✅ **Fast training** with HDF5-cached spectrograms
- ✅ **Comprehensive metrics** (accuracy, F1, confusion matrix)
- ✅ **Early stopping** to prevent overfitting

## Citation

If you use the DreamCatcher dataset, please cite:
```bibtex
@inproceedings{dreamcatcher2024,
  title={DreamCatcher: A Dataset for Sleep Event Detection from In-Ear Audio},
  author={...},
  booktitle={NeurIPS 2024 Datasets and Benchmarks Track},
  year={2024}
}
```

---