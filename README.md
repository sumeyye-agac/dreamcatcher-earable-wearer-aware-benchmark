# DreamCatcher Earable Wearer-Aware Benchmark

[![Status](https://img.shields.io/badge/status-under%20development-yellow.svg)](#)

**Note:** This repository is under active development. Teacher models and KD pipeline have been implemented but performance benchmarks are still being validated.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)

Resource-efficient wearer-aware event recognition on earables (DreamCatcher dataset).

This repository provides a reproducible benchmarking framework for earable-based
event recognition under real-world sleep conditions. The focus is on building
lightweight and resource-aware models using attention mechanisms and knowledge
distillation, while maintaining strong recognition performance.

The current implementation is audio-only, based on the publicly available
DreamCatcher release, and is designed to be easily extendable to multimodal
audio + IMU settings when additional data access is available.

---

## Problem

Wearable sleep monitoring using earables introduces unique challenges:

- Events such as snoring, breathing irregularities, coughing, or body movements
  are captured by in-ear microphones, often mixed with sounds from a co-sleeper.
- The system must distinguish wearer-generated events from external events
  (wearer-aware recognition).
- Models are expected to operate under strict resource constraints, as earables
  require low-latency and low-power inference.

This repository studies these challenges from a resource-aware applied machine
learning perspective.

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
   python3 -m venv .venv-earable
   source .venv-earable/bin/activate  # On Windows: .venv-earable\Scripts\activate
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
python3 -m src.training.train_baseline --cache_dir "/path/with/space/hf_datasets_cache" ...
python3 -m src.training.train_kd --cache_dir "/path/with/space/hf_datasets_cache" ...
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
├── docs/                        # Additional documentation
│   ├── TEACHER_TRAINING.md     # Teacher training guide
│   └── WORKFLOW.md             # Complete experiment workflow
├── scripts/                     # Utility scripts
│   ├── earable_cli.py          # CLI tool for dataset operations
│   ├── monitor_experiments.sh  # Monitor running experiments
│   ├── watch_experiments.sh    # Watch experiment progress
│   └── plot_distribution.py    # Dataset visualization
├── experiments/                 # Shell scripts for experiments
│   ├── train_teachers.sh       # Train teacher models
│   ├── run_kd.sh              # Knowledge distillation
│   ├── run_audio_benchmark.sh # Baseline training
│   └── ...                    # Other experiment scripts
├── src/                        # Source code
│   ├── data/                  # Dataset loading and preprocessing
│   ├── models/                # Model implementations
│   │   ├── teacher/          # Teacher models (ViT, EfficientNet)
│   │   ├── tinycnn.py        # Lightweight CNN
│   │   ├── crnn.py           # CNN + RNN
│   │   └── crnn_cbam.py      # CRNN + CBAM attention
│   ├── training/              # Training scripts
│   │   ├── train_teacher.py  # Train teachers
│   │   ├── train_kd.py       # KD training
│   │   └── train_baseline.py # Baseline training
│   ├── evaluation/            # Metrics and evaluation
│   └── utils/                 # Utilities
├── configs/                    # Configuration files
├── results/                    # Experiment outputs (gitignored)
│   ├── runs/                 # Individual run outputs
│   ├── leaderboard.csv       # All results
│   └── run_steps.csv         # Processing steps log
└── logs/                      # Training logs (gitignored)
```

---

## Dataset

DreamCatcher dataset (introduced in NeurIPS 2024, Datasets & Benchmarks Track)

- Modality (public release): in-ear audio
- Setting: overnight sleep recordings with co-sleepers
- Key property: wearer-aware event annotations
- Config: `sleep_event_classification` (automatically used by the code)
- Source: [THU-PI-Sensing/DreamCatcher](https://huggingface.co/datasets/THU-PI-Sensing/DreamCatcher) on HuggingFace

Note: The public version used in this repository contains audio data only.
Multimodal pipelines are implemented in a modular way but require explicit dataset
access approval.

---

## Scope

**Relation to the original DreamCatcher work.**  
The original DreamCatcher study introduces the dataset and formulates the
wearer-aware sleep event recognition problem.  
This repository builds on that foundation by focusing on **systematic model
benchmarking under wearable constraints**, including attention mechanisms,
knowledge distillation, and explicit evaluation of accuracy–efficiency trade-offs
(parameter count and CPU latency).

This repository focuses on:

- Segment-level audio event classification on earables (event_label)
- Wearer-aware classification under real-world interference
- Lightweight model design (TinyCNN, CRNN)
- Attention mechanisms (CBAM)
- Knowledge distillation (teacher–student)
- Resource-aware evaluation (parameter count and CPU latency)

The repository does not target speech recognition or ASR.

---

## Models

**Student models (edge-oriented):**
- TinyCNN
- CRNN
- CRNN + CBAM

**Teacher models (training only):**
- Vision Transformer (ViT): google/vit-base-patch16-224 (~86M params)
- EfficientNet-B0: google/efficientnet-b0 (~5M params)

Both teachers use pretrained vision models fine-tuned on audio spectrograms.

*(Teacher models are used during knowledge distillation training only.)*

---

## Reproducibility

All experiments are scriptable and log results to `results/leaderboard.csv`.

**Baselines and attention models:**
```bash
bash experiments/run_audio_benchmark.sh
```
**Quick smoke run (fast end-to-end sanity check):**
```bash
bash experiments/run_audio_smoke.sh
```
**Knowledge distillation:**
```bash
bash experiments/run_kd.sh
```
**KD smoke run:**
```bash
bash experiments/run_kd_smoke.sh
```

### Hyperparameters (KD / tuning)

At the moment, this repository does **not** include an automated sweep/grid-search runner.
KD hyperparameters (e.g., `alpha`, `tau`, `lr`, `batch_size`) are set **explicitly** in the experiment scripts
(`experiments/run_kd*.sh`) and passed through to the training entrypoint (`src/training/train_kd.py`).

**Recommended workflow today (manual sweep):**
- Duplicate/loop over settings in a bash script (e.g., nested loops over `alpha` and `tau`)
- Give each run a unique `--run_name`
- Compare runs via `results/leaderboard.csv` (and per-run `results/runs/<run_name>/metrics.json`)

**TODO (planned):** add a first-class sweep command, e.g. `earable sweep kd --alpha 0.3,0.5,0.7 --tau 2,5,10 ...`,
to run combinations and log them consistently.
---

## Sweeps / Grid Search

You can run a simple grid-search style sweep for KD via the `earable` CLI (runs are logged to the same leaderboard and per-run artifacts):

```bash
# Example KD sweep (smoke)
earable sweep kd \
  --students crnn,crnn_cbam \
  --alpha 0.3,0.7 \
  --tau 2,5 \
  --lr 1e-3 \
  --batch_size 4 \
  --dataset_mode smoke \
  --max_samples 512 \
  --jobs 2
```

Tip: add `--dry_run` to print the commands without running them.

Note: when using `--jobs > 1`, runs will append to `results/leaderboard.csv` concurrently; the repo uses a simple file lock
to prevent CSV corruption.

### Summarize what you tried (after a sweep)

Runs write per-run artifacts to `results/runs/<run_name>/`. You can summarize a sweep by prefix:

```bash
# Example: summarize the most recent sweep (replace prefix)
earable summarize --prefix kd_sweep_ --metric f1_macro --topk 10
```

## Dataset Analysis & Visualization

### Class Distribution

Visualize the class distribution in the DreamCatcher dataset (all splits combined):

```bash
python scripts/plot_distribution.py
```

Generates two high-quality visualizations saved to `results/visualizations/`:
- **class_distribution_bar_total.png** - Bar chart with sample counts and percentages
- **class_distribution_pie_total.png** - Pie chart with percentage breakdown

**Example output (total: 447,648 samples):**
```
  quiet               :  210,546 samples ( 47.03%)
  breathe             :  141,885 samples ( 31.70%)
  non_wearer          :   42,013 samples (  9.39%)
  snore               :   38,611 samples (  8.63%)
  bruxism             :    8,023 samples (  1.79%)
  movements           :    3,569 samples (  0.80%)
  swallow             :    2,179 samples (  0.49%)
  somniloquy          :      714 samples (  0.16%)
  cough               :       98 samples (  0.02%)
```

**Custom confusion matrix:**
```bash
python scripts/plot_distribution.py --confusion-matrix <path>
```

## Key Takeaways

- Lightweight CRNN models benefit significantly from attention mechanisms in earable audio event recognition.
- Knowledge distillation narrows the performance gap between compact students and stronger pretrained teachers.
- Benchmarking accuracy together with latency and parameter count is critical for realistic wearable deployment scenarios.

---