# DreamCatcher Earable Wearer-Aware Benchmark

[![Status](https://img.shields.io/badge/status-under%20development-yellow.svg)](#)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-DreamCatcher-lightgrey.svg)](https://huggingface.co/datasets/THU-PI-Sensing/DreamCatcher)
[![Paper](https://img.shields.io/badge/NeurIPS%202024-Paper-b31b1b.svg)](https://dl.acm.org/doi/10.5555/3737916.3740620)

> **Note:** This benchmark is under active development. Results and experiment configurations may be updated as new runs complete.

Most people who snore have no idea they do, and by the time a sleep disorder is caught, it has often gone unnoticed for years. Clinical diagnosis requires an overnight stay in a lab, which is expensive, intrusive, and impractical for routine screening. Earables (lightweight in-ear devices) can change this by passively capturing respiratory audio while the wearer sleeps, but only if the models are small and accurate enough to run directly on the device without streaming private audio to the cloud.

This repository benchmarks lightweight classifiers on three sleep-relevant sound events (`quiet`, `breathe`, `snore`) using the [DreamCatcher dataset (NeurIPS 2024)](https://dl.acm.org/doi/10.5555/3737916.3740620). The focus is on the trade-off between model size and classification performance through attention mechanisms (CBAM) and knowledge distillation.

## What This Repository Focuses On

- Lightweight models for wearable inference (`TinyCNN`, `CRNN`)
- Attention ablations (CBAM on small models only: `TinyCNN_CBAM`)
- Knowledge distillation from stronger teacher models
- Reproducible experiment artifacts (`metrics.json`, confusion matrix CSV, leaderboard)
- Practical deployment metrics (parameter count, model size, CPU latency)

## Results Snapshot (Best-Only Narrative)

- Campaign status: **Phase-1 = 11/11 complete**, **Phase-2 KD = 18/18 complete**
- Full run-level source of truth: `results/leaderboard.csv`

### Model Journey (Best-Only)

`delta_vs_tinycnn_baseline` is computed against `p1_tinycnn_seed42`.

| stage | run_name | params | test_f1 | test_acc | delta_vs_tinycnn_baseline |
|---|---|---:|---:|---:|---:|
| Teacher baseline (CRNN) | `p1_crnn_seed42` | 73,411 | 82.82% | 86.12% | +5.95pp |
| Student baseline (TinyCNN) | `p1_tinycnn_seed42` | 23,491 | 76.87% | 80.57% | 0.00pp |
| Student + CBAM baseline (rr8/sk3) | `p1_tinycnn_cbam_rr8_sk3_seed42` | 23,801 | 79.77% | 82.74% | +2.90pp |
| Best KD on TinyCNN | `p2_kd_tinycnn_a0p6_t5_seed42` | 23,491 | 78.87% | 82.23% | +2.00pp |
| Best KD on TinyCNN_CBAM | `p2_kd_tinycnn_cbam_rr8_sk3_a0p9_t3_seed42` | 23,801 | 81.71% | 84.95% | +4.84pp |

Interpretation:
- CBAM gives a strong lift over TinyCNN baseline (+2.90pp test F1).
- KD improves both student tracks in best settings (TinyCNN and TinyCNN_CBAM).
- Best overall compact student is KD+CBAM (`p2_kd_tinycnn_cbam_rr8_sk3_a0p9_t3_seed42`) with +4.84pp over TinyCNN baseline.
- For this campaign, combining architectural attention (CBAM) and distillation yields the strongest small-model result.

### Parameter Exploration Caveat

This repo includes a broad KD sweep (`alpha x tau`, 18 runs).  
README shows only the best-path narrative; full sensitivity and run-level comparisons are in:
- `notebooks/results_analysis.ipynb`
- `results/leaderboard.csv`

### Detailed Run-Level Comparisons

All run-level comparisons are in `notebooks/results_analysis.ipynb`.

## Dataset

Based on: [Wang, Zeyu, et al. "DreamCatcher: A Wearer-aware Multi-modal Sleep Event Dataset Based on Earables in Non-restrictive Environments." *NeurIPS 2024 (Dec)*.](https://dl.acm.org/doi/10.5555/3737916.3740620)

- Source: [THU-PI-Sensing/DreamCatcher](https://huggingface.co/datasets/THU-PI-Sensing/DreamCatcher)
- Task config: `sleep_event_classification`
- This repo uses a filtered 3-class subset: `quiet`, `breathe`, `snore`
- Audio features: 64-band log-mel spectrograms at 16 kHz
- Class weights: `1.0,1.5,5.5` (validated against current class imbalance; inverse-frequency approximation)

## Quick Start

### 1. Environment Setup

```bash
python3 -m venv dream-env
source dream-env/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

### 2. Authenticate for Gated Dataset Access

```bash
hf auth login
```

### 3. Build Spectrogram Cache

```bash
python3 scripts/preprocess.py
```

This generates:

- `results/cache/spectrograms/train.h5`
- `results/cache/spectrograms/validation.h5`
- `results/cache/spectrograms/test.h5`

### 4. Run a Baseline Training

```bash
python3 -m src.training.train \
  --model tinycnn \
  --epochs 50 \
  --batch_size 64 \
  --lr 1e-3 \
  --early_stop_patience 5 \
  --class_weights 1.0,1.5,5.5 \
  --run_name p1_tinycnn_seed42
```

## Reproduce Core Experiments

### Train an Improved Teacher (`CRNN`)

```bash
python3 -m src.training.train \
  --model crnn \
  --epochs 50 \
  --batch_size 64 \
  --lr 1e-3 \
  --weight_decay 1e-5 \
  --grad_clip 0.0 \
  --early_stop_patience 5 \
  --class_weights 1.0,1.5,5.5 \
  --rnn_hidden 64 \
  --rnn_layers 2 \
  --run_name p1_crnn_seed42
```

### Distill to a Smaller Student (`TinyCNN`)

```bash
python3 -m src.training.train_kd \
  --student_model tinycnn \
  --teacher_model crnn \
  --teacher_checkpoint results/runs/p1_crnn_seed42/checkpoints/best_model.pth \
  --temperature 5.0 \
  --alpha 0.6 \
  --epochs 50 \
  --batch_size 64 \
  --lr 1e-3 \
  --early_stop_patience 5 \
  --class_weights 1.0,1.5,5.5 \
  --run_name p2_kd_tinycnn_a0p6_t5_seed42
```

## Automated End-to-End Manifest

Run the full reproducible pipeline (phase-1 baselines/CBAM, teacher gap gate, then KD):

```bash
python3 scripts/run_experiment_manifest.py \
  --manifest experiments/manifest_repro_v1.json \
  --resume
```

Fresh clean replay (optional):

```bash
python3 scripts/run_experiment_manifest.py \
  --manifest experiments/manifest_repro_v1.json \
  --fresh-start
```

Notes:

- Uses one seed (`42`) by default.
- Single-stage execution: all 11 Phase-1 runs train at `50` epochs (`early_stop_patience=5`).
  - 2 baselines: `crnn`, `tinycnn`
  - 9 CBAM combinations: `tinycnn_cbam` (3 reduction x 3 kernel)
- KD full stage: `50` epochs for **all KD combinations** (no KD top-k pruning).
- Results logged to `results/leaderboard.csv`.
- KD starts only if `CRNN` exceeds the best student by at least `0.03` validation macro-F1.
- If gap is below `0.03`, teacher tuning runs automatically before KD.

Dry-run plan preview:

```bash
python3 scripts/run_experiment_manifest.py \
  --manifest experiments/manifest_repro_v1.json \
  --dry-run
```

## Device Behavior

- Training device priority is: **CUDA -> MPS -> CPU**
- `pin_memory` is enabled only on CUDA (disabled on MPS/CPU to avoid unnecessary warnings and overhead)
- The default manifest (`experiments/manifest_repro_v1.json`) keeps portable auto-device behavior.
- Optional local workflow: use a local manifest with `runtime_policy.preferred_device="mps"` and CPU fallback confirmation settings if you want an explicit prompt before CPU fallback.

## Reproducibility Contract (Quick Audit)

Run these checks from repo root:

```bash
python3 scripts/check_consistency.py --strict
python3 scripts/run_experiment_manifest.py --manifest experiments/manifest_repro_v1.json --dry-run
```

Expected outcome:

- Policy totals match across manifest and docs (Phase-1=11, KD=18).
- KD alpha/temperature grids match across manifest and docs.
- No stale policy phrases in README/decision log.

## Negative Results Policy

Negative or dominated outcomes are documented, not hidden:

- `docs/negative_results.md`

Each entry links experiment evidence (run name + artifact path) and the resulting decision.

## Consistency Policy

Canonical experiment policy lives in `experiments/manifest_repro_v1.json`.

- Docs must mirror manifest policy totals and KD grid.
- Any policy update requires updating docs in the same commit.
- `scripts/check_consistency.py --strict` is the pre-push gate.

<!-- consistency: phase1_total=11 kd_total=18 kd_alphas=0.3,0.6,0.9 kd_taus=3.0,4.0,5.0 -->

## Artifacts and Tracking

For each run, this repo writes (mandatory contract):

- `results/runs/<run_name>/metrics.json`
- `results/runs/<run_name>/epoch_metrics.csv`
- `results/runs/<run_name>/test_metrics.csv`
- `results/runs/<run_name>/class_metrics.csv`
- `results/runs/<run_name>/test_confusion_matrix.csv`
- `results/runs/<run_name>/checkpoints/best_model.pth`
- `results/runs/<run_name>/checkpoints/last_model.pth`
- `results/runs/<run_name>/checkpoints/optimizer_last.pth`
- `results/runs/<run_name>/checkpoints/rng_state.pth`
- `results/runs/<run_name>/resolved_config.json`
- `results/runs/<run_name>/env.json`
- `results/runs/<run_name>/git.json`
- `results/runs/<run_name>/data_fingerprint.json`
- `results/runs/<run_name>/early_stop.json`

Global tracking files:

- `results/leaderboard.csv`
- `results/run_steps.csv`

Contract documentation:

- `docs/artifact_contract.md`
- `src/utils/csv_schemas.py` (mandatory CSV header definitions)
- `docs/decision_log.md` (project-level experiment decisions)

Execution policy decision (pinned):

- Single-stage: all Phase-1 runs (baselines + CBAM) train at full `50` epochs with early stopping.
- CBAM is applied only to small models (`tinycnn_cbam`). CRNN is the teacher and does not use CBAM.
- For KD stage, run all KD combinations (no KD top-k pruning).
- Start KD only after teacher gap check (`>= 0.03` macro-F1).

## Repository Layout

```text
dreamcatcher-earable-wearer-aware-benchmark/
├── scripts/
│   └── preprocess.py
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── utils/
├── experiments/
├── results/
│   ├── cache/
│   ├── runs/
│   └── leaderboard.csv
└── logs/
```
