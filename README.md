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

## Results Snapshot (Phase-1 + Phase-2 Completed)

> Full leaderboard: `results/leaderboard.csv` (Phase-1: 11/11, Phase-2 KD: 18/18)

### Baselines

| Model | Params | Size (MB) | Val F1 (macro) | Val Acc | Test F1 (macro) | Test Acc |
|-------|--------|-----------|--------|---------|---------|----------|
| CRNN (teacher) | 73.4K | 0.28 | 79.57% | 83.45% | 82.82% | 86.12% |
| TinyCNN | 23.5K | 0.09 | 73.69% | 77.72% | 76.87% | 80.57% |

### Best CBAM Configurations

| Model | CBAM (rr, sk) | Params | Size (MB) | Val F1 (macro) | Val Acc | Test F1 (macro) | Test Acc |
|-------|---------------|--------|-----------|--------|---------|---------|----------|
| TinyCNN + CBAM | rr8, sk3 | 23.8K | 0.09 | 75.37% | 78.83% | 79.77% | 82.74% |

### KD Campaign Summary

- Teacher (`CRNN`) test F1 / acc: **82.82% / 86.12%**
- Best KD for `tinycnn`: `p2_kd_tinycnn_a0p6_t5_seed42` -> **78.87% F1**, **82.23% acc**
- Best KD for `tinycnn_cbam`: `p2_kd_tinycnn_cbam_rr8_sk3_a0p9_t3_seed42` -> **81.71% F1**, **84.95% acc**

## KD vs Non-KD Decision Table

Source: `results/leaderboard.csv`  
Teacher reference: `p1_crnn_seed42` (test F1 = 82.82%)

Baseline references:

- `tinycnn` baseline (`p1_tinycnn_seed42`): test F1 = 76.87%, test acc = 80.57%
- `tinycnn_cbam` baseline (`p1_tinycnn_cbam_rr8_sk3_seed42`): test F1 = 79.77%, test acc = 82.74%

### TinyCNN KD Runs (Non-CBAM)

| run_name | teacher_test_f1 | kd_test_f1 | delta_test_f1_vs_baseline | kd_test_acc | delta_test_acc_vs_baseline | teacher_gap_closed |
|---|---:|---:|---:|---:|---:|---:|
| `p2_kd_tinycnn_a0p6_t5_seed42` | 82.82% | 78.87% | 2.00% | 82.23% | 1.67% | 33.63% |
| `p2_kd_tinycnn_a0p9_t3_seed42` | 82.82% | 78.48% | 1.61% | 82.17% | 1.60% | 27.07% |
| `p2_kd_tinycnn_a0p3_t4_seed42` | 82.82% | 78.44% | 1.57% | 82.03% | 1.46% | 26.35% |
| `p2_kd_tinycnn_a0p6_t3_seed42` | 82.82% | 78.39% | 1.52% | 82.13% | 1.57% | 25.56% |
| `p2_kd_tinycnn_a0p3_t3_seed42` | 82.82% | 78.13% | 1.26% | 82.01% | 1.44% | 21.13% |
| `p2_kd_tinycnn_a0p9_t5_seed42` | 82.82% | 78.12% | 1.25% | 81.89% | 1.32% | 21.01% |
| `p2_kd_tinycnn_a0p3_t5_seed42` | 82.82% | 77.33% | 0.46% | 81.37% | 0.80% | 7.73% |
| `p2_kd_tinycnn_a0p6_t4_seed42` | 82.82% | 77.19% | 0.32% | 81.33% | 0.77% | 5.40% |
| `p2_kd_tinycnn_a0p9_t4_seed42` | 82.82% | 76.65% | -0.22% | 80.97% | 0.40% | -3.69% |

### TinyCNN_CBAM KD Runs

| run_name | teacher_test_f1 | kd_test_f1 | delta_test_f1_vs_baseline | kd_test_acc | delta_test_acc_vs_baseline | teacher_gap_closed |
|---|---:|---:|---:|---:|---:|---:|
| `p2_kd_tinycnn_cbam_rr8_sk3_a0p9_t3_seed42` | 82.82% | 81.71% | 1.95% | 84.95% | 2.21% | 63.69% |
| `p2_kd_tinycnn_cbam_rr8_sk3_a0p6_t5_seed42` | 82.82% | 81.40% | 1.63% | 84.52% | 1.79% | 53.36% |
| `p2_kd_tinycnn_cbam_rr8_sk3_a0p9_t5_seed42` | 82.82% | 81.36% | 1.60% | 84.63% | 1.89% | 52.19% |
| `p2_kd_tinycnn_cbam_rr8_sk3_a0p6_t3_seed42` | 82.82% | 81.14% | 1.38% | 84.38% | 1.65% | 45.06% |
| `p2_kd_tinycnn_cbam_rr8_sk3_a0p6_t4_seed42` | 82.82% | 81.12% | 1.35% | 84.54% | 1.80% | 44.25% |
| `p2_kd_tinycnn_cbam_rr8_sk3_a0p3_t4_seed42` | 82.82% | 80.78% | 1.01% | 84.00% | 1.26% | 33.08% |
| `p2_kd_tinycnn_cbam_rr8_sk3_a0p9_t4_seed42` | 82.82% | 80.35% | 0.59% | 83.90% | 1.16% | 19.23% |
| `p2_kd_tinycnn_cbam_rr8_sk3_a0p3_t5_seed42` | 82.82% | 80.19% | 0.42% | 83.29% | 0.55% | 13.76% |
| `p2_kd_tinycnn_cbam_rr8_sk3_a0p3_t3_seed42` | 82.82% | 80.09% | 0.32% | 83.44% | 0.70% | 10.57% |

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
