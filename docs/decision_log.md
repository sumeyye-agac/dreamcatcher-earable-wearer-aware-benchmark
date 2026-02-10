# Decision Log

## Logging Rule

- Time format: UTC (`YYYY-MM-DD HH:MM UTC`)
- Scope: important configuration/process decisions and repository-wide changes
- This file is the primary memory for experiment process decisions

## 2026-02-08 Timeline (UTC)

### 2026-02-08 --- Speed Optimization: batch64, num_workers=4

- **batch_size**: 32 → 64 (all training: baseline, CBAM, KD)
- **num_workers**: 0 → 4 (DataLoader parallelism for spectrogram loading)
- **lr**: 0.001 for all models (uniform). Initially considered 0.002 for CRNN (linear scaling), reverted to keep all models on the same lr for simplicity and fair comparison.
- **DataLoader kwargs**: added `pin_memory` (CUDA only), `persistent_workers` (when num_workers > 0)
- **Lazy h5py open**: CachedDataset now opens HDF5 lazily in `__getitem__` (multiprocessing-safe for num_workers > 0)
- **Rationale**: M3 Pro has 11 CPU cores and 18GB unified memory — 4 workers is conservative and safe. Batch 64 better utilizes GPU/MPS compute.
- **Teacher tuning lr grid**: [0.002, 0.001, 0.0005] (0.002 added as upper bound)
- Manifest version bumped to `2026-02-08.v2`
- Updated: `defaults.py`, `train.py`, `train_kd.py`, `manifest_repro_v1.json`, orchestrator

### 2026-02-08 04:30 UTC

- Removed `crnn_cbam` and `crnn_blocks` from the pipeline entirely
- CBAM attention is now small-model-only: `tinycnn_cbam`
- Rationale: CRNN is the teacher model — it doesn't need attention boost. CBAM is a tool to boost small student models.
- Deleted `src/models/crnn_cbam.py` and `src/models/crnn_blocks.py`
- Removed two-stage screening entirely — all Phase-1 runs execute at full 50 epochs with early stopping
- Rationale: without CRNN_CBAM, remaining models are small and train fast; screening overhead is not justified
- New Phase-1 total: 11 runs (2 baselines + 9 CBAM combinations)
- Manifest version bumped to `2026-02-08.v1`
- Updated: manifest, orchestrator, train.py, train_kd.py, README, model docstrings

## 2026-02-07 Timeline (UTC)

### 2026-02-07 13:15 UTC

- Created pre-clean snapshot: `/tmp/dreamcatcher_preclean_20260207_131518`
- Moved old logs/results/experiment scripts into snapshot for rollback safety

### 2026-02-07 13:20 UTC

- Reset repository runtime outputs for clean start
- Recreated empty runtime folders (`results/runs`, `results/cache/spectrograms`, `logs`)

### 2026-02-07 13:30 UTC

- Refactored training pipelines for reproducibility contract
- Enforced canonical feature setup: `n_mels=64`
- Enforced device priority: `CUDA -> MPS -> CPU`
- Added detailed run artifacts and checkpoints (best/last/optimizer/rng)

### 2026-02-07 13:45 UTC

- Added mandatory artifact contract validator
- Added mandatory CSV schema definitions
- Added fail-fast checks before leaderboard writes

### 2026-02-07 14:00 UTC

- Added parameter-count and architecture logging per run
- Added `model_architecture.txt` (and KD `teacher_architecture.txt`)

### 2026-02-07 18:44 UTC

- Added initial automated experiment manifest and orchestrator
- Added teacher-student gap gate (`>= 0.03` macro-F1) before KD

### 2026-02-07 18:50 UTC

- Removed outdated README snapshot claims
- Synced README and notebook with current reproducible pipeline
- Cleared notebook outputs/execution counts for clean repository state

### 2026-02-07 18:56 UTC

- Implemented true two-stage execution in orchestrator
- Stages: screening first, full runs for selected finalists next

### 2026-02-07 19:03 UTC

- Updated runtime policy
- Screening leaderboard separated: `results/leaderboard_screening.csv`
- Resume hardened with config/signature matching (not only artifact existence)
- Screening reduced to `12` epochs, patience `2`
- Removed extra ultra-small model variants from this campaign

### 2026-02-07 19:05 UTC

- KD policy updated
- Keep KD screening stage
- Run **all KD combinations** in KD full stage (no KD top-k pruning)
- Keep `alpha=0.0` in the experiment grid
- Deferred teacher-logits cache optimization for now

### 2026-02-07 19:10 UTC

- Deleted temporary backup snapshot to recover disk
- Removed `/tmp/dreamcatcher_preclean_20260207_131518` (`~16 GB` recovered)

### 2026-02-07 19:31 UTC

- Fixed 3-class index scan bottleneck in `src/data/dreamcatcher_dataset.py`
- Switched index generation to fast label-column scan (`label`/`event_label`/`class`) without audio decode
- Verified train split index generation speed-up and cache refresh (`results/cache/indices_full_train.json`)

### 2026-02-07 19:33 UTC

- Restarted orchestrator with fresh start after scan fix
- Cache build (`scripts/preprocess.py`) is now running with active artifact writes under `results/cache/spectrograms`

### 2026-02-07 23:20 UTC

- Fixed cache preprocessing flow to avoid unnecessary recomputation after interruption
- `scripts/preprocess.py` now supports `--splits` and `--skip-existing`
- Orchestrator cache step now requests only missing splits (instead of rebuilding all splits)

### 2026-02-07 23:22 UTC

- Added cache file integrity checks (shape/attrs/config) before reusing `.h5` artifacts
- Existing `results/cache/spectrograms/train.h5` detected as invalid/incomplete; orchestrator now rebuilds it safely

### 2026-02-07 23:25 UTC

- Added preprocessing throughput optimizations
- Replaced per-sample HDF5 writes with chunked batch writes
- Added configurable precompute knobs: `preprocess_batch_size`, `preprocess_compression`
- Switched cache compression from gzip to fast options; benchmark kept `compression=none` for max throughput
- Added atomic cache write (`.tmp` then replace) to avoid leaving partial final `.h5` files
- Benchmarked precompute chunk sizes on 2,048-sample smoke run; set default to `preprocess_batch_size=128`

### 2026-02-07 23:35 UTC

- Removed CLI-default drift between `train.py` and `train_kd.py`
- Added `src/training/defaults.py` as shared default source
- Unified critical defaults: batch size, early stop patience/min_delta, RNN layers, CBAM reduction/kernel, class weights, seed

## Current Pinned Policy (`experiments/manifest_repro_v1.json`)

- Seed: `42` (single-seed campaign for now)
- Class weights: `1.0,1.5,5.5`
- Single-stage: all Phase-1 runs at `50` epochs (`early_stop_patience=5`)
- Phase-1: 2 baselines (`crnn`, `tinycnn`) + 9 CBAM (`tinycnn_cbam` × 3 reduction × 3 kernel) = 11 runs
- Training: `batch_size=64`, `num_workers=4`, `lr=0.001` for all models
- KD grid: `alpha=[0.0, 0.3, 0.6, 0.9]`, `temperature=[3.0, 4.0, 5.0]` with students (`tinycnn`, `tinycnn_cbam`) → 24 combinations
- KD start condition: teacher gap gate must pass (`>= 0.03` macro-F1)
- Teacher logits cache: postponed
