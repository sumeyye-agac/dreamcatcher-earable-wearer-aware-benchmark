# Mandatory Artifact Contract

This repository enforces a strict per-run artifact contract for reproducibility.

- Contract version: `2026-02-07.v1`
- Schema version: `v2`
- Enforcement points: `src/training/train.py`, `src/training/train_kd.py`
- Validator module: `src/utils/artifact_contract.py`

## Required Run Files

For every run `results/runs/<run_name>/`, the following files must exist:

- `resolved_config.json`
- `env.json`
- `git.json`
- `data_fingerprint.json`
- `early_stop.json`
- `metrics.json`
- `epoch_metrics.csv`
- `test_metrics.csv`
- `class_metrics.csv`
- `test_confusion_matrix.csv`
- `checkpoints/best_model.pth`
- `checkpoints/last_model.pth`
- `checkpoints/optimizer_last.pth`
- `checkpoints/rng_state.pth`

## Mandatory CSV Schema (Locked)

The following CSV files are schema-locked and validated before leaderboard write:

- `epoch_metrics.csv`
: must match one of:
: `TRAIN_EPOCH_METRICS_FIELDNAMES` or `KD_EPOCH_METRICS_FIELDNAMES` in `src/utils/csv_schemas.py`
- `test_metrics.csv`
: must match `TEST_METRICS_FIELDNAMES` in `src/utils/csv_schemas.py`
- `class_metrics.csv`
: must match `CLASS_METRICS_FIELDNAMES` in `src/utils/csv_schemas.py`

Each required CSV must contain header + at least one data row.

## Required `metrics.json` Structure

Top-level required keys:

- `schema_version` (`"v2"`)
- `run_id`
- `run_name`
- `task_name`
- `ts_start_utc`
- `ts_end_utc`
- `duration_s`
- `seed`
- `config_hash`
- `device`
- `device_name`
- `model`
- `optimizer`
- `scheduler`
- `hyperparameters`
- `best_epoch`
- `epochs_ran`
- `early_stop`
- `validation_best`
- `validation_current`
- `test`
- `params`
- `model_size_mb`
- `cpu_latency_ms`
- `data_fingerprint`
- `artifacts`

Required nested keys:

- `early_stop`: `enabled`, `patience`, `min_delta`, `early_stopped`, `stopped_epoch`, `best_epoch`, `best_val_f1`
- `validation_best`, `validation_current`, `test`: `loss`, `acc`, `balanced_acc`, `f1_macro`, `precision_macro`, `recall_macro`
- `artifacts`: `epoch_metrics_csv`, `test_metrics_csv`, `class_metrics_csv`, `test_confusion_matrix_csv`, `best_checkpoint`, `last_checkpoint`, `optimizer_checkpoint`, `rng_checkpoint`

All artifact paths in `metrics.json["artifacts"]` must resolve to existing files.

## Required Leaderboard Row Contract

Every row appended to `results/leaderboard.csv` must include the full schema in:

- `src/utils/benchmarking.py`: `LEADERBOARD_FIELDNAMES`

Additionally, critical fields must be non-empty (run identity, timing, device, core hyperparameters, artifact paths).

## Fail-Fast Behavior

Before a row is appended to leaderboard:

1. Run artifacts are validated.
2. CSV headers are validated against mandatory schemas.
3. `metrics.json` schema is validated.
4. Leaderboard row schema/non-empty requirements are validated.

If any contract check fails, training exits with an explicit error and does **not** write an invalid leaderboard row.
