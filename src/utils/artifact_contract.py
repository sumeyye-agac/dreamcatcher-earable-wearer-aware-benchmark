from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.utils.benchmarking import LEADERBOARD_FIELDNAMES
from src.utils.csv_schemas import (
    CLASS_METRICS_FIELDNAMES,
    KD_EPOCH_METRICS_FIELDNAMES,
    TEST_METRICS_FIELDNAMES,
    TRAIN_EPOCH_METRICS_FIELDNAMES,
)

ARTIFACT_CONTRACT_VERSION = "2026-02-07.v1"

# Files that must exist for every completed run under results/runs/<run_name>.
REQUIRED_RUN_FILES = (
    "resolved_config.json",
    "env.json",
    "git.json",
    "data_fingerprint.json",
    "early_stop.json",
    "metrics.json",
    "epoch_metrics.csv",
    "test_metrics.csv",
    "class_metrics.csv",
    "test_confusion_matrix.csv",
    "checkpoints/best_model.pth",
    "checkpoints/last_model.pth",
    "checkpoints/optimizer_last.pth",
    "checkpoints/rng_state.pth",
)

REQUIRED_METRICS_KEYS = (
    "schema_version",
    "run_id",
    "run_name",
    "task_name",
    "ts_start_utc",
    "ts_end_utc",
    "duration_s",
    "seed",
    "config_hash",
    "device",
    "device_name",
    "model",
    "optimizer",
    "scheduler",
    "hyperparameters",
    "best_epoch",
    "epochs_ran",
    "early_stop",
    "validation_best",
    "validation_current",
    "test",
    "params",
    "model_size_mb",
    "cpu_latency_ms",
    "data_fingerprint",
    "artifacts",
)

REQUIRED_EARLY_STOP_KEYS = (
    "enabled",
    "patience",
    "min_delta",
    "early_stopped",
    "stopped_epoch",
    "best_epoch",
    "best_val_f1",
)

REQUIRED_SPLIT_METRIC_KEYS = (
    "loss",
    "acc",
    "balanced_acc",
    "f1_macro",
    "precision_macro",
    "recall_macro",
)

REQUIRED_ARTIFACT_MAP_KEYS = (
    "epoch_metrics_csv",
    "test_metrics_csv",
    "class_metrics_csv",
    "test_confusion_matrix_csv",
    "best_checkpoint",
    "last_checkpoint",
    "optimizer_checkpoint",
    "rng_checkpoint",
)

REQUIRED_LEADERBOARD_NONEMPTY_KEYS = (
    "schema_version",
    "run_id",
    "run_name",
    "task_name",
    "ts_start_utc",
    "ts_end_utc",
    "duration_s",
    "seed",
    "config_hash",
    "device",
    "epochs",
    "epochs_ran",
    "best_epoch",
    "lr",
    "weight_decay",
    "batch_size",
    "optimizer",
    "sr",
    "n_mels",
    "params",
    "model_size_mb",
    "cpu_latency_ms",
    "artifact_dir",
    "metrics_json",
    "epoch_metrics_csv",
    "test_metrics_csv",
    "class_metrics_csv",
    "test_cm_csv",
    "best_checkpoint",
    "last_checkpoint",
    "optimizer_checkpoint",
    "rng_checkpoint",
)

REQUIRED_CSV_SCHEMAS = {
    "epoch_metrics.csv": [TRAIN_EPOCH_METRICS_FIELDNAMES, KD_EPOCH_METRICS_FIELDNAMES],
    "test_metrics.csv": [TEST_METRICS_FIELDNAMES],
    "class_metrics.csv": [CLASS_METRICS_FIELDNAMES],
}


def _require_keys(payload: dict[str, Any], keys: tuple[str, ...], context: str) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ValueError(f"{context} missing required keys: {', '.join(missing)}")


def _resolve_path(raw_path: str, run_dir: Path) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    rel_to_run = run_dir / raw_path
    if rel_to_run.exists():
        return rel_to_run
    return path


def _validate_csv_schema(path: Path, allowed_headers: list[list[str]], min_rows: int) -> None:
    with path.open(newline="", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    if not reader:
        raise ValueError(f"CSV is empty: {path}")
    header = reader[0]
    if not any(header == allowed for allowed in allowed_headers):
        allowed_text = " OR ".join([",".join(h) for h in allowed_headers])
        raise ValueError(
            f"CSV header mismatch for {path.name}. "
            f"got={','.join(header)} expected={allowed_text}"
        )
    data_rows = len(reader) - 1
    if data_rows < min_rows:
        raise ValueError(f"CSV has no data rows: {path}")


def validate_leaderboard_row_contract(row: dict[str, Any]) -> None:
    missing_schema = [k for k in LEADERBOARD_FIELDNAMES if k not in row]
    if missing_schema:
        raise ValueError(f"leaderboard row missing schema fields: {', '.join(missing_schema)}")

    empty_required = [
        key for key in REQUIRED_LEADERBOARD_NONEMPTY_KEYS if str(row.get(key, "")).strip() == ""
    ]
    if empty_required:
        raise ValueError(
            "leaderboard row has empty required values: " + ", ".join(empty_required)
        )

    if row.get("schema_version") != "v2":
        raise ValueError("leaderboard row must use schema_version='v2'")


def validate_run_artifact_contract(run_dir: str | Path, metrics_json_path: str | Path) -> None:
    run_dir = Path(run_dir)
    metrics_json_path = Path(metrics_json_path)

    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    missing_files = [p for p in REQUIRED_RUN_FILES if not (run_dir / p).exists()]
    if missing_files:
        raise FileNotFoundError(
            "run is missing required artifacts: " + ", ".join(missing_files)
        )

    for csv_name, allowed_headers in REQUIRED_CSV_SCHEMAS.items():
        _validate_csv_schema(run_dir / csv_name, allowed_headers, min_rows=1)

    if not metrics_json_path.exists():
        raise FileNotFoundError(f"metrics.json not found: {metrics_json_path}")

    with metrics_json_path.open(encoding="utf-8") as f:
        metrics = json.load(f)

    _require_keys(metrics, REQUIRED_METRICS_KEYS, "metrics.json")
    if metrics["schema_version"] != "v2":
        raise ValueError("metrics.json must use schema_version='v2'")

    early_stop = metrics["early_stop"]
    if not isinstance(early_stop, dict):
        raise ValueError("metrics.json early_stop must be an object")
    _require_keys(early_stop, REQUIRED_EARLY_STOP_KEYS, "metrics.json early_stop")

    validation_best = metrics["validation_best"]
    validation_current = metrics["validation_current"]
    test_metrics = metrics["test"]
    for split_name, split_payload in (
        ("validation_best", validation_best),
        ("validation_current", validation_current),
        ("test", test_metrics),
    ):
        if not isinstance(split_payload, dict):
            raise ValueError(f"metrics.json {split_name} must be an object")
        _require_keys(split_payload, REQUIRED_SPLIT_METRIC_KEYS, f"metrics.json {split_name}")

    artifacts = metrics["artifacts"]
    if not isinstance(artifacts, dict):
        raise ValueError("metrics.json artifacts must be an object")
    _require_keys(artifacts, REQUIRED_ARTIFACT_MAP_KEYS, "metrics.json artifacts")

    for artifact_key in REQUIRED_ARTIFACT_MAP_KEYS:
        raw_path = str(artifacts[artifact_key])
        if not raw_path.strip():
            raise ValueError(f"metrics.json artifacts.{artifact_key} is empty")
        artifact_path = _resolve_path(raw_path, run_dir)
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"metrics.json artifacts.{artifact_key} path does not exist: {raw_path}"
            )

    if str(metrics.get("run_name", "")).strip() != run_dir.name:
        raise ValueError(
            f"metrics.json run_name={metrics.get('run_name')} does not match run dir {run_dir.name}"
        )


def enforce_artifact_contract(
    run_dir: str | Path,
    metrics_json_path: str | Path,
    leaderboard_row: dict[str, Any],
) -> None:
    validate_run_artifact_contract(run_dir, metrics_json_path)
    validate_leaderboard_row_contract(leaderboard_row)
