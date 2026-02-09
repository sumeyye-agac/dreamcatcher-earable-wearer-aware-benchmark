import csv
import os
from contextlib import contextmanager

import torch

LEADERBOARD_FIELDNAMES = [
    "schema_version",
    "run_id",
    "run_name",
    "task_name",
    "model",
    "teacher_model",
    "ts_start_utc",
    "ts_end_utc",
    "duration_s",
    "seed",
    "config_hash",
    "git_commit_sha",
    "git_dirty",
    "device",
    "device_name",
    "epochs",
    "epochs_ran",
    "early_stop_patience",
    "early_stop_min_delta",
    "early_stopped",
    "stopped_epoch",
    "best_epoch",
    "lr",
    "weight_decay",
    "batch_size",
    "optimizer",
    "scheduler",
    "sr",
    "n_mels",
    "clip_seconds",
    "class_weights",
    "dataset_commit",
    "cache_fingerprint",
    "best_val_loss",
    "best_val_f1",
    "best_val_acc",
    "best_val_precision_macro",
    "best_val_recall_macro",
    "best_val_balanced_acc",
    "test_loss",
    "test_f1",
    "test_acc",
    "test_precision_macro",
    "test_recall_macro",
    "test_balanced_acc",
    "params",
    "teacher_params",
    "compression_ratio",
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
    "scheduler_checkpoint",
    "rng_checkpoint",
    "rnn_hidden",
    "rnn_layers",
    "cbam_reduction",
    "cbam_sa_kernel",
    "att_mode",
    "alpha",
    "tau",
]


def count_params(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Rough on-device footprint estimate: parameters + buffers in MB.
    (Assumes tensors are stored densely; excludes framework overhead.)
    """
    n_bytes = 0
    for p in model.parameters():
        n_bytes += p.numel() * p.element_size()
    for b in model.buffers():
        n_bytes += b.numel() * b.element_size()
    return n_bytes / (1024 * 1024)


def measure_cpu_latency(
    model: torch.nn.Module, input_shape: tuple, n_warmup: int = 10, n_runs: int = 100
) -> float:
    """
    Measure CPU inference latency in milliseconds.

    Args:
        model: PyTorch model
        input_shape: Input shape tuple (batch_size, channels, height, width)
        n_warmup: Number of warmup runs
        n_runs: Number of runs for averaging

    Returns:
        Average latency in milliseconds
    """
    model.eval()
    device = torch.device("cpu")
    model = model.to(device)

    # Create dummy input
    dummy_input = torch.randn(input_shape).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_input)

    # Measure
    import time

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.time()
            _ = model(dummy_input)
            end = time.time()
            times.append((end - start) * 1000)  # Convert to milliseconds

    return sum(times) / len(times)


@contextmanager
def _file_lock(lock_path: str):
    """
    Cross-platform best-effort file lock.
    Uses:
      - fcntl.flock on POSIX
      - msvcrt.locking on Windows
    """
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    f = open(lock_path, "a+")
    try:
        if os.name == "nt":
            import msvcrt

            # lock 1 byte region
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        finally:
            f.close()


def _normalize_leaderboard_csv(csv_path: str, fieldnames: list[str]) -> None:
    """
    Ensure `csv_path` exists and has the provided header.

    Supports upgrading:
    - headerless files (assumes legacy column order is a prefix of fieldnames)
    - older headered files (pads rows to new schema)
    """
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(fieldnames)
        return

    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(fieldnames)
        return

    first = rows[0]
    if first == fieldnames:
        return

    # Determine if the first row is a header (contains run_name) or data.
    if "run_name" in first:
        old_header = first
        data_rows = rows[1:]
    else:
        # Headerless legacy file: assume the legacy schema is a prefix of current schema.
        old_header = fieldnames[: len(first)]
        data_rows = rows

    old_set = set(old_header)
    new_set = set(fieldnames)
    if not old_set.issubset(new_set):
        # Unknown schema: don't try to rewrite.
        return

    # Rewrite with new header and padded rows.
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in data_rows:
            d = {k: "" for k in fieldnames}
            for i, k in enumerate(old_header):
                if i < len(r):
                    d[k] = r[i]
            w.writerow(d)


def append_to_leaderboard(csv_path: str, row: dict):
    """
    Append a row to the leaderboard CSV file.
    Creates the file with headers if it doesn't exist.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Concurrency-safe: take a lock for normalize + append.
    lock_path = csv_path + ".lock"
    with _file_lock(lock_path):
        _normalize_leaderboard_csv(csv_path, LEADERBOARD_FIELDNAMES)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LEADERBOARD_FIELDNAMES)
            writer.writerow(row)
