from __future__ import annotations

import csv
import hashlib
import json
import random
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.utils.benchmarking import _file_lock


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def select_device() -> tuple[torch.device, str]:
    if torch.cuda.is_available():
        return torch.device("cuda"), "CUDA GPU"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "Apple MPS GPU"
    return torch.device("cpu"), "CPU"


def write_csv_row(path: str | Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = str(path) + ".lock"
    with _file_lock(lock_path):
        exists = path.exists() and path.stat().st_size > 0
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow(row)


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def git_snapshot() -> dict[str, Any]:
    def _run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                list(args),
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            return None

    sha = _run("git", "rev-parse", "HEAD")
    branch = _run("git", "rev-parse", "--abbrev-ref", "HEAD")
    dirty = None
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        dirty = bool(status.strip())
    except Exception:
        pass
    return {
        "commit_sha": sha,
        "branch": branch,
        "dirty": dirty,
    }


def dataset_fingerprint_from_cached_dataset(dataset: Any) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    h5file = getattr(dataset, "h5file", None)
    if h5file is None:
        return attrs
    for k, v in h5file.attrs.items():
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
        else:
            attrs[k] = v
    attrs["fingerprint_ts_utc"] = utc_now_iso()
    return attrs


def rng_state_dict() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_random_state_all"] = torch.cuda.get_rng_state_all()
    return state
