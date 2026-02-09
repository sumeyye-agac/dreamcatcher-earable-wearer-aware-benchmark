from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def run_dir(run_name: str, base: str = "results/runs") -> Path:
    p = Path(base) / run_name
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(obj):
        obj = asdict(obj)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def _get_git_commit() -> str | None:
    """Get current git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
    except Exception:
        return None


def _get_git_branch() -> str | None:
    """Get current git branch."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
    except Exception:
        return None


def env_snapshot() -> dict[str, Any]:
    """
    Capture full environment snapshot for reproducibility.

    Returns dict with Python version, PyTorch version, CUDA, platform,
    git info, and package versions.
    """
    import torch

    snap: dict[str, Any] = {}

    # Python environment
    snap["python_version"] = sys.version
    snap["python_implementation"] = platform.python_implementation()

    # PyTorch environment
    snap["torch_version"] = torch.__version__
    snap["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        snap["cuda_version"] = torch.version.cuda
        snap["cudnn_version"] = torch.backends.cudnn.version()
    snap["mps_available"] = torch.backends.mps.is_available()

    # Platform info
    snap["platform"] = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

    # Git info for code version
    snap["git_commit"] = _get_git_commit()
    snap["git_branch"] = _get_git_branch()

    # Package versions (key dependencies)
    try:
        import numpy
        snap["numpy_version"] = numpy.__version__
    except ImportError:
        pass

    try:
        import librosa
        snap["librosa_version"] = librosa.__version__
    except ImportError:
        pass

    try:
        import pandas
        snap["pandas_version"] = pandas.__version__
    except ImportError:
        pass

    try:
        import sklearn
        snap["sklearn_version"] = sklearn.__version__
    except ImportError:
        pass

    # Environment variables (sanitized - no secrets)
    env_keys = [
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HF_DATASETS_CACHE",
        "PYTHONHASHSEED",
        "CUBLAS_WORKSPACE_CONFIG",
    ]
    snap["env_vars"] = {}
    for k in env_keys:
        if k in os.environ:
            # Sanitize tokens
            if "TOKEN" in k:
                snap["env_vars"][k] = "***set***"
            else:
                snap["env_vars"][k] = os.environ[k]

    return snap
