from __future__ import annotations

import json
import os
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


def env_snapshot() -> dict[str, Any]:
    # Keep it minimal (no secrets).
    keys = [
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HF_DATASETS_CACHE",
    ]
    snap: dict[str, Any] = {}
    for k in keys:
        if k in os.environ:
            snap[k] = "***set***"
    return snap
