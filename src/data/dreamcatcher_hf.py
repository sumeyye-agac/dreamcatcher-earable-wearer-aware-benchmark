from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import sys
import os
import shutil
import time
import threading

import numpy as np
from datasets import DownloadConfig, load_dataset_builder
from huggingface_hub import get_token

from .audio_features import compute_log_mel
from src.utils.runlog import StepLogger

# DreamCatcher HF config uses 9 labels for sleep_event_classification.
LABELS = [
    "quiet",
    "non_wearer",
    "bruxism",
    "swallow",
    "somniloquy",
    "breathe",
    "cough",
    "snore",
    "movements",
]
LABEL2ID = {k: i for i, k in enumerate(LABELS)}

_BUILDER_CACHE: dict[tuple[str, str], object] = {}


def _safe_dir_size_bytes(root: str) -> int:
    """
    Best-effort directory size calculator.
    Avoids crashing on permission errors / broken symlinks.
    """
    total = 0
    try:
        with os.scandir(root) as it:
            for entry in it:
                try:
                    if entry.is_symlink():
                        continue
                    if entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
                    elif entry.is_dir(follow_symlinks=False):
                        total += _safe_dir_size_bytes(entry.path)
                except OSError:
                    continue
    except OSError:
        return 0
    return total


def _fmt_gb(n_bytes: int) -> str:
    return f"{n_bytes / 1e9:.2f}GB"


def _get_builder(dataset_mode: str, cache_dir: str, logger: StepLogger | None = None):
    base = "https://huggingface.co/datasets/THU-PI-Sensing/DreamCatcher/resolve/main/"
    dataset_mode = dataset_mode.lower().strip()
    if dataset_mode not in {"full", "smoke"}:
        raise ValueError("dataset_mode must be one of: full, smoke")

    # Important: "smoke" must not reuse the already-prepared "full" cache artifacts.
    # Give it a separate cache root so datasets doesn't silently reuse the full prepared dataset.
    effective_cache_dir = cache_dir if dataset_mode == "full" else os.path.join(cache_dir, "_dreamcatcher_smoke")

    key = (dataset_mode, effective_cache_dir)
    if key in _BUILDER_CACHE:
        if logger is not None:
            logger.log("dataset_builder_cache_hit", detail=f"mode={dataset_mode} cache_dir={effective_cache_dir}")
        return _BUILDER_CACHE[key]

    if dataset_mode == "smoke":
        data_files = {
            "train": base + "data/validation.tar.gz",
            "test": base + "data/validation.tar.gz",
            "validation": base + "data/validation.tar.gz",
        }
        imu_files = {
            "train": base + "imu/validation.tar.gz",
            "test": base + "imu/validation.tar.gz",
            "validation": base + "imu/validation.tar.gz",
        }
    else:
        data_files = {
            "train": base + "data/train.tar.gz",
            "test": base + "data/test.tar.gz",
            "validation": base + "data/validation.tar.gz",
        }
        imu_files = {
            "train": base + "imu/train.tar.gz",
            "test": base + "imu/test.tar.gz",
            "validation": base + "imu/validation.tar.gz",
        }

    download_config = DownloadConfig(cache_dir=effective_cache_dir, token=True)
    if download_config.token is True and not get_token():
        raise RuntimeError(
            "HuggingFace token not found. The DreamCatcher dataset is gated.\n"
            "1) Request access: https://huggingface.co/datasets/THU-PI-Sensing/DreamCatcher\n"
            "2) Login: `hf auth login` (or set env var `HUGGINGFACE_HUB_TOKEN` / `HF_TOKEN`)."
        )

    if logger is not None:
        logger.log("dataset_builder_prepare_start", detail=f"mode={dataset_mode} cache_dir={effective_cache_dir}")
    t0 = time.time()
    builder = load_dataset_builder(
        "THU-PI-Sensing/DreamCatcher",
        name="sleep_event_classification",
        cache_dir=effective_cache_dir,
        download_config=download_config,
    )
    builder.config.data_files = data_files
    builder.config.imu_files = imu_files

    # Heartbeat logging: show that we're still working and roughly how cache grows.
    hb_stop = threading.Event()
    hb_interval_s = float(os.environ.get("DREAMCATCHER_HF_HEARTBEAT_S", "15"))
    cache_start = _safe_dir_size_bytes(effective_cache_dir)

    def _heartbeat_loop():
        last = cache_start
        while not hb_stop.wait(hb_interval_s):
            cur = _safe_dir_size_bytes(effective_cache_dir)
            delta = cur - cache_start
            step_delta = cur - last
            last = cur
            if logger is not None:
                logger.log(
                    "dataset_builder_heartbeat",
                    t0=t0,
                    detail=f"cache={_fmt_gb(cur)} (+{_fmt_gb(delta)} total, +{_fmt_gb(step_delta)} since last)",
                )
            print(
                f"[heartbeat] still downloading/preparing... cache={_fmt_gb(cur)} (+{_fmt_gb(delta)} total)",
                file=sys.stderr,
                flush=True,
            )

    hb_thread = threading.Thread(target=_heartbeat_loop, name="dreamcatcher_hf_heartbeat", daemon=True)
    hb_thread.start()
    try:
        builder.download_and_prepare(download_config=download_config)
    finally:
        hb_stop.set()
        try:
            hb_thread.join(timeout=2.0)
        except RuntimeError:
            pass

    if logger is not None:
        logger.log("dataset_builder_prepare_done", t0=t0)

    _BUILDER_CACHE[key] = builder
    return builder


@dataclass
class DreamCatcherHFAudioConfig:
    sample_rate: int = 16000
    n_mels: int = 64


class DreamCatcherHFAudioDataset:
    """
    HuggingFace-backed DreamCatcher audio dataset.

    Returns:
        x: log-mel spectrogram [n_mels, time]
        y: integer class id
    """

    def __init__(
        self,
        split: str = "train",
        cfg: DreamCatcherHFAudioConfig | None = None,
        dataset_mode: str = "full",
        run_name: str = "",
        steps_csv: str = "results/run_steps.csv",
    ):
        self.cfg = cfg or DreamCatcherHFAudioConfig()
        self._logger = StepLogger(run_name=run_name, csv_path=steps_csv)
        
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Loading DreamCatcher Dataset", file=sys.stderr)
        print(f"  Split: {split}", file=sys.stderr)
        print(f"  Config: sleep_event_classification", file=sys.stderr)
        print(f"  Source: THU-PI-Sensing/DreamCatcher (HuggingFace)", file=sys.stderr)
        print(f"  Dataset Mode: {dataset_mode}", file=sys.stderr)
        if run_name:
            print(f"  Run Name: {run_name}", file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)
        
        cache_dir = os.environ.get("HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets"))
        du = shutil.disk_usage(os.path.expanduser("~"))
        self._logger.log(
            "dataset_init",
            detail=f"split={split} mode={dataset_mode} cache_dir={cache_dir} free_gb={du.free/1e9:.1f}",
        )

        print("Preparing dataset (download + generate). This can take a while on first run...", file=sys.stderr)
        t_prepare = time.time()
        builder = _get_builder(dataset_mode=dataset_mode, cache_dir=cache_dir, logger=self._logger)
        self._logger.log("dataset_builder_ready", t0=t_prepare)

        t_as = time.time()
        self.ds = builder.as_dataset(split=split)
        self._logger.log("dataset_split_loaded", t0=t_as, detail=f"split={split} len={len(self.ds)}")
        print(f"Dataset loaded successfully: {len(self.ds)} samples\n", file=sys.stderr)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        row = self.ds[idx]

        if "audio" in row:
            audio = row["audio"]
        elif "audio_data" in row:
            audio = row["audio_data"]
        else:
            raise KeyError("Expected 'audio' (or legacy 'audio_data') in dataset row.")
        y = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio["sampling_rate"])

        if y.ndim == 2:
            y = y.mean(axis=1)

        if sr != self.cfg.sample_rate:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=self.cfg.sample_rate)

        # Avoid librosa warnings / degenerate spectrograms for extremely short clips
        if y.shape[0] < 1024:
            y = np.pad(y, (0, 1024 - y.shape[0]), mode="constant")

        x = compute_log_mel(y=y, sr=self.cfg.sample_rate, n_mels=self.cfg.n_mels)

        if "label" in row:
            label_val = row["label"]
        elif "event_label" in row:
            label_val = row["event_label"]
        elif "class" in row:
            label_val = row["class"]
        else:
            raise KeyError("Expected 'label' (or 'event_label'/'class') in dataset row.")

        if isinstance(label_val, (int, np.integer)):
            y_id = int(label_val)
        else:
            label_str = str(label_val)
            if label_str not in LABEL2ID:
                raise ValueError(f"Unknown label: {label_str}. Known: {LABELS}")
            y_id = LABEL2ID[label_str]

        return x, y_id


def load_dreamcatcher_hf_split(
    split: str,
    *,
    dataset_mode: str = "full",
    run_name: str = "",
    steps_csv: str = "results/run_steps.csv",
):
    """
    Convenience loader for HuggingFace DreamCatcher splits with the same
    token-aware builder + dataset_mode logic used by `DreamCatcherHFAudioDataset`.

    Returns a `datasets.Dataset` (rows still contain raw audio dicts).
    """
    cache_dir = os.environ.get("HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets"))
    logger = StepLogger(run_name=run_name, csv_path=steps_csv)
    builder = _get_builder(dataset_mode=dataset_mode, cache_dir=cache_dir, logger=logger)
    return builder.as_dataset(split=split)
