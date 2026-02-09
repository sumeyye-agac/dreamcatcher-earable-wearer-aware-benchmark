from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np
from datasets import DownloadConfig, load_dataset_builder
from huggingface_hub import get_token

from src.utils.runlog import StepLogger

from .audio_features import compute_log_mel
from .constants import DATASET_COMMIT

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
    # Use pinned dataset commit from constants (single source of truth)
    base = f"https://huggingface.co/datasets/THU-PI-Sensing/DreamCatcher/resolve/{DATASET_COMMIT}/"
    dataset_mode = dataset_mode.lower().strip()
    if dataset_mode not in {"full", "smoke"}:
        raise ValueError("dataset_mode must be one of: full, smoke")

    # Important: "smoke" must not reuse the already-prepared "full" cache artifacts.
    # Give it a separate cache root so datasets doesn't silently reuse the full prepared dataset.
    effective_cache_dir = (
        cache_dir if dataset_mode == "full" else os.path.join(cache_dir, "_dreamcatcher_smoke")
    )

    key = (dataset_mode, effective_cache_dir)
    if key in _BUILDER_CACHE:
        if logger is not None:
            logger.log(
                "dataset_builder_cache_hit",
                detail=f"mode={dataset_mode} cache_dir={effective_cache_dir}",
            )
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

    # Allow training with cached dataset even without token
    # Token is only required for downloading, not for using cached data
    token_available = get_token()
    download_config = DownloadConfig(cache_dir=effective_cache_dir, token=token_available or True)
    if not token_available:
        # Check if dataset is already cached
        cache_path = os.path.join(effective_cache_dir, "THU-PI-Sensing___dream_catcher")
        if not os.path.exists(cache_path):
            raise RuntimeError(
                "HuggingFace token not found and dataset not cached locally.\n"
                "The DreamCatcher dataset is gated.\n"
                "1) Request access: https://huggingface.co/datasets/THU-PI-Sensing/DreamCatcher\n"
                "2) Login: `hf auth login` (or set env var `HUGGINGFACE_HUB_TOKEN` / `HF_TOKEN`)."
            )

    if logger is not None:
        logger.log(
            "dataset_builder_prepare_start",
            detail=f"mode={dataset_mode} cache_dir={effective_cache_dir}",
        )
    t0 = time.time()
    # Pin dataset to specific commit for full reproducibility (uses same commit as data_files)
    builder = load_dataset_builder(
        "THU-PI-Sensing/DreamCatcher",
        name="sleep_event_classification",
        revision=DATASET_COMMIT,
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

    hb_thread = threading.Thread(
        target=_heartbeat_loop, name="dreamcatcher_hf_heartbeat", daemon=True
    )
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
    n_mels: int = 128
    clip_seconds: float = 5.0
    # How to handle rare malformed audio buffers:
    # - "skip": never use invalid audio; search nearby indices; error if none found
    # - "resample": like skip, but fall back to silence if still invalid (not recommended for benchmarks)
    # - "zero": always replace invalid audio with silence (fast, but can bias results)
    invalid_audio_policy: str = "skip"


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
        cache_dir: str | None = None,
        max_samples: int = 0,
    ):
        self.cfg = cfg or DreamCatcherHFAudioConfig()
        self._logger = StepLogger(run_name=run_name, csv_path=steps_csv)

        print(f"\n{'=' * 60}", file=sys.stderr)
        print("Loading DreamCatcher Dataset", file=sys.stderr)
        print(f"  Split: {split}", file=sys.stderr)
        print("  Config: sleep_event_classification", file=sys.stderr)
        print("  Source: THU-PI-Sensing/DreamCatcher (HuggingFace)", file=sys.stderr)
        print(f"  Dataset Mode: {dataset_mode}", file=sys.stderr)
        if run_name:
            print(f"  Run Name: {run_name}", file=sys.stderr)
        print(f"{'=' * 60}\n", file=sys.stderr)

        cache_dir = cache_dir or os.environ.get(
            "HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets")
        )
        du = shutil.disk_usage(cache_dir)
        self._logger.log(
            "dataset_init",
            detail=f"split={split} mode={dataset_mode} cache_dir={cache_dir} free_gb={du.free / 1e9:.1f}",
        )

        print(
            "Preparing dataset (download + generate). This can take a while on first run...",
            file=sys.stderr,
        )
        t_prepare = time.time()
        builder = _get_builder(dataset_mode=dataset_mode, cache_dir=cache_dir, logger=self._logger)
        self._logger.log("dataset_builder_ready", t0=t_prepare)

        t_as = time.time()
        self.ds = builder.as_dataset(split=split)
        if max_samples and max_samples > 0:
            n = min(int(max_samples), len(self.ds))
            self.ds = self.ds.select(range(n))
        self._logger.log(
            "dataset_split_loaded",
            t0=t_as,
            detail=f"split={split} len={len(self.ds)} max_samples={max_samples if max_samples else ''}",
        )
        print(f"Dataset loaded successfully: {len(self.ds)} samples\n", file=sys.stderr)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        def _load_row_audio(i: int):
            row0 = self.ds[i]
            if "audio" in row0:
                audio0 = row0["audio"]
            elif "audio_data" in row0:
                audio0 = row0["audio_data"]
            else:
                raise KeyError("Expected 'audio' (or legacy 'audio_data') in dataset row.")
            y0 = np.asarray(audio0["array"], dtype=np.float32)
            sr0 = int(audio0["sampling_rate"])
            return y0, sr0, row0

        y, sr, row = _load_row_audio(idx)

        # Robustify against rare malformed / empty audio buffers
        if y.ndim == 2:
            # Audio is stereo with shape (channels, samples) e.g. (2, 15629)
            # Average across channels (axis=0) to get mono (samples,)
            if y.shape[0] == 0 or y.shape[1] == 0:
                y = np.zeros((0,), dtype=np.float32)
            else:
                y = y.mean(axis=0)

        def _is_invalid_audio(y1: np.ndarray) -> bool:
            return (y1.size == 0) or (not np.isfinite(y1).all())

        if _is_invalid_audio(y):
            policy = str(getattr(self.cfg, "invalid_audio_policy", "resample")).lower().strip()
            if policy not in {"skip", "resample", "zero"}:
                policy = "skip"

            if policy in {"skip", "resample"}:
                max_tries = min(1024, max(1, len(self.ds) - 1))
                for k in range(1, max_tries + 1):
                    j = (idx + k) % len(self.ds)
                    y2, sr2, row2 = _load_row_audio(j)
                    if y2.ndim == 2:
                        if y2.shape[0] == 0 or y2.shape[1] == 0:
                            y2 = np.zeros((0,), dtype=np.float32)
                        else:
                            y2 = y2.mean(axis=0)
                    if not _is_invalid_audio(y2):
                        self._logger.log("invalid_audio_resampled", detail=f"idx={idx} -> {j}")
                        y, sr, row = y2, sr2, row2
                        break

            if _is_invalid_audio(y):
                if policy == "zero":
                    self._logger.log("invalid_audio_zero_fallback", detail=f"idx={idx}")
                    y = np.zeros((0,), dtype=np.float32)
                elif policy == "resample":
                    # Last resort: keep training running, but avoid crashes.
                    self._logger.log("invalid_audio_zero_fallback", detail=f"idx={idx}")
                    y = np.zeros((0,), dtype=np.float32)
                else:
                    # "skip" policy: do not silently fabricate an example.
                    raise RuntimeError(
                        f"Invalid audio encountered at idx={idx} and could not find a valid replacement "
                        f"within {max_tries} tries. Consider policy='resample' or 'zero' if you just want to run."
                    )

        # Replace NaN/inf with zeros so librosa doesn't hard-error
        if y.size > 0:
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        if sr != self.cfg.sample_rate:
            import librosa

            y = librosa.resample(y, orig_sr=sr, target_sr=self.cfg.sample_rate)

        # Standardize waveform length to a fixed clip (critical for stable training)
        # DreamCatcher paper uses 5s clips for event classification benchmarks.
        clip_s = float(getattr(self.cfg, "clip_seconds", 5.0))
        target_len = int(self.cfg.sample_rate * clip_s)

        # Avoid librosa warnings / degenerate spectrograms for extremely short clips
        if y.shape[0] < 1024:
            y = np.pad(y, (0, 1024 - y.shape[0]), mode="constant")

        # Enforce fixed length (pad/crop)
        if y.shape[0] < target_len:
            y = np.pad(y, (0, target_len - y.shape[0]), mode="constant")
        elif y.shape[0] > target_len:
            # For now, head-crop. (We can switch to center-crop once pipeline is stable.)
            y = y[:target_len]

        x = compute_log_mel(y=y, sr=self.cfg.sample_rate, n_mels=self.cfg.n_mels)

        if "label" in row:
            label_val = row["label"]
        elif "event_label" in row:
            label_val = row["event_label"]
        elif "class" in row:
            label_val = row["class"]
        else:
            raise KeyError("Expected 'label' (or 'event_label'/'class') in dataset row.")

        if isinstance(label_val, int | np.integer):
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
    cache_dir: str | None = None,
):
    """
    Convenience loader for HuggingFace DreamCatcher splits with the same
    token-aware builder + dataset_mode logic used by `DreamCatcherHFAudioDataset`.

    Returns a `datasets.Dataset` (rows still contain raw audio dicts).
    """
    cache_dir = cache_dir or os.environ.get(
        "HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets")
    )
    logger = StepLogger(run_name=run_name, csv_path=steps_csv)
    builder = _get_builder(dataset_mode=dataset_mode, cache_dir=cache_dir, logger=logger)
    return builder.as_dataset(split=split)
