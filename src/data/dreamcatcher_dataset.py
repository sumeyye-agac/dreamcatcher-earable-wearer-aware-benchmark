"""
DreamCatcher sleep event classification dataset (3-class: quiet, breathe, snore).
"""

from __future__ import annotations

import json
import sys
from numbers import Integral
from pathlib import Path

from .constants import DATASET_COMMIT
from .dreamcatcher_hf import LABEL2ID, DreamCatcherHFAudioConfig, DreamCatcherHFAudioDataset

# 3-class configuration
CLASS_LABELS = ["quiet", "breathe", "snore"]
ORIGINAL_INDICES = [0, 5, 7]  # quiet=0, breathe=5, snore=7 from 9-class original dataset
LABEL_MAP = {0: 0, 5: 1, 7: 2}  # 9-class -> 3-class remapping


class DreamCatcherDataset:
    """
    DreamCatcher sleep event classification dataset (3-class: quiet, breathe, snore).

    Filters the original 9-class DreamCatcher dataset to 3 sleep event classes.

    Label mapping:
    - quiet (original 0) -> 0
    - breathe (original 5) -> 1
    - snore (original 7) -> 2

    Returns:
        x: log-mel spectrogram [n_mels, time]
        y: integer class id in [0, 1, 2]
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
        self.split = split
        self.cfg = cfg or DreamCatcherHFAudioConfig()
        self.dataset_mode = dataset_mode
        self.run_name = run_name

        print(f"\n{'=' * 60}", file=sys.stderr)
        print("Loading DreamCatcher Dataset (3-class)", file=sys.stderr)
        print("  Classes: quiet, breathe, snore", file=sys.stderr)
        print(f"  Split: {split}", file=sys.stderr)
        print(f"  Dataset Mode: {dataset_mode}", file=sys.stderr)
        print(f"{'=' * 60}\n", file=sys.stderr)

        # Load full dataset
        self.full_ds = DreamCatcherHFAudioDataset(
            split=split,
            cfg=cfg,
            dataset_mode=dataset_mode,
            run_name=run_name,
            steps_csv=steps_csv,
            cache_dir=cache_dir,
            max_samples=0,
        )

        # Filter to 3 classes
        print("Filtering dataset to 3-class subset...", file=sys.stderr)
        self.indices = self._get_or_create_filtered_indices()

        # Apply max_samples limit after filtering
        if max_samples and max_samples > 0:
            n = min(int(max_samples), len(self.indices))
            self.indices = self.indices[:n]

        print(f"Dataset loaded: {len(self.indices)} samples", file=sys.stderr)
        print(f"  (Original dataset: {len(self.full_ds)} samples)\n", file=sys.stderr)

    def _get_cache_path(self) -> Path:
        """Get path to cached index file."""
        cache_root = Path("results/cache")
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_file = f"indices_{self.dataset_mode}_{self.split}.json"
        return cache_root / cache_file

    def _get_or_create_filtered_indices(self) -> list[int]:
        """Get filtered indices from cache or create by scanning dataset."""
        cache_path = self._get_cache_path()

        # Try to load from cache with config validation
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    cached = json.load(f)

                # Validate dataset commit matches (critical for reproducibility)
                cached_commit = cached.get("dataset_commit") or cached.get("dataset_version")
                if cached_commit != DATASET_COMMIT:
                    print(
                        f"  Dataset version mismatch (cached: {cached_commit}, current: {DATASET_COMMIT}), "
                        f"regenerating...",
                        file=sys.stderr
                    )
                    # Dataset version changed, must regenerate
                    pass  # Fall through to regeneration
                # Validate cache config matches current config
                elif "config" in cached:
                    cached_cfg = cached["config"]
                    current_cfg = {
                        "sample_rate": self.cfg.sample_rate,
                        "n_mels": self.cfg.n_mels,
                        "clip_seconds": self.cfg.clip_seconds,
                        "invalid_audio_policy": self.cfg.invalid_audio_policy,
                    }
                    if cached_cfg != current_cfg:
                        print(
                            f"  Cache config mismatch (cached: {cached_cfg}, current: {current_cfg}), "
                            f"regenerating...",
                            file=sys.stderr
                        )
                        # Config changed, regenerate cache
                        pass  # Fall through to regeneration
                    else:
                        print(
                            f"  Loaded cached indices from {cache_path} "
                            f"(dataset={DATASET_COMMIT[:8]}, config validated)",
                            file=sys.stderr
                        )
                        return cached["indices"]
                else:
                    # Old cache format without config, regenerate for safety
                    print("  Cache missing config metadata, regenerating...", file=sys.stderr)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  Cache invalid ({e}), regenerating...", file=sys.stderr)

        # Scan dataset to find 3-class samples
        print(
            "  Scanning dataset for 3-class samples (this may take a moment)...", file=sys.stderr
        )
        indices = []
        raw_ds = getattr(self.full_ds, "ds", None)
        label_col = None
        if raw_ds is not None:
            for candidate in ("label", "event_label", "class"):
                if candidate in getattr(raw_ds, "column_names", []):
                    label_col = candidate
                    break

        if raw_ds is not None and label_col is not None:
            print(
                f"  Using fast label-column scan: '{label_col}' (no audio decode)",
                file=sys.stderr,
            )
            labels = raw_ds[label_col]
            total = len(labels)
            for idx, label_val in enumerate(labels):
                label = self._normalize_label_id(label_val)
                if label in ORIGINAL_INDICES:
                    indices.append(idx)
                if (idx + 1) % 50000 == 0:
                    print(
                        f"  Scanned {idx + 1}/{total} labels, found {len(indices)} samples...",
                        file=sys.stderr,
                    )
        else:
            print("  Falling back to slow sample scan (audio decode path).", file=sys.stderr)
            for idx in range(len(self.full_ds)):
                try:
                    _, label = self.full_ds[idx]
                    if label in ORIGINAL_INDICES:
                        indices.append(idx)
                except Exception as e:
                    print(f"  Warning: Skipped sample {idx} due to error: {e}", file=sys.stderr)
                    continue

                if (idx + 1) % 10000 == 0:
                    print(
                        f"  Scanned {idx + 1}/{len(self.full_ds)} samples, found {len(indices)} samples...",
                        file=sys.stderr,
                    )

        print(f"  Scan complete: {len(indices)} 3-class samples found", file=sys.stderr)

        # Save to cache with full config for version tracking
        try:
            cache_metadata = {
                "split": self.split,
                "dataset_mode": self.dataset_mode,
                "indices": indices,
                "total_samples": len(self.full_ds),
                "filtered_samples": len(indices),
                # Config metadata for cache validation
                "config": {
                    "sample_rate": self.cfg.sample_rate,
                    "n_mels": self.cfg.n_mels,
                    "clip_seconds": self.cfg.clip_seconds,
                    "invalid_audio_policy": self.cfg.invalid_audio_policy,
                },
                # Dataset commit hash for reproducibility
                "dataset_commit": DATASET_COMMIT,
            }
            with open(cache_path, "w") as f:
                json.dump(
                    cache_metadata,
                    f,
                    indent=2,
                )
            print(f"  Cached indices to {cache_path}", file=sys.stderr)
        except Exception as e:
            print(f"  Warning: Failed to cache indices: {e}", file=sys.stderr)

        return indices

    @staticmethod
    def _normalize_label_id(label_val: object) -> int:
        """Normalize dataset row label to integer ID (0..8 original taxonomy)."""
        if isinstance(label_val, Integral):
            return int(label_val)
        label_str = str(label_val)
        if label_str not in LABEL2ID:
            raise ValueError(f"Unknown label value in DreamCatcher dataset: {label_str}")
        return int(LABEL2ID[label_str])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        """Get a (spectrogram, label) tuple with remapped 3-class label."""
        actual_idx = self.indices[idx]
        spec, orig_label = self.full_ds[actual_idx]

        # Remap to 3-class space
        new_label = LABEL_MAP[orig_label]
        return spec, new_label
