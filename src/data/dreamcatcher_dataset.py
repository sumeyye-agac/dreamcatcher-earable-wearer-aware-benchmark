"""
DreamCatcher sleep event classification dataset (3-class: quiet, breathe, snore).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .dreamcatcher_hf import DreamCatcherHFAudioConfig, DreamCatcherHFAudioDataset

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

        # Try to load from cache
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    cached = json.load(f)
                print(f"  Loaded cached indices from {cache_path}", file=sys.stderr)
                return cached["indices"]
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  Cache invalid ({e}), regenerating...", file=sys.stderr)

        # Scan dataset to find 3-class samples
        print(
            "  Scanning dataset for 3-class samples (this may take a moment)...", file=sys.stderr
        )
        indices = []

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

        # Save to cache
        try:
            with open(cache_path, "w") as f:
                json.dump(
                    {
                        "split": self.split,
                        "dataset_mode": self.dataset_mode,
                        "indices": indices,
                        "total_samples": len(self.full_ds),
                        "filtered_samples": len(indices),
                    },
                    f,
                    indent=2,
                )
            print(f"  Cached indices to {cache_path}", file=sys.stderr)
        except Exception as e:
            print(f"  Warning: Failed to cache indices: {e}", file=sys.stderr)

        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        """Get a (spectrogram, label) tuple with remapped 3-class label."""
        actual_idx = self.indices[idx]
        spec, orig_label = self.full_ds[actual_idx]

        # Remap to 3-class space
        new_label = LABEL_MAP[orig_label]
        return spec, new_label
