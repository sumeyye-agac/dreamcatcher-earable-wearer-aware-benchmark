"""
Filtered subset datasets for DreamCatcher.
Provides respiratory events subset (breathe, cough, snore).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

from .dreamcatcher_hf import DreamCatcherHFAudioDataset, DreamCatcherHFAudioConfig

# Respiratory events subset configuration (3-class)
RESPIRATORY_LABELS = ["breathe", "cough", "snore"]
RESPIRATORY_ORIGINAL_INDICES = [5, 6, 7]  # Indices in original 9-class LABELS
RESPIRATORY_LABEL_MAP = {5: 0, 6: 1, 7: 2}  # 9-class -> 3-class remapping

# Balanced 4-class subset configuration
BALANCED4_LABELS = ["quiet", "breathe", "non_wearer", "snore"]
BALANCED4_ORIGINAL_INDICES = [0, 5, 1, 7]  # quiet=0, breathe=5, non_wearer=1, snore=7
BALANCED4_LABEL_MAP = {0: 0, 5: 1, 1: 2, 7: 3}  # 9-class -> 4-class remapping


class DreamCatcherRespiratorySubset:
    """
    DreamCatcher dataset filtered to respiratory events only (breathe, cough, snore).

    Filters samples to only include respiratory classes and remaps labels to 3-class space:
    - breathe (original 5) -> 0
    - cough (original 6) -> 1
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
        self.dataset_mode = dataset_mode
        self.run_name = run_name

        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Loading Respiratory Subset (breathe, cough, snore)", file=sys.stderr)
        print(f"  Split: {split}", file=sys.stderr)
        print(f"  Dataset Mode: {dataset_mode}", file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)

        # Load full dataset
        self.full_ds = DreamCatcherHFAudioDataset(
            split=split,
            cfg=cfg,
            dataset_mode=dataset_mode,
            run_name=run_name,
            steps_csv=steps_csv,
            cache_dir=cache_dir,
            max_samples=0,  # Don't limit here, we'll filter first
        )

        # Filter to respiratory events
        print("Filtering dataset to respiratory events...", file=sys.stderr)
        self.indices = self._get_or_create_filtered_indices()

        # Apply max_samples limit after filtering
        if max_samples and max_samples > 0:
            n = min(int(max_samples), len(self.indices))
            self.indices = self.indices[:n]

        print(f"Respiratory subset loaded: {len(self.indices)} samples", file=sys.stderr)
        print(f"  (Original dataset: {len(self.full_ds)} samples)\n", file=sys.stderr)

    def _get_cache_path(self) -> Path:
        """Get path to cached index file."""
        cache_root = Path("results/cache")
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_file = f"respiratory_indices_{self.dataset_mode}_{self.split}.json"
        return cache_root / cache_file

    def _get_or_create_filtered_indices(self) -> list[int]:
        """Get filtered indices from cache or create by scanning dataset."""
        cache_path = self._get_cache_path()

        # Try to load from cache
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    cached = json.load(f)
                print(f"  Loaded cached indices from {cache_path}", file=sys.stderr)
                return cached["indices"]
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  Cache invalid ({e}), regenerating...", file=sys.stderr)

        # Scan dataset to find respiratory samples
        print("  Scanning dataset for respiratory events (this may take a moment)...", file=sys.stderr)
        indices = []

        for idx in range(len(self.full_ds)):
            try:
                _, label = self.full_ds[idx]
                if label in RESPIRATORY_ORIGINAL_INDICES:
                    indices.append(idx)
            except Exception as e:
                # Skip invalid samples
                print(f"  Warning: Skipped sample {idx} due to error: {e}", file=sys.stderr)
                continue

            # Progress indicator
            if (idx + 1) % 10000 == 0:
                print(f"  Scanned {idx + 1}/{len(self.full_ds)} samples, found {len(indices)} respiratory events...", file=sys.stderr)

        print(f"  Scan complete: {len(indices)} respiratory samples found", file=sys.stderr)

        # Save to cache
        try:
            with open(cache_path, "w") as f:
                json.dump({
                    "split": self.split,
                    "dataset_mode": self.dataset_mode,
                    "indices": indices,
                    "total_samples": len(self.full_ds),
                    "respiratory_samples": len(indices),
                }, f, indent=2)
            print(f"  Cached indices to {cache_path}", file=sys.stderr)
        except Exception as e:
            print(f"  Warning: Failed to cache indices: {e}", file=sys.stderr)

        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """Get sample and remap label from 9-class to 3-class space."""
        # Get sample from full dataset at filtered index
        full_idx = self.indices[idx]
        x, y_full = self.full_ds[full_idx]

        # Remap label: 5->0, 6->1, 7->2
        if y_full not in RESPIRATORY_LABEL_MAP:
            raise ValueError(
                f"Invalid label {y_full} at filtered index {idx} (full index {full_idx}). "
                f"Expected one of {RESPIRATORY_ORIGINAL_INDICES}"
            )

        y_subset = RESPIRATORY_LABEL_MAP[y_full]
        return x, y_subset


def load_respiratory_hf_split(
    split: str,
    *,
    dataset_mode: str = "full",
    run_name: str = "",
    steps_csv: str = "results/run_steps.csv",
    cache_dir: str | None = None,
):
    """
    Load HuggingFace DreamCatcher split filtered to respiratory events.

    Returns raw HF dataset (for KD with teacher) with only respiratory samples.
    Labels are kept in original 9-class space (5, 6, 7) for teacher compatibility.

    Use this for KD training where teacher expects 9-class logits.
    """
    from .dreamcatcher_hf import load_dreamcatcher_hf_split
    import os

    # Load full dataset
    full_ds = load_dreamcatcher_hf_split(
        split=split,
        dataset_mode=dataset_mode,
        run_name=run_name,
        steps_csv=steps_csv,
        cache_dir=cache_dir,
    )

    # Filter to respiratory events
    print(f"Filtering {split} split to respiratory events...", file=sys.stderr)

    def is_respiratory(example):
        label = example.get("label") or example.get("event_label") or example.get("class")
        return label in RESPIRATORY_ORIGINAL_INDICES

    filtered_ds = full_ds.filter(is_respiratory)
    print(f"Filtered {split}: {len(filtered_ds)} samples (from {len(full_ds)})\n", file=sys.stderr)

    return filtered_ds


class DreamCatcherBalanced4Subset:
    """
    DreamCatcher dataset filtered to balanced 4 classes (quiet, breathe, non_wearer, snore).

    Filters samples to major classes and remaps labels to 4-class space:
    - quiet (original 0) -> 0
    - breathe (original 5) -> 1
    - non_wearer (original 1) -> 2
    - snore (original 7) -> 3

    Returns:
        x: log-mel spectrogram [n_mels, time]
        y: integer class id in [0, 1, 2, 3]
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
        self.dataset_mode = dataset_mode
        self.run_name = run_name

        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Loading Balanced 4-Class Subset", file=sys.stderr)
        print(f"  Classes: quiet, breathe, non_wearer, snore", file=sys.stderr)
        print(f"  Split: {split}", file=sys.stderr)
        print(f"  Dataset Mode: {dataset_mode}", file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)

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

        # Filter to balanced 4 classes
        print("Filtering dataset to balanced 4 classes...", file=sys.stderr)
        self.indices = self._get_or_create_filtered_indices()

        # Apply max_samples limit after filtering
        if max_samples and max_samples > 0:
            n = min(int(max_samples), len(self.indices))
            self.indices = self.indices[:n]

        print(f"Balanced 4-class subset loaded: {len(self.indices)} samples", file=sys.stderr)
        print(f"  (Original dataset: {len(self.full_ds)} samples)\n", file=sys.stderr)

    def _get_cache_path(self) -> Path:
        """Get path to cached index file."""
        cache_root = Path("results/cache")
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_file = f"balanced4_indices_{self.dataset_mode}_{self.split}.json"
        return cache_root / cache_file

    def _get_or_create_filtered_indices(self) -> list[int]:
        """Get filtered indices from cache or create by scanning dataset."""
        cache_path = self._get_cache_path()

        # Try to load from cache
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    cached = json.load(f)
                print(f"  Loaded cached indices from {cache_path}", file=sys.stderr)
                return cached["indices"]
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  Cache invalid ({e}), regenerating...", file=sys.stderr)

        # Scan dataset to find balanced 4-class samples
        print("  Scanning dataset for balanced 4 classes (this may take a moment)...", file=sys.stderr)
        indices = []

        for idx in range(len(self.full_ds)):
            try:
                _, label = self.full_ds[idx]
                if label in BALANCED4_ORIGINAL_INDICES:
                    indices.append(idx)
            except Exception as e:
                print(f"  Warning: Skipped sample {idx} due to error: {e}", file=sys.stderr)
                continue

            if (idx + 1) % 10000 == 0:
                print(f"  Scanned {idx + 1}/{len(self.full_ds)} samples, found {len(indices)} balanced samples...", file=sys.stderr)

        print(f"  Scan complete: {len(indices)} balanced 4-class samples found", file=sys.stderr)

        # Save to cache
        try:
            with open(cache_path, "w") as f:
                json.dump({
                    "split": self.split,
                    "dataset_mode": self.dataset_mode,
                    "indices": indices,
                    "total_samples": len(self.full_ds),
                    "balanced4_samples": len(indices),
                }, f, indent=2)
            print(f"  Cached indices to {cache_path}", file=sys.stderr)
        except Exception as e:
            print(f"  Warning: Failed to cache indices: {e}", file=sys.stderr)

        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """Get sample and remap label from 9-class to 4-class space."""
        full_idx = self.indices[idx]
        x, y_full = self.full_ds[full_idx]

        # Remap label: 0->0, 5->1, 1->2, 7->3
        if y_full not in BALANCED4_LABEL_MAP:
            raise ValueError(
                f"Invalid label {y_full} at filtered index {idx} (full index {full_idx}). "
                f"Expected one of {BALANCED4_ORIGINAL_INDICES}"
            )

        y_subset = BALANCED4_LABEL_MAP[y_full]
        return x, y_subset


def load_balanced4_hf_split(
    split: str,
    *,
    dataset_mode: str = "full",
    run_name: str = "",
    steps_csv: str = "results/run_steps.csv",
    cache_dir: str | None = None,
):
    """
    Load HuggingFace DreamCatcher split filtered to balanced 4 classes.

    Returns raw HF dataset (for KD with teacher) with only balanced 4-class samples.
    Labels are kept in original 9-class space (0, 5, 1, 7) for teacher compatibility.
    """
    from .dreamcatcher_hf import load_dreamcatcher_hf_split

    full_ds = load_dreamcatcher_hf_split(
        split=split,
        dataset_mode=dataset_mode,
        run_name=run_name,
        steps_csv=steps_csv,
        cache_dir=cache_dir,
    )

    print(f"Filtering {split} split to balanced 4 classes...", file=sys.stderr)

    def is_balanced4(example):
        label = example.get("label") or example.get("event_label") or example.get("class")
        return label in BALANCED4_ORIGINAL_INDICES

    filtered_ds = full_ds.filter(is_balanced4)
    print(f"Filtered {split}: {len(filtered_ds)} samples (from {len(full_ds)})\n", file=sys.stderr)

    return filtered_ds
