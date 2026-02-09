"""Fast cached dataset loader for DreamCatcher pre-computed spectrograms."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
from torch.utils.data import Dataset

from src.data.constants import DATASET_COMMIT


class CachedDataset(Dataset):
    """
    Loads pre-computed spectrograms from HDF5 cache.

    3 classes: quiet (0), breathe (1), snore (2)

    Args:
        split: train, validation, or test
        cache_dir: Directory containing {split}.h5 files
    """

    def __init__(self, split: str, cache_dir: str = "results/cache/spectrograms"):
        self.split = split
        self.cache_path = str(Path(cache_dir) / f"{split}.h5")

        if not Path(self.cache_path).exists():
            raise FileNotFoundError(
                f"Cache not found: {self.cache_path}\n"
                f"Run: python3 scripts/preprocess.py"
            )

        print(f"Loading cached spectrograms from: {self.cache_path}", file=sys.stderr)

        # Open once to validate and read metadata, then close.
        # Workers will re-open lazily in __getitem__ for multiprocessing safety.
        with h5py.File(self.cache_path, "r") as f:
            self._validate_cache_metadata(f)
            self.n_samples = int(f.attrs["n_samples"])
            self.n_mels = int(f.attrs["n_mels"])
            self.max_time = int(f.attrs["max_time"])

        # Lazy handle — opened per-worker on first __getitem__ call
        self._h5file = None

        print(f"  Loaded {self.n_samples} samples ({self.n_mels} mels, {self.max_time} time)", file=sys.stderr)

    def _validate_cache_metadata(self, h5file=None):
        """Validate that cache was generated with expected dataset version and config."""
        attrs = h5file.attrs if h5file is not None else self._h5file.attrs

        # Expected config (must match preprocess.py)
        expected_config = {
            "dataset_commit": DATASET_COMMIT,
            "sample_rate": 16000,
            "n_mels": 64,  # Consistent with all existing experiments for comparability
            "clip_seconds": 5.0,
            "invalid_audio_policy": "skip",
        }

        # Check for metadata presence (old caches may not have these)
        if "dataset_commit" not in attrs:
            raise ValueError(
                "Cache missing metadata (old cache format).\n"
                "Please regenerate cache: python3 scripts/preprocess.py"
            )

        # Validate each field
        mismatches = []
        for key, expected in expected_config.items():
            cached = attrs.get(key)
            # Handle string encoding from HDF5
            if isinstance(cached, bytes):
                cached = cached.decode("utf-8")
            if cached != expected:
                mismatches.append(f"  - {key}: cached={cached}, expected={expected}")

        if mismatches:
            raise ValueError(
                "Cache config mismatch! Cache was generated with different settings.\n"
                + "\n".join(mismatches)
                + "\n\nPlease regenerate cache: python3 scripts/preprocess.py"
            )

    def __len__(self):
        return self.n_samples

    def _ensure_h5(self):
        """Lazily open HDF5 file (safe for multiprocessing workers)."""
        if self._h5file is None:
            self._h5file = h5py.File(self.cache_path, "r")

    def __getitem__(self, idx):
        """
        Returns:
            spec: [n_mels, time] numpy array
            label: int (0=quiet, 1=breathe, 2=snore)
        """
        self._ensure_h5()
        spec = self._h5file["spectrograms"][idx]  # [n_mels, time]
        label = int(self._h5file["labels"][idx])
        return spec, label

    def __del__(self):
        """Close HDF5 file when dataset is destroyed."""
        if getattr(self, "_h5file", None) is not None:
            self._h5file.close()
