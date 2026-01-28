"""Fast cached dataset loader for DreamCatcher pre-computed spectrograms."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset


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
        cache_path = Path(cache_dir) / f"{split}.h5"

        if not cache_path.exists():
            raise FileNotFoundError(
                f"Cache not found: {cache_path}\n"
                f"Run: python3 scripts/preprocess.py"
            )

        print(f"Loading cached spectrograms from: {cache_path}", file=sys.stderr)

        # Load HDF5 file (keep file handle open for fast access)
        self.h5file = h5py.File(cache_path, "r")
        self.spectrograms = self.h5file["spectrograms"]
        self.labels = self.h5file["labels"]

        # Get metadata
        self.n_samples = self.h5file.attrs["n_samples"]
        self.n_mels = self.h5file.attrs["n_mels"]
        self.max_time = self.h5file.attrs["max_time"]

        print(f"  Loaded {self.n_samples} samples ({self.n_mels} mels, {self.max_time} time)", file=sys.stderr)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Returns:
            spec: [n_mels, time] numpy array
            label: int (0=quiet, 1=breathe, 2=snore)
        """
        spec = self.spectrograms[idx]  # [n_mels, time]
        label = int(self.labels[idx])
        return spec, label

    def __del__(self):
        """Close HDF5 file when dataset is destroyed."""
        if hasattr(self, "h5file"):
            self.h5file.close()
