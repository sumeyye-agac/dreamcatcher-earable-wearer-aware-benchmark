"""
Cached Balanced4 dataset that loads pre-computed spectrograms from disk.
Much faster than computing spectrograms on-the-fly.
"""

from pathlib import Path

import h5py
import numpy as np


class CachedBalanced4Dataset:
    """
    Fast dataset that loads pre-computed spectrograms from HDF5 cache.

    10-20x faster than DreamCatcherBalanced4Subset because:
    - No audio loading from HuggingFace
    - No audio resampling
    - No spectrogram computation
    - Direct disk read of pre-computed features
    """

    def __init__(
        self,
        split: str,
        cache_dir: str = "results/cache/spectrograms",
        max_samples: int = 0,
    ):
        self.split = split
        cache_path = Path(cache_dir) / f"balanced4_{split}.h5"

        if not cache_path.exists():
            raise FileNotFoundError(
                f"Cache file not found: {cache_path}\n"
                f"Please run: python scripts/preprocess_balanced4_cache.py"
            )

        print(f"Loading cached spectrograms from: {cache_path}")
        self.h5_file = h5py.File(cache_path, "r")
        self.spectrograms = self.h5_file["spectrograms"]
        self.labels = self.h5_file["labels"]

        total_samples = self.h5_file.attrs["n_samples"]
        self.n_mels = self.h5_file.attrs["n_mels"]
        self.max_time = self.h5_file.attrs["max_time"]

        # Support max_samples for smoke test
        if max_samples > 0:
            self.n_samples = min(max_samples, total_samples)
            print(
                f"  Using {self.n_samples}/{total_samples} samples "
                f"({self.n_mels} mels, {self.max_time} time)"
            )
        else:
            self.n_samples = total_samples
            print(f"  Loaded {self.n_samples} samples ({self.n_mels} mels, {self.max_time} time)")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        """
        Returns:
            spectrogram: [n_mels, time] float32
            label: int (0-3 for 4-class)
        """
        spec = self.spectrograms[idx]  # [n_mels, time]
        label = int(self.labels[idx])
        return spec, label

    def __del__(self):
        """Close HDF5 file on cleanup."""
        if hasattr(self, "h5_file"):
            self.h5_file.close()
