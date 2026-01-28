"""
Pre-compute and cache spectrograms for DreamCatcher dataset (quiet, breathe, snore).
"""

import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from src.data.dreamcatcher_hf import DreamCatcherHFAudioConfig
from src.data.dreamcatcher_dataset import DreamCatcherDataset


def preprocess_split(split: str, output_dir: Path):
    """Pre-compute spectrograms for one split and save to HDF5."""
    print(f"\n{'=' * 60}")
    print(f"Pre-processing {split} split (3-class)")
    print(f"{'=' * 60}\n")

    # Load dataset
    cfg = DreamCatcherHFAudioConfig(
        sample_rate=16000,
        n_mels=128,
        clip_seconds=5.0,
        invalid_audio_policy="pad",
    )
    dataset = DreamCatcherDataset(
        split=split,
        cfg=cfg,
        dataset_mode="full",
        run_name="preprocess",
        steps_csv="results/run_steps.csv",
        max_samples=0,
    )

    print(f"Dataset size: {len(dataset)} samples")
    print(f"Extracting spectrograms...")

    # Pre-allocate arrays
    n_samples = len(dataset)
    max_time = 0

    # First pass: determine max time dimension
    print("First pass: determining max time dimension...")
    for i in tqdm(range(min(1000, n_samples)), desc="Sampling"):
        spec, _ = dataset[i]
        max_time = max(max_time, spec.shape[1])

    print(f"Max time dimension (sampled): {max_time}")
    print(f"Using max_time = {max_time} for all spectrograms")

    # Create HDF5 file
    output_file = output_dir / f"{split}.h5"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_file, "w") as f:
        # Create datasets
        spectrograms = f.create_dataset(
            "spectrograms",
            shape=(n_samples, 128, max_time),
            dtype=np.float32,
            chunks=(1, 128, max_time),
            compression="gzip",
            compression_opts=4,
        )
        labels = f.create_dataset(
            "labels",
            shape=(n_samples,),
            dtype=np.int32,
        )

        # Process and save
        print(f"\nSecond pass: extracting and saving spectrograms...")
        for i in tqdm(range(n_samples), desc=f"Processing {split}"):
            spec, label = dataset[i]

            # Pad or truncate to max_time
            if spec.shape[1] < max_time:
                padded = np.zeros((128, max_time), dtype=np.float32)
                padded[:, : spec.shape[1]] = spec
                spec = padded
            elif spec.shape[1] > max_time:
                spec = spec[:, :max_time]

            spectrograms[i] = spec
            labels[i] = label

        # Save metadata
        f.attrs["n_samples"] = n_samples
        f.attrs["n_mels"] = 128
        f.attrs["max_time"] = max_time
        f.attrs["sample_rate"] = 16000
        f.attrs["split"] = split
        f.attrs["n_classes"] = 3
        f.attrs["classes"] = "quiet,breathe,snore"

    print(f"\n✓ Saved to: {output_file}")
    print(f"  Size: {output_file.stat().st_size / (1024**3):.2f} GB")


def main():
    output_dir = Path("results/cache/spectrograms")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("Pre-computing DreamCatcher Spectrograms (quiet, breathe, snore)")
    print("=" * 60)

    for split in ["train", "validation", "test"]:
        preprocess_split(split, output_dir)

    print("\n" + "=" * 60)
    print("✓ All splits pre-processed successfully!")
    print("=" * 60)
    print(f"\nCache location: {output_dir}")


if __name__ == "__main__":
    main()
