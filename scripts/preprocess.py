"""
Pre-compute and cache spectrograms for DreamCatcher dataset (quiet, breathe, snore).
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from src.data.constants import DATASET_COMMIT
from src.data.dreamcatcher_dataset import DreamCatcherDataset
from src.data.dreamcatcher_hf import DreamCatcherHFAudioConfig


def is_cache_file_valid(path: Path, split: str) -> bool:
    """Lightweight cache integrity check."""
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as f:
            if "spectrograms" not in f or "labels" not in f:
                return False
            specs = f["spectrograms"]
            labels = f["labels"]
            if specs.ndim != 3 or labels.ndim != 1:
                return False
            if specs.shape[0] <= 0 or labels.shape[0] <= 0:
                return False
            if specs.shape[0] != labels.shape[0]:
                return False
            n_samples_attr = int(f.attrs.get("n_samples", -1))
            n_mels_attr = int(f.attrs.get("n_mels", -1))
            sr_attr = int(f.attrs.get("sample_rate", -1))
            split_attr = str(f.attrs.get("split", ""))
            max_samples_attr = int(f.attrs.get("max_samples", 0))
            if n_samples_attr != specs.shape[0]:
                return False
            if n_mels_attr != 64:
                return False
            if sr_attr != 16000:
                return False
            if split_attr != split:
                return False
            if max_samples_attr != 0:
                return False
        return True
    except Exception:
        return False


def _normalize_compression(compression: str) -> str | None:
    c = compression.lower().strip()
    if c in {"none", "off", "false", "0"}:
        return None
    if c in {"lzf", "gzip"}:
        return c
    raise ValueError("compression must be one of: none, lzf, gzip")


def preprocess_split(
    split: str,
    output_dir: Path,
    *,
    batch_size: int,
    compression: str | None,
    compression_level: int,
    max_samples: int,
    resume_partial: bool,
):
    """Pre-compute spectrograms for one split and save to HDF5."""
    print(f"\n{'=' * 60}")
    print(f"Pre-processing {split} split (3-class)")
    print(f"{'=' * 60}\n")

    # Load dataset
    cfg = DreamCatcherHFAudioConfig(
        sample_rate=16000,
        n_mels=64,  # Consistent with all existing experiments for comparability
        clip_seconds=5.0,
        invalid_audio_policy="skip",
    )
    dataset = DreamCatcherDataset(
        split=split,
        cfg=cfg,
        dataset_mode="full",
        run_name="preprocess",
        steps_csv="results/run_steps.csv",
        max_samples=max_samples,
    )

    print(f"Dataset size: {len(dataset)} samples")
    print("Extracting spectrograms...")

    n_samples = len(dataset)
    if n_samples <= 0:
        raise RuntimeError(f"Split '{split}' produced empty dataset.")

    # Clip length is fixed; one sample is enough to lock the time axis.
    first_spec, _ = dataset[0]
    max_time = int(first_spec.shape[1])
    print(f"Using max_time = {max_time} (from first sample)")
    print(f"Write config: batch_size={batch_size}, compression={compression or 'none'}")

    output_file = output_dir / f"{split}.h5"
    tmp_file = output_dir / f".{split}.h5.tmp"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    resume_cursor = 0

    chunk_rows = max(1, min(batch_size, n_samples))
    h5_kwargs = {
        "shape": (n_samples, cfg.n_mels, max_time),
        "dtype": np.float32,
        "chunks": (chunk_rows, cfg.n_mels, max_time),
    }
    if compression is not None:
        h5_kwargs["compression"] = compression
        if compression == "gzip":
            h5_kwargs["compression_opts"] = int(compression_level)

    if tmp_file.exists() and resume_partial:
        try:
            with h5py.File(tmp_file, "r", rdcc_nbytes=64 * 1024 * 1024) as fchk:
                ok = (
                    "spectrograms" in fchk
                    and "labels" in fchk
                    and int(fchk.attrs.get("n_samples", -1)) == n_samples
                    and int(fchk.attrs.get("n_mels", -1)) == cfg.n_mels
                    and int(fchk.attrs.get("max_time", -1)) == max_time
                    and str(fchk.attrs.get("split", "")) == split
                    and int(fchk.attrs.get("max_samples", 0)) == int(max_samples)
                )
                if ok:
                    resume_cursor = int(fchk.attrs.get("next_index", 0))
                    if resume_cursor < 0 or resume_cursor > n_samples:
                        ok = False
            if ok and resume_cursor > 0:
                print(
                    f"Resuming partial cache: {tmp_file} from index {resume_cursor}/{n_samples}",
                )
            elif not ok:
                tmp_file.unlink()
        except Exception:
            tmp_file.unlink(missing_ok=True)
    elif tmp_file.exists() and not resume_partial:
        tmp_file.unlink()

    file_mode = "r+" if tmp_file.exists() else "w"
    with h5py.File(tmp_file, file_mode, rdcc_nbytes=64 * 1024 * 1024) as f:
        if file_mode == "w":
            spectrograms = f.create_dataset(
                "spectrograms",
                **h5_kwargs,
            )
            labels = f.create_dataset(
                "labels",
                shape=(n_samples,),
                dtype=np.int32,
            )
        else:
            spectrograms = f["spectrograms"]
            labels = f["labels"]

        print("\nBatch pass: extracting and saving spectrograms...")
        cursor = int(resume_cursor)
        block_specs = np.zeros((batch_size, cfg.n_mels, max_time), dtype=np.float32)
        block_labels = np.zeros((batch_size,), dtype=np.int32)
        block_count = 0

        for i in tqdm(
            range(resume_cursor, n_samples),
            total=n_samples,
            initial=resume_cursor,
            desc=f"Processing {split}",
        ):
            spec, label = dataset[i]
            t = int(spec.shape[1])
            if t >= max_time:
                block_specs[block_count] = spec[:, :max_time]
            else:
                block_specs[block_count].fill(0.0)
                block_specs[block_count, :, :t] = spec
            block_labels[block_count] = int(label)
            block_count += 1

            if block_count == batch_size or i == (n_samples - 1):
                end = cursor + block_count
                spectrograms[cursor:end] = block_specs[:block_count]
                labels[cursor:end] = block_labels[:block_count]
                cursor = end
                block_count = 0
                f.attrs["next_index"] = cursor
                if cursor % (batch_size * 20) == 0 or cursor == n_samples:
                    f.flush()

        if cursor != n_samples:
            raise RuntimeError(
                f"Split '{split}' write count mismatch: wrote {cursor}, expected {n_samples}"
            )

        f.attrs["n_samples"] = n_samples
        f.attrs["n_mels"] = cfg.n_mels
        f.attrs["max_time"] = max_time
        f.attrs["sample_rate"] = cfg.sample_rate
        f.attrs["clip_seconds"] = cfg.clip_seconds
        f.attrs["invalid_audio_policy"] = cfg.invalid_audio_policy
        f.attrs["dataset_commit"] = DATASET_COMMIT
        f.attrs["split"] = split
        f.attrs["n_classes"] = 3
        f.attrs["classes"] = "quiet,breathe,snore"
        f.attrs["max_samples"] = int(max_samples)
        f.attrs["next_index"] = n_samples

    tmp_file.replace(output_file)

    print(f"\n✓ Saved to: {output_file}")
    print(f"  Size: {output_file.stat().st_size / (1024**3):.2f} GB")


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute DreamCatcher spectrogram cache.")
    parser.add_argument(
        "--splits",
        type=str,
        default="train,validation,test",
        help="Comma-separated split names to preprocess.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a split if corresponding HDF5 cache already exists.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Precompute write chunk size.",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="lzf",
        help="HDF5 compression: none, lzf, gzip.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=1,
        help="gzip level when --compression=gzip.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap for quick checks (0 means full split).",
    )
    parser.add_argument(
        "--no-resume-partial",
        action="store_true",
        help="Disable resuming from existing .tmp partial cache files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path("results/cache/spectrograms")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("Pre-computing DreamCatcher Spectrograms (quiet, breathe, snore)")
    print("=" * 60)

    raw_splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    if not raw_splits:
        raise ValueError("--splits resolved to empty list.")

    valid = {"train", "validation", "test"}
    invalid = [s for s in raw_splits if s not in valid]
    if invalid:
        raise ValueError(f"Invalid split(s): {invalid}. Valid splits: {sorted(valid)}")

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.max_samples < 0:
        raise ValueError("--max-samples must be >= 0")
    compression = _normalize_compression(str(args.compression))

    for split in raw_splits:
        output_file = output_dir / f"{split}.h5"
        if args.skip_existing and output_file.exists():
            if is_cache_file_valid(output_file, split):
                print(f"\n- Skipping {split}: valid cache exists at {output_file}")
                continue
            print(f"\n- Rebuilding {split}: existing cache is invalid or incomplete ({output_file})")
        preprocess_split(
            split,
            output_dir,
            batch_size=int(args.batch_size),
            compression=compression,
            compression_level=int(args.compression_level),
            max_samples=int(args.max_samples),
            resume_partial=not bool(args.no_resume_partial),
        )

    print("\n" + "=" * 60)
    print("✓ All splits pre-processed successfully!")
    print("=" * 60)
    print(f"\nCache location: {output_dir}")


if __name__ == "__main__":
    main()
