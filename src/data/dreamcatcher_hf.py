from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import sys
import os
import tempfile
import shutil

import numpy as np
from datasets import load_dataset
from datasets import DownloadMode

from .audio_features import compute_log_mel

# Stable label space (event_label) for reproducibility
LABELS = ["breathe", "movements", "swallow", "bruxism", "snore"]
LABEL2ID = {k: i for i, k in enumerate(LABELS)}


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

    def __init__(self, split: str = "train", cfg: DreamCatcherHFAudioConfig | None = None):
        self.cfg = cfg or DreamCatcherHFAudioConfig()
        
        # Log dataset loading
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Loading DreamCatcher Dataset", file=sys.stderr)
        print(f"  Split: {split}", file=sys.stderr)
        print(f"  Config: sleep_event_classification", file=sys.stderr)
        print(f"  Source: THU-PI-Sensing/DreamCatcher (HuggingFace)", file=sys.stderr)
        print(f"  Mode: Normal (with temp cache)", file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)
        
        # Use temporary cache directory to avoid LocalFileSystem issues
        # This prevents the "Loading a dataset cached in a LocalFileSystem is not supported" error
        temp_cache_dir = None
        original_cache = os.environ.get("HF_DATASETS_CACHE")
        default_cache = os.path.expanduser("~/.cache/huggingface/datasets")
        
        try:
            # Create a temporary cache directory for this session
            temp_cache_dir = tempfile.mkdtemp(prefix="hf_cache_")
            os.environ["HF_DATASETS_CACHE"] = temp_cache_dir
            
            # Clear any existing cache for this dataset to avoid LocalFileSystem issues
            cache_to_clear = original_cache if original_cache else default_cache
            dataset_cache_path = os.path.join(cache_to_clear, "THU-PI-Sensing___dream_catcher")
            if os.path.exists(dataset_cache_path):
                try:
                    shutil.rmtree(dataset_cache_path)
                except:
                    pass
            
            print("Loading dataset (this may take a few minutes)...", file=sys.stderr)
            self.ds = load_dataset(
                "THU-PI-Sensing/DreamCatcher", 
                "sleep_event_classification", 
                split=split, 
                streaming=False,
                download_mode=DownloadMode.FORCE_REDOWNLOAD
            )
            num_samples = len(self.ds)
            
            # Clean up temp cache after loading
            if temp_cache_dir and os.path.exists(temp_cache_dir):
                try:
                    shutil.rmtree(temp_cache_dir)
                except:
                    pass
            
            # Restore original cache path
            if original_cache:
                os.environ["HF_DATASETS_CACHE"] = original_cache
            elif "HF_DATASETS_CACHE" in os.environ:
                del os.environ["HF_DATASETS_CACHE"]
            
            print(f"Dataset loaded successfully: {num_samples} samples\n", file=sys.stderr)
            
        except Exception as e:
            # Clean up on error
            if temp_cache_dir and os.path.exists(temp_cache_dir):
                try:
                    shutil.rmtree(temp_cache_dir)
                except:
                    pass
            
            # Restore original cache path
            if original_cache:
                os.environ["HF_DATASETS_CACHE"] = original_cache
            elif "HF_DATASETS_CACHE" in os.environ:
                del os.environ["HF_DATASETS_CACHE"]
            
            raise

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        row = self.ds[idx]

        # HF Audio feature standard: dict with 'array' and 'sampling_rate'
        audio = row["audio_data"]
        y = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio["sampling_rate"])

        # stereo -> mono baseline
        if y.ndim == 2:
            y = y.mean(axis=1)

        if sr != self.cfg.sample_rate:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=self.cfg.sample_rate)

        x = compute_log_mel(y=y, sr=self.cfg.sample_rate, n_mels=self.cfg.n_mels)

        label_str = row.get("event_label", None) or row.get("label", None) or row.get("class", None)
        if label_str is None:
            raise KeyError("Expected 'event_label', 'label', or 'class' in dataset row.")

        if label_str not in LABEL2ID:
            raise ValueError(f"Unknown label: {label_str}. Known: {LABELS}")

        return x, LABEL2ID[label_str]
