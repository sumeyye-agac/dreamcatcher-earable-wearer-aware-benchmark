from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf

from .audio_features import compute_log_mel


@dataclass
class Sample:
    audio_path: Path
    label: int


class DreamCatcherAudioDataset:
    """
    Minimal DreamCatcher audio dataset loader.
    Assumes:
      - audio files are stored under data_root (possibly nested)
      - labels are provided as a mapping: {relative_audio_path: int_label}
    """

    def __init__(self, data_root: str, labels: Dict[str, int], sr: int = 16000, n_mels: int = 64):
        self.data_root = Path(data_root)
        self.labels = labels
        self.sr = sr
        self.n_mels = n_mels

        self.samples: List[Sample] = self._build_index()

    def _build_index(self) -> List[Sample]:
        samples: List[Sample] = []
        for rel_path, lab in self.labels.items():
            p = self.data_root / rel_path
            if p.exists():
                samples.append(Sample(audio_path=p, label=int(lab)))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        s = self.samples[idx]
        y, sr = sf.read(str(s.audio_path), always_2d=False)

        # Convert to mono if needed
        if y.ndim == 2:
            y = y.mean(axis=1)

        # Resample if needed
        if sr != self.sr:
            # librosa resample (lazy import to keep imports minimal)
            import librosa
            y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=self.sr)

        # Compute log-mel
        feat = compute_log_mel(y=y, sr=self.sr, n_mels=self.n_mels)
        return feat, s.label
