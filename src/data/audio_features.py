import librosa
import numpy as np


def compute_log_mel(
    y: np.ndarray,
    sr: int,
    n_mels: int = 128,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    fmin: int = 20,
    fmax: int | None = None,
) -> np.ndarray:
    """
    Compute log-mel spectrogram.

    Default params match DreamCatcher paper (EarVAS config):
    - 25ms window (400 samples @ 16kHz)
    - 10ms stride (160 samples @ 16kHz)
    - 128 mel bins

    Returns shape: [n_mels, time]
    """
    if win_length is None:
        win_length = n_fft
    if fmax is None:
        fmax = sr // 2

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)
