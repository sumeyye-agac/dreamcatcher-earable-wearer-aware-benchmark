from src.data.dreamcatcher_loader import DreamCatcherAudioDataset

# Example dummy label mapping (replace with your actual mapping once you parse labels)
labels = {
    "subject_01/sample_0001.wav": 0,
    "subject_01/sample_0002.wav": 1,
}

ds = DreamCatcherAudioDataset(data_root="data/audio", labels=labels, sr=16000, n_mels=64)

print("Num samples:", len(ds))
if len(ds) > 0:
    x, y = ds[0]
    print("Feature shape:", x.shape, "Label:", y)
