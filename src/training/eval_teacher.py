"""
Evaluate the Wav2Vec2 teacher model on DreamCatcher test set.
This provides a baseline to compare student models and KD effectiveness.
"""
from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dreamcatcher_hf import LABELS, load_dreamcatcher_hf_split
from src.evaluation.metrics import classification_metrics
from src.models.teacher.wav2vec2_teacher import Wav2Vec2Teacher
from src.utils.artifacts import run_dir, write_json
from src.utils.benchmarking import append_to_leaderboard, count_params, estimate_model_size_mb


def collate_fn(batch, sr: int = 16000):
    """
    Collate function for teacher evaluation.
    Teacher input: raw audio waveforms [B, T]
    """
    ys_raw = []
    labels = []

    for row in batch:
        audio_dict = row.get("audio") or row.get("audio_data")
        y_raw = np.asarray(audio_dict["array"], dtype=np.float32)
        sample_rate = int(audio_dict["sampling_rate"])

        # Handle multi-channel audio
        if y_raw.ndim == 2:
            if y_raw.shape[0] == 0 or y_raw.shape[1] == 0:
                y_raw = np.zeros((0,), dtype=np.float32)
            else:
                y_raw = y_raw.mean(axis=1)

        # Handle invalid audio
        if y_raw.size == 0 or not np.isfinite(y_raw).all():
            y_raw = np.zeros((16000,), dtype=np.float32)  # 1 second of silence

        y_raw = np.nan_to_num(y_raw, nan=0.0, posinf=0.0, neginf=0.0)

        # Resample if needed
        if sample_rate != sr:
            import librosa
            y_raw = librosa.resample(y_raw, orig_sr=sample_rate, target_sr=sr)

        # Ensure minimum length for Wav2Vec2 (needs at least 320 samples for feature extraction)
        min_samples = 1024  # Safe minimum for Wav2Vec2
        if y_raw.shape[0] < min_samples:
            y_raw = np.pad(y_raw, (0, min_samples - y_raw.shape[0]), mode='constant')

        ys_raw.append(torch.from_numpy(y_raw))

        # Extract label
        label_val = row.get("label") or row.get("event_label") or row.get("class")
        if label_val is None:
            raise KeyError(f"No label found in row. Keys: {list(row.keys())}")
        if isinstance(label_val, (int, np.integer)):
            labels.append(int(label_val))
        else:
            from src.data.dreamcatcher_hf import LABEL2ID
            label_str = str(label_val)
            if label_str not in LABEL2ID:
                raise ValueError(f"Unknown label: {label_str}. Known labels: {list(LABEL2ID.keys())}")
            labels.append(LABEL2ID[label_str])

    # Pad sequences to max length in batch
    max_len = max(y.shape[0] for y in ys_raw)
    ys_padded = []
    for y in ys_raw:
        if y.shape[0] < max_len:
            padding = torch.zeros(max_len - y.shape[0])
            y = torch.cat([y, padding])
        ys_padded.append(y)

    xs = torch.stack(ys_padded)
    ys = torch.tensor(labels, dtype=torch.long)

    return xs, ys


def evaluate_teacher(
    teacher: Wav2Vec2Teacher,
    test_ds,
    device: str = "cpu",
    batch_size: int = 4,
    sr: int = 16000,
):
    """Evaluate teacher model on test set."""
    teacher.eval()
    teacher.to(device)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, sr=sr),
        num_workers=0,
    )

    all_preds = []
    all_true = []

    print(f"\nEvaluating teacher on {len(test_ds)} test samples...")
    with torch.no_grad():
        for xs, ys in tqdm(test_loader, desc="Teacher inference"):
            xs = xs.to(device)
            ys = ys.to(device)

            logits = teacher(xs)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_true.extend(ys.cpu().numpy())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    metrics = classification_metrics(all_true, all_preds, n_classes=len(LABELS))
    cm = confusion_matrix(all_true, all_preds, labels=list(range(len(LABELS))))

    return metrics, cm


def main():
    parser = argparse.ArgumentParser(description="Evaluate Wav2Vec2 teacher model")
    parser.add_argument("--teacher_name", default="facebook/wav2vec2-base", help="HuggingFace teacher model name")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate")
    parser.add_argument("--dataset_mode", default="full", choices=["full", "smoke"], help="Dataset mode")
    parser.add_argument("--max_samples", type=int, default=0, help="Max samples (0 = all)")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda/mps)")
    parser.add_argument("--run_name", default="wav2vec2_teacher", help="Run name")
    parser.add_argument("--out_csv", default="results/leaderboard.csv", help="Leaderboard CSV")
    parser.add_argument("--steps_csv", default="results/run_steps.csv", help="Run steps CSV")

    args = parser.parse_args()

    run_name = args.run_name
    run_started_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    t_run0 = time.time()

    rd = run_dir(run_name)
    write_json(rd / "args.json", vars(args))
    write_json(rd / "env.json", {})

    print(f"\n{'='*60}")
    print(f"Evaluating Teacher Model: {args.teacher_name}")
    print(f"{'='*60}\n")

    # Determine device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, using CPU")
    elif device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
        print("MPS not available, using CPU")

    # Load test dataset
    print("Loading test dataset...")
    test_ds = load_dreamcatcher_hf_split(
        "test",
        dataset_mode=args.dataset_mode,
        run_name=run_name,
        steps_csv=args.steps_csv,
    )

    if args.max_samples and args.max_samples > 0:
        n = min(int(args.max_samples), len(test_ds))
        test_ds = test_ds.select(range(n))

    print(f"Test set size: {len(test_ds)}")

    # Load teacher model
    print(f"\nLoading teacher model: {args.teacher_name}")
    teacher = Wav2Vec2Teacher(
        model_name=args.teacher_name,
        n_classes=len(LABELS),
    )

    # Evaluate
    te_m, cm = evaluate_teacher(
        teacher=teacher,
        test_ds=test_ds,
        device=device,
        batch_size=args.batch_size,
        sr=args.sr,
    )

    print("\n" + "="*60)
    print("Teacher Test Results:")
    print("="*60)
    print(f"Accuracy:          {te_m.acc:.4f}")
    print(f"Balanced Accuracy: {te_m.balanced_acc:.4f}")
    print(f"F1 (macro):        {te_m.f1_macro:.4f}")
    print(f"Precision (macro): {te_m.precision_macro:.4f}")
    print(f"Recall (macro):    {te_m.recall_macro:.4f}")
    print("="*60)

    # Save confusion matrix
    cm_path = rd / "test_confusion_matrix.csv"
    with open(cm_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred", *LABELS])
        for i, row_vals in enumerate(cm.tolist()):
            w.writerow([LABELS[i], *row_vals])
    print(f"\nConfusion matrix saved: {cm_path}")

    # Benchmark metrics
    param_count = count_params(teacher)
    model_size_mb = estimate_model_size_mb(teacher)
    # Note: Latency measurement skipped for teacher (too slow and not relevant for comparison)
    lat_ms = 0.0

    wall_time_s = time.time() - t_run0
    run_finished_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    # Save to leaderboard
    row = {
        "run_started_at_utc": run_started_at_utc,
        "run_finished_at_utc": run_finished_at_utc,
        "run_name": run_name,
        "task": "audio_event_label",
        "model": "wav2vec2_teacher",
        "teacher": args.teacher_name,
        "seed": 42,
        "epochs": 0,
        "best_epoch": 0,
        "epochs_ran": 0,
        "batch_size": args.batch_size,
        "lr": "",
        "sr": args.sr,
        "n_mels": "",
        "rnn_hidden": "",
        "rnn_layers": "",
        "cbam_reduction": "",
        "cbam_sa_kernel": "",
        "att_mode": "",
        "alpha": "",
        "tau": "",
        "dataset_mode": args.dataset_mode,
        "max_samples": args.max_samples if args.max_samples else "",
        "invalid_audio_policy": "",
        "best_val_f1": "",
        "best_val_acc": "",
        "best_val_precision_macro": "",
        "best_val_recall_macro": "",
        "best_val_balanced_acc": "",
        "test_f1": round(te_m.f1_macro, 6),
        "test_acc": round(te_m.acc, 6),
        "test_precision_macro": round(te_m.precision_macro, 6),
        "test_recall_macro": round(te_m.recall_macro, 6),
        "test_balanced_acc": round(te_m.balanced_acc, 6),
        "params": param_count,
        "model_size_mb": round(model_size_mb, 4),
        "cpu_latency_ms": lat_ms,
        "wall_time_s": round(wall_time_s, 3),
        "test_cm_csv": str(cm_path),
    }

    append_to_leaderboard(args.out_csv, row)
    print(f"\nLogged to {args.out_csv}")

    # Save metrics JSON
    write_json(
        rd / "metrics.json",
        {
            "run_name": run_name,
            "teacher_model": args.teacher_name,
            "test": te_m.__dict__,
            "params": param_count,
            "model_size_mb": model_size_mb,
            "cpu_latency_ms": lat_ms,
            "wall_time_s": wall_time_s,
            "run_started_at_utc": run_started_at_utc,
            "run_finished_at_utc": run_finished_at_utc,
            "test_confusion_matrix_csv": str(cm_path),
        },
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
