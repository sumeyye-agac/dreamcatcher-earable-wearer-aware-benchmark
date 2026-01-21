"""
Evaluate teacher models (ViT/EfficientNet) on DreamCatcher test set (Respiratory subset).
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

from src.data.dreamcatcher_subset import RESPIRATORY_LABELS, RESPIRATORY_ORIGINAL_INDICES, RESPIRATORY_LABEL_MAP, load_respiratory_hf_split

# Use respiratory subset labels (3 classes)
LABELS = RESPIRATORY_LABELS
from src.evaluation.metrics import classification_metrics
from src.models.teacher import ViTTeacher, EfficientNetTeacher
from src.utils.artifacts import run_dir, write_json
from src.utils.benchmarking import append_to_leaderboard, count_params, estimate_model_size_mb


def collate_fn(batch, sr: int = 16000, n_mels: int = 64):
    """
    Collate function for teacher evaluation.
    Teacher input: log-mel spectrograms [B, n_mels, T]
    """
    xs_mel = []
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

        # Ensure minimum length for spectrogram generation
        min_samples = 1024
        if y_raw.shape[0] < min_samples:
            y_raw = np.pad(y_raw, (0, min_samples - y_raw.shape[0]), mode='constant')

        # Convert to log-mel spectrogram
        from src.data.audio_features import compute_log_mel
        mel = compute_log_mel(y=y_raw, sr=sr, n_mels=n_mels)
        xs_mel.append(mel)

        # Extract label
        label_val = None
        if "label" in row:
            label_val = row["label"]
        elif "event_label" in row:
            label_val = row["event_label"]
        elif "class" in row:
            label_val = row["class"]

        if label_val is None or (isinstance(label_val, dict) and not label_val):
            raise KeyError(f"No valid label found in row. Keys: {list(row.keys())}, label value: {row.get('label')}")

        # Remap label from 9-class (5,6,7) to 3-class (0,1,2)
        if isinstance(label_val, (int, np.integer)):
            label_9class = int(label_val)
        else:
            from src.data.dreamcatcher_hf import LABEL2ID as LABEL2ID_9CLASS
            label_str = str(label_val)
            if label_str not in LABEL2ID_9CLASS:
                raise ValueError(f"Unknown label: {label_str}. Known labels: {list(LABEL2ID_9CLASS.keys())}")
            label_9class = LABEL2ID_9CLASS[label_str]

        # Remap to 3-class space
        if label_9class not in RESPIRATORY_LABEL_MAP:
            raise ValueError(f"Invalid label {label_9class}, expected one of {RESPIRATORY_ORIGINAL_INDICES}")
        label_3class = RESPIRATORY_LABEL_MAP[label_9class]
        labels.append(label_3class)

    # Pad spectrograms to max length in batch [B, n_mels, T]
    max_t = max(mel.shape[1] for mel in xs_mel)
    x_pad = np.zeros((len(xs_mel), n_mels, max_t), dtype=np.float32)
    for i, mel in enumerate(xs_mel):
        x_pad[i, :, :mel.shape[1]] = mel

    xs = torch.from_numpy(x_pad)
    ys = torch.tensor(labels, dtype=torch.long)

    return xs, ys


def evaluate_teacher(
    teacher,
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

            logits = teacher(xs)  # [B, 3]
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_true.extend(ys.cpu().numpy())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    metrics = classification_metrics(all_true, all_preds)
    cm = confusion_matrix(all_true, all_preds, labels=list(range(len(LABELS))))

    return metrics, cm


def main():
    parser = argparse.ArgumentParser(description="Evaluate teacher model on respiratory subset")
    parser.add_argument(
        "--teacher_type",
        type=str,
        default="vit",
        choices=["vit", "efficientnet"],
        help="Teacher model type: vit (ViT-base) or efficientnet (EfficientNet-b0)"
    )
    parser.add_argument("--teacher_name", default="google/vit-base-patch16-224", help="HuggingFace teacher model name")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate")
    parser.add_argument("--dataset_mode", default="full", choices=["full", "smoke"], help="Dataset mode")
    parser.add_argument("--max_samples", type=int, default=0, help="Max samples (0 = all)")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda/mps)")
    parser.add_argument("--run_name", default="vit_teacher_respiratory", help="Run name")
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

    # Load test dataset (respiratory subset)
    print("Loading test dataset (respiratory subset)...")
    test_ds = load_respiratory_hf_split(
        "test",
        dataset_mode=args.dataset_mode,
        run_name=run_name,
        steps_csv=args.steps_csv,
    )

    if args.max_samples and args.max_samples > 0:
        n = min(int(args.max_samples), len(test_ds))
        test_ds = test_ds.select(range(n))

    print(f"Test set size: {len(test_ds)}")

    # Load teacher model (outputs 3 classes for respiratory subset)
    print(f"\nLoading teacher model: {args.teacher_type} - {args.teacher_name}")
    if args.teacher_type == "vit":
        teacher = ViTTeacher(n_classes=len(LABELS), model_name=args.teacher_name)
    elif args.teacher_type == "efficientnet":
        teacher = EfficientNetTeacher(n_classes=len(LABELS), model_name=args.teacher_name)
    else:
        raise ValueError(f"Unknown teacher type: {args.teacher_type}")

    teacher.to(device)
    teacher.eval()

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
        "task": "respiratory_subset",
        "model": f"{args.teacher_type}_teacher",
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
