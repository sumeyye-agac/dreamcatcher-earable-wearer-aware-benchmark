"""
Train teacher models (ViT/EfficientNet) on DreamCatcher dataset.
Fine-tunes pre-trained vision models on audio spectrograms.
"""
from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.audio_features import compute_log_mel
from src.data.dreamcatcher_hf import LABELS, LABEL2ID, load_dreamcatcher_hf_split
from src.evaluation.metrics import classification_metrics
from src.models.teacher import ViTTeacher, EfficientNetTeacher
from src.utils.artifacts import env_snapshot, run_dir, write_json
from src.utils.benchmarking import append_to_leaderboard, count_params, estimate_model_size_mb
from src.utils.reproducibility import set_seed


def collate_fn(batch, n_mels: int = 64, sr: int = 16000):
    """
    Collate function for teacher training.
    Teacher input: log-mel spectrograms [B, n_mels, T]
    """
    xs_mel = []
    ys = []

    for row in batch:
        if "audio" in row:
            audio = row["audio"]
        elif "audio_data" in row:
            audio = row["audio_data"]
        else:
            raise KeyError("Expected 'audio' (or legacy 'audio_data') in dataset row.")
        y = np.asarray(audio["array"], dtype=np.float32)
        a_sr = int(audio["sampling_rate"])

        if y.ndim == 2:
            if y.shape[0] == 0 or y.shape[1] == 0:
                y = np.zeros((0,), dtype=np.float32)
            else:
                y = y.mean(axis=1)

        # Sanitize NaN/inf
        if y.size > 0:
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        if a_sr != sr:
            import librosa
            y = librosa.resample(y, orig_sr=a_sr, target_sr=sr)

        # Ensure minimum length for spectrogram generation
        if y.shape[0] < 1024:
            y = np.pad(y, (0, 1024 - y.shape[0]), mode="constant")

        xs_mel.append(compute_log_mel(y=y, sr=sr, n_mels=n_mels))

        label_val = row.get("label", None)
        if label_val is None:
            label_val = row.get("event_label", None)
        if label_val is None:
            label_val = row.get("class", None)
        if label_val is None:
            raise KeyError("Expected 'label' (or 'event_label'/'class') in dataset row.")

        if isinstance(label_val, (int, np.integer)):
            ys.append(int(label_val))
        else:
            label_str = str(label_val)
            ys.append(LABEL2ID[label_str])

    # Pad spectrograms [B, n_mels, T]
    max_t = max(x.shape[1] for x in xs_mel)
    x_pad = np.zeros((len(xs_mel), n_mels, max_t), dtype=np.float32)
    for i, x in enumerate(xs_mel):
        x_pad[i, :, : x.shape[1]] = x

    return (
        torch.from_numpy(x_pad),
        torch.tensor(ys, dtype=torch.long),
    )


def train_epoch(model, dl, opt, device, criterion):
    """Train for one epoch."""
    model.train()
    all_true, all_pred = [], []
    total_loss = 0.0

    for xb, yb in tqdm(dl, desc="Training"):
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = criterion(logits, yb)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += float(loss.item()) * xb.size(0)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
        all_pred.extend(preds)
        all_true.extend(yb.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(1, len(dl.dataset))
    m = classification_metrics(all_true, all_pred)
    return avg_loss, m


@torch.no_grad()
def evaluate(model, dl, device, criterion):
    """Evaluate the model."""
    model.eval()
    all_true, all_pred = [], []
    total_loss = 0.0

    for xb, yb in tqdm(dl, desc="Evaluating"):
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = criterion(logits, yb)

        total_loss += float(loss.item()) * xb.size(0)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
        all_pred.extend(preds)
        all_true.extend(yb.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(1, len(dl.dataset))
    m = classification_metrics(all_true, all_pred)
    return avg_loss, m, all_true, all_pred


def main():
    parser = argparse.ArgumentParser(description="Train teacher model on DreamCatcher")
    parser.add_argument(
        "--teacher_type",
        type=str,
        required=True,
        choices=["vit", "efficientnet"],
        help="Teacher model type"
    )
    parser.add_argument("--teacher_name", type=str, help="HuggingFace model name (optional override)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--n_mels", type=int, default=64, help="Number of mel bins")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--early_stop_patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.001, help="Early stopping min delta")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--run_name", type=str, default="", help="Run name")
    parser.add_argument("--out_csv", type=str, default="results/leaderboard.csv", help="Leaderboard CSV")
    parser.add_argument("--steps_csv", type=str, default="results/run_steps.csv", help="Run steps CSV")
    parser.add_argument("--dataset_mode", type=str, default="full", choices=["full", "smoke"])
    parser.add_argument("--max_samples", type=int, default=0, help="Max samples (0 = all)")
    parser.add_argument("--freeze_encoder", action="store_true", help="Keep encoder frozen (only train head)")

    args = parser.parse_args()
    t_run0 = time.time()
    run_started_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    set_seed(args.seed)

    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Set default teacher name based on type
    if args.teacher_name is None:
        if args.teacher_type == "vit":
            args.teacher_name = "google/vit-base-patch16-224"
        else:
            args.teacher_name = "google/efficientnet-b0"

    # Create run name
    if not args.run_name:
        args.run_name = f"{args.teacher_type}_teacher"

    rd = run_dir(args.run_name)
    write_json(rd / "args.json", vars(args))
    write_json(rd / "env.json", env_snapshot())

    print(f"\n{'='*60}")
    print(f"Training Teacher Model: {args.teacher_type}")
    print(f"Model: {args.teacher_name}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"Freeze encoder: {args.freeze_encoder}")
    print(f"{'='*60}\n")

    # Load datasets
    print("Loading datasets...")
    train_ds = load_dreamcatcher_hf_split(
        "train",
        dataset_mode=args.dataset_mode,
        run_name=args.run_name,
        steps_csv=args.steps_csv,
    )
    val_ds = load_dreamcatcher_hf_split(
        "validation",
        dataset_mode=args.dataset_mode,
        run_name=args.run_name,
        steps_csv=args.steps_csv,
    )
    test_ds = load_dreamcatcher_hf_split(
        "test",
        dataset_mode=args.dataset_mode,
        run_name=args.run_name,
        steps_csv=args.steps_csv,
    )

    if args.max_samples and args.max_samples > 0:
        train_ds = train_ds.select(range(min(args.max_samples, len(train_ds))))
        val_ds = val_ds.select(range(min(args.max_samples, len(val_ds))))
        test_ds = test_ds.select(range(min(args.max_samples, len(test_ds))))

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, n_mels=args.n_mels, sr=args.sr),
        num_workers=0,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, n_mels=args.n_mels, sr=args.sr),
        num_workers=0,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, n_mels=args.n_mels, sr=args.sr),
        num_workers=0,
    )

    # Initialize model
    print(f"\nInitializing {args.teacher_type} teacher...")
    if args.teacher_type == "vit":
        model = ViTTeacher(n_classes=len(LABELS), model_name=args.teacher_name)
    else:
        model = EfficientNetTeacher(n_classes=len(LABELS), model_name=args.teacher_name)

    # Optionally unfreeze encoder
    if not args.freeze_encoder:
        print("Unfreezing encoder for fine-tuning...")
        for p in model.encoder.parameters():
            p.requires_grad = True

    model.to(device)

    # Optimizer and loss
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_f1 = -1.0
    best_val_metrics = None
    best_state = None
    best_epoch = -1
    patience_counter = 0

    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")

        tr_loss, tr_m = train_epoch(model, train_dl, opt, device, criterion)
        val_loss, val_m, _, _ = evaluate(model, val_dl, device, criterion)

        print(f"Train Loss: {tr_loss:.4f}, Acc: {tr_m.acc:.4f}, F1: {tr_m.f1_macro:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Acc: {val_m.acc:.4f}, F1: {val_m.f1_macro:.4f}")

        # Early stopping
        if val_m.f1_macro > best_val_f1 + args.early_stop_min_delta:
            best_val_f1 = val_m.f1_macro
            best_val_metrics = val_m
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
            print(f"✓ New best F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            print(f"⏳ Patience: {patience_counter}/{args.early_stop_patience}")

        if patience_counter >= args.early_stop_patience:
            print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
            break

    # Restore best model
    print(f"\nRestoring best model from epoch {best_epoch+1}")
    model.load_state_dict(best_state)

    # Save model checkpoint
    checkpoint_path = rd / "teacher_checkpoint.pth"
    torch.save({
        'model_state_dict': best_state,
        'model_type': args.teacher_type,
        'model_name': args.teacher_name,
        'n_classes': len(LABELS),
        'best_val_f1': best_val_f1,
        'epoch': best_epoch,
    }, checkpoint_path)
    print(f"✓ Saved checkpoint: {checkpoint_path}")

    # Final evaluation on test set
    print(f"\n{'='*60}")
    print("Final Test Evaluation")
    print(f"{'='*60}")

    te_loss, te_m, te_true, te_pred = evaluate(model, test_dl, device, criterion)
    cm = confusion_matrix(te_true, te_pred, labels=list(range(len(LABELS))))

    print(f"\nTest Loss: {te_loss:.4f}")
    print(f"Test Acc:  {te_m.acc:.4f}")
    print(f"Test F1:   {te_m.f1_macro:.4f}")
    print(f"Test Precision: {te_m.precision_macro:.4f}")
    print(f"Test Recall: {te_m.recall_macro:.4f}")
    print(f"Test Balanced Acc: {te_m.balanced_acc:.4f}")

    # Save confusion matrix
    cm_path = rd / "test_confusion_matrix.csv"
    with open(cm_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred", *LABELS])
        for i, row_vals in enumerate(cm.tolist()):
            w.writerow([LABELS[i], *row_vals])
    print(f"✓ Saved confusion matrix: {cm_path}")

    # Benchmark metrics
    param_count = count_params(model)
    model_size_mb = estimate_model_size_mb(model)
    wall_time_s = time.time() - t_run0
    run_finished_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    # Save to leaderboard
    row = {
        "run_started_at_utc": run_started_at_utc,
        "run_finished_at_utc": run_finished_at_utc,
        "run_name": args.run_name,
        "task": "audio_event_label",
        "model": f"{args.teacher_type}_teacher",
        "teacher": args.teacher_name,
        "seed": args.seed,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "epochs_ran": epoch + 1,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "sr": args.sr,
        "n_mels": args.n_mels,
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
        "best_val_f1": round(best_val_f1, 6),
        "best_val_acc": round(best_val_metrics.acc, 6),
        "best_val_precision_macro": round(best_val_metrics.precision_macro, 6),
        "best_val_recall_macro": round(best_val_metrics.recall_macro, 6),
        "best_val_balanced_acc": round(best_val_metrics.balanced_acc, 6),
        "test_f1": round(te_m.f1_macro, 6),
        "test_acc": round(te_m.acc, 6),
        "test_precision_macro": round(te_m.precision_macro, 6),
        "test_recall_macro": round(te_m.recall_macro, 6),
        "test_balanced_acc": round(te_m.balanced_acc, 6),
        "params": param_count,
        "model_size_mb": round(model_size_mb, 4),
        "cpu_latency_ms": 0.0,  # Not measured for teachers
        "wall_time_s": round(wall_time_s, 3),
        "test_cm_csv": str(cm_path),
    }

    append_to_leaderboard(args.out_csv, row)
    print(f"\n✓ Logged to {args.out_csv}")

    # Save metrics JSON
    write_json(
        rd / "metrics.json",
        {
            "run_name": args.run_name,
            "teacher_type": args.teacher_type,
            "teacher_model": args.teacher_name,
            "freeze_encoder": args.freeze_encoder,
            "best_val": best_val_metrics.__dict__,
            "test": te_m.__dict__,
            "params": param_count,
            "model_size_mb": model_size_mb,
            "wall_time_s": wall_time_s,
            "best_epoch": best_epoch,
            "epochs_ran": epoch + 1,
            "run_started_at_utc": run_started_at_utc,
            "run_finished_at_utc": run_finished_at_utc,
            "checkpoint_path": str(checkpoint_path),
            "test_confusion_matrix_csv": str(cm_path),
        },
    )

    print(f"\n{'='*60}")
    print("✓ Training complete!")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Best Val F1: {best_val_f1:.4f} (epoch {best_epoch+1})")
    print(f"Test F1: {te_m.f1_macro:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
