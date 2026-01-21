from __future__ import annotations

import argparse
import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datetime import datetime, timezone
from sklearn.metrics import confusion_matrix

from src.data.audio_features import compute_log_mel
from src.data.dreamcatcher_hf import LABELS, LABEL2ID, load_dreamcatcher_hf_split
from src.evaluation.metrics import classification_metrics
from src.models.tinycnn import TinyCNN
from src.models.crnn import CRNN
from src.models.crnn_cbam import CRNN_CBAM
from src.models.teacher import ViTTeacher, EfficientNetTeacher
from src.utils.reproducibility import set_seed
from src.utils.benchmarking import (
    count_params,
    estimate_model_size_mb,
    measure_cpu_latency,
    append_to_leaderboard,
)
from src.utils.artifacts import env_snapshot, run_dir, write_json


def make_student(model_name: str, n_classes: int, args) -> torch.nn.Module:
    model_name = model_name.lower()
    if model_name == "tinycnn":
        return TinyCNN(n_classes=n_classes)
    if model_name == "crnn":
        return CRNN(n_classes=n_classes, rnn_hidden=args.rnn_hidden, rnn_layers=args.rnn_layers)
    if model_name == "crnn_cbam":
        use_ca = args.att_mode in ("cbam", "ca")
        use_sa = args.att_mode in ("cbam", "sa")
        return CRNN_CBAM(
            n_classes=n_classes,
            rnn_hidden=args.rnn_hidden,
            rnn_layers=args.rnn_layers,
            cbam_reduction=args.cbam_reduction,
            cbam_sa_kernel=args.cbam_sa_kernel,
            use_ca=use_ca,
            use_sa=use_sa,
        )
    raise ValueError("student must be one of: tinycnn, crnn, crnn_cbam")


def collate_fn(batch, n_mels: int = 64, sr: int = 16000):
    """
    Student input: log-mel [B, 1, n_mels, Tpad]
    Teacher input: log-mel [B, n_mels, Tpad] (for ViT/EfficientNet)
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

        # sanitize NaN/inf
        if y.size > 0:
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        if a_sr != sr:
            import librosa

            y = librosa.resample(y, orig_sr=a_sr, target_sr=sr)

        # Avoid degenerate spectrograms / teacher crashes on extremely short clips
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

    # pad mel for student [B, 1, n_mels, T]
    max_t = max(x.shape[1] for x in xs_mel)
    x_pad_student = np.zeros((len(xs_mel), 1, n_mels, max_t), dtype=np.float32)
    for i, x in enumerate(xs_mel):
        x_pad_student[i, 0, :, : x.shape[1]] = x

    # pad mel for teacher [B, n_mels, T] (no channel dimension)
    x_pad_teacher = np.zeros((len(xs_mel), n_mels, max_t), dtype=np.float32)
    for i, x in enumerate(xs_mel):
        x_pad_teacher[i, :, : x.shape[1]] = x

    return (
        torch.from_numpy(x_pad_student),
        torch.tensor(ys, dtype=torch.long),
        torch.from_numpy(x_pad_teacher),
    )


def kd_loss(student_logits, teacher_logits, y_true, alpha: float, tau: float):
    """
    Loss = alpha * CE + (1-alpha) * KL(soft teacher || soft student) * tau^2
    """
    ce = F.cross_entropy(student_logits, y_true)
    s_logp = F.log_softmax(student_logits / tau, dim=1)
    t_p = F.softmax(teacher_logits / tau, dim=1)
    kl = F.kl_div(s_logp, t_p, reduction="batchmean") * (tau * tau)
    return alpha * ce + (1.0 - alpha) * kl


def run_train_epoch(student, teacher, dl, opt, device, alpha, tau):
    student.train()
    all_true, all_pred = [], []
    total_loss = 0.0

    for xb_mel, yb, xb_mel_teacher in dl:
        xb_mel = xb_mel.to(device)
        yb = yb.to(device)
        xb_mel_teacher = xb_mel_teacher.to(device)

        with torch.no_grad():
            t_logits = teacher(xb_mel_teacher)

        s_logits = student(xb_mel)
        loss = kd_loss(s_logits, t_logits, yb, alpha=alpha, tau=tau)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += float(loss.item()) * xb_mel.size(0)
        preds = torch.argmax(s_logits, dim=1).detach().cpu().numpy().tolist()
        all_pred.extend(preds)
        all_true.extend(yb.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(1, len(dl.dataset))
    m = classification_metrics(all_true, all_pred)
    return avg_loss, m


@torch.no_grad()
def evaluate(student, dl, device, *, return_preds: bool = False):
    student.eval()
    all_true, all_pred = [], []

    for xb_mel, yb, xb_mel_teacher in dl:
        xb_mel = xb_mel.to(device)
        yb = yb.to(device)

        logits = student(xb_mel)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
        all_pred.extend(preds)
        all_true.extend(yb.detach().cpu().numpy().tolist())

    m = classification_metrics(all_true, all_pred)
    if return_preds:
        return m, all_true, all_pred
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--student", type=str, default="crnn", choices=["tinycnn", "crnn", "crnn_cbam"]
    )

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="Early stopping patience on val_f1 (0 disables early stopping).",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=0.0,
        help="Minimum val_f1 improvement to reset patience.",
    )

    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n_mels", type=int, default=64)
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="",
        help="Optional HF datasets cache dir override (use this if disk is full).",
    )

    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--tau", type=float, default=5.0)

    parser.add_argument("--rnn_hidden", type=int, default=64)
    parser.add_argument("--rnn_layers", type=int, default=1)
    parser.add_argument("--cbam_reduction", type=int, default=8)
    parser.add_argument("--cbam_sa_kernel", type=int, default=7)
    parser.add_argument(
        "--att_mode",
        type=str,
        default="cbam",
        choices=["cbam", "ca", "sa"],
        help="Attention mode for crnn_cbam: cbam=CA+SA, ca=channel-only, sa=spatial-only.",
    )

    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--out_csv", type=str, default="results/leaderboard.csv")
    parser.add_argument("--latency_T", type=int, default=400)
    parser.add_argument(
        "--teacher_type",
        type=str,
        default="vit",
        choices=["vit", "efficientnet"],
        help="Teacher model type: vit (ViT-base) or efficientnet (EfficientNet-b0)",
    )
    parser.add_argument("--teacher_name", type=str, default="google/vit-base-patch16-224")
    parser.add_argument(
        "--teacher_checkpoint",
        type=str,
        default="",
        help="Path to trained teacher checkpoint (optional)",
    )
    parser.add_argument("--dataset_mode", type=str, default="full", choices=["full", "smoke"])
    parser.add_argument("--steps_csv", type=str, default="results/run_steps.csv")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Optional cap per split for faster smoke runs (0 = no cap).",
    )

    args = parser.parse_args()
    import time

    t_run0 = time.time()
    run_started_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    set_seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if args.run_name:
        run_name = args.run_name
    else:
        suffix = f"_att{args.att_mode}" if args.student == "crnn_cbam" else ""
        run_name = f"kd_{args.student}{suffix}_a{args.alpha}_t{args.tau}_seed{args.seed}"
    rd = run_dir(run_name)
    write_json(rd / "args.json", vars(args) | {"run_name": run_name})
    write_json(rd / "env.json", env_snapshot())

    train_ds = load_dreamcatcher_hf_split(
        "train",
        dataset_mode=args.dataset_mode,
        run_name=run_name,
        steps_csv=args.steps_csv,
        cache_dir=(args.cache_dir or None),
    )
    val_ds = load_dreamcatcher_hf_split(
        "validation",
        dataset_mode=args.dataset_mode,
        run_name=run_name,
        steps_csv=args.steps_csv,
        cache_dir=(args.cache_dir or None),
    )
    test_ds = load_dreamcatcher_hf_split(
        "test",
        dataset_mode=args.dataset_mode,
        run_name=run_name,
        steps_csv=args.steps_csv,
        cache_dir=(args.cache_dir or None),
    )

    if args.max_samples and args.max_samples > 0:
        n_tr = min(args.max_samples, len(train_ds))
        n_va = min(args.max_samples, len(val_ds))
        n_te = min(args.max_samples, len(test_ds))
        train_ds = train_ds.select(range(n_tr))
        val_ds = val_ds.select(range(n_va))
        test_ds = test_ds.select(range(n_te))
        print(f"[kd] max_samples applied: train={n_tr} val={n_va} test={n_te}")

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, n_mels=args.n_mels, sr=args.sr),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, n_mels=args.n_mels, sr=args.sr),
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, n_mels=args.n_mels, sr=args.sr),
    )

    student = make_student(args.student, n_classes=len(LABELS), args=args).to(device)

    # Initialize teacher based on type
    if args.teacher_checkpoint:
        # Load from trained checkpoint
        print(f"Loading teacher from checkpoint: {args.teacher_checkpoint}")
        if args.teacher_type == "vit":
            teacher = ViTTeacher.load_from_checkpoint(args.teacher_checkpoint, device=str(device))
        elif args.teacher_type == "efficientnet":
            teacher = EfficientNetTeacher.load_from_checkpoint(
                args.teacher_checkpoint, device=str(device)
            )
        else:
            raise ValueError(f"Unknown teacher type: {args.teacher_type}")
    else:
        # Initialize from pre-trained (not recommended - teacher needs training first)
        print(
            f"⚠️  WARNING: Initializing teacher from pre-trained {args.teacher_name} without fine-tuning!"
        )
        print("    For best results, train the teacher first using train_teacher.py")
        if args.teacher_type == "vit":
            teacher = ViTTeacher(n_classes=len(LABELS), model_name=args.teacher_name)
        elif args.teacher_type == "efficientnet":
            teacher = EfficientNetTeacher(n_classes=len(LABELS), model_name=args.teacher_name)
        else:
            raise ValueError(f"Unknown teacher type: {args.teacher_type}")
        teacher.to(device)

    teacher.eval()

    opt = torch.optim.Adam(student.parameters(), lr=args.lr)

    best_val_f1 = -1.0
    best_val_metrics = None
    best_state = None
    best_epoch = 0
    epochs_ran = 0
    no_improve = 0
    patience = int(args.early_stop_patience) if args.early_stop_patience else 0
    min_delta = float(args.early_stop_min_delta) if args.early_stop_min_delta else 0.0

    for epoch in range(1, args.epochs + 1):
        epochs_ran = epoch
        tr_loss, tr_m = run_train_epoch(
            student, teacher, train_dl, opt, device, args.alpha, args.tau
        )
        va_m = evaluate(student, val_dl, device)

        improved = va_m.f1_macro > (best_val_f1 + min_delta)
        if improved:
            best_val_f1 = va_m.f1_macro
            best_val_metrics = va_m
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"epoch={epoch} "
            f"train_loss={tr_loss:.4f} train_acc={tr_m.acc * 100:.2f}% train_f1={tr_m.f1_macro * 100:.2f}% | "
            f"val_acc={va_m.acc * 100:.2f}% val_f1={va_m.f1_macro * 100:.2f}%"
        )

        if patience > 0 and no_improve >= patience:
            print(
                f"[kd] early_stop at epoch={epoch} best_epoch={best_epoch} best_val_f1={best_val_f1 * 100:.2f}%"
            )
            break

    if best_state is not None:
        student.load_state_dict(best_state)

    te_m, te_true, te_pred = evaluate(student, test_dl, device, return_preds=True)
    print(f"test_acc={te_m.acc * 100:.2f}% test_f1={te_m.f1_macro * 100:.2f}%")

    # Save test confusion matrix as a per-run artifact (CSV).
    cm = confusion_matrix(te_true, te_pred, labels=list(range(len(LABELS))))
    cm_path = rd / "test_confusion_matrix.csv"
    with cm_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred", *LABELS])
        for i, row_vals in enumerate(cm.tolist()):
            w.writerow([LABELS[i], *row_vals])

    # Benchmark logging
    param_count = count_params(student)
    model_size_mb = estimate_model_size_mb(student)
    lat_ms = measure_cpu_latency(student, input_shape=(1, 1, args.n_mels, args.latency_T))
    wall_time_s = time.time() - t_run0
    run_finished_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    row = {
        "run_started_at_utc": run_started_at_utc,
        "run_finished_at_utc": run_finished_at_utc,
        "run_name": run_name,
        "task": "audio_event_label",
        "model": args.student,
        "teacher": args.teacher_name,
        "seed": args.seed,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "epochs_ran": epochs_ran,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "sr": args.sr,
        "n_mels": args.n_mels,
        "rnn_hidden": args.rnn_hidden if args.student != "tinycnn" else "",
        "rnn_layers": args.rnn_layers if args.student != "tinycnn" else "",
        "cbam_reduction": args.cbam_reduction if args.student == "crnn_cbam" else "",
        "cbam_sa_kernel": args.cbam_sa_kernel if args.student == "crnn_cbam" else "",
        "att_mode": args.att_mode if args.student == "crnn_cbam" else "",
        "alpha": args.alpha,
        "tau": args.tau,
        "dataset_mode": args.dataset_mode,
        "max_samples": args.max_samples if args.max_samples else "",
        "invalid_audio_policy": "",
        "best_val_f1": round(best_val_f1, 6),
        "best_val_acc": round(best_val_metrics.acc, 6) if best_val_metrics is not None else "",
        "best_val_precision_macro": round(best_val_metrics.precision_macro, 6)
        if best_val_metrics is not None
        else "",
        "best_val_recall_macro": round(best_val_metrics.recall_macro, 6)
        if best_val_metrics is not None
        else "",
        "best_val_balanced_acc": round(best_val_metrics.balanced_acc, 6)
        if best_val_metrics is not None
        else "",
        "test_f1": round(te_m.f1_macro, 6),
        "test_acc": round(te_m.acc, 6),
        "test_precision_macro": round(te_m.precision_macro, 6),
        "test_recall_macro": round(te_m.recall_macro, 6),
        "test_balanced_acc": round(te_m.balanced_acc, 6),
        "params": param_count,
        "model_size_mb": round(model_size_mb, 4),
        "cpu_latency_ms": round(lat_ms, 4),
        "wall_time_s": round(wall_time_s, 3),
        "test_cm_csv": str(cm_path),
    }

    append_to_leaderboard(args.out_csv, row)
    print(f"Logged to {args.out_csv}")
    write_json(
        rd / "metrics.json",
        {
            "run_name": run_name,
            "best_epoch": best_epoch,
            "epochs_ran": epochs_ran,
            "best_val": (best_val_metrics.__dict__ if best_val_metrics is not None else None),
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
    print("Done.")


if __name__ == "__main__":
    main()
