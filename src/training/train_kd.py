from __future__ import annotations

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.audio_features import compute_log_mel
from src.data.dreamcatcher_hf import LABELS, LABEL2ID, load_dreamcatcher_hf_split
from src.evaluation.metrics import classification_metrics
from src.models.tinycnn import TinyCNN
from src.models.crnn import CRNN
from src.models.crnn_cbam import CRNN_CBAM
from src.models.teacher.wav2vec2_teacher import Wav2Vec2Teacher
from src.utils.reproducibility import set_seed
from src.utils.benchmarking import count_params, estimate_model_size_mb, measure_cpu_latency, append_to_leaderboard
from src.utils.artifacts import env_snapshot, run_dir, write_json


def make_student(model_name: str, n_classes: int, args) -> torch.nn.Module:
    model_name = model_name.lower()
    if model_name == "tinycnn":
        return TinyCNN(n_classes=n_classes)
    if model_name == "crnn":
        return CRNN(n_classes=n_classes, rnn_hidden=args.rnn_hidden, rnn_layers=args.rnn_layers)
    if model_name == "crnn_cbam":
        return CRNN_CBAM(
            n_classes=n_classes,
            rnn_hidden=args.rnn_hidden,
            rnn_layers=args.rnn_layers,
            cbam_reduction=args.cbam_reduction,
            cbam_sa_kernel=args.cbam_sa_kernel,
        )
    raise ValueError("student must be one of: tinycnn, crnn, crnn_cbam")


def collate_fn(batch, n_mels: int = 64, sr: int = 16000):
    """
    Student input: log-mel [B, 1, n_mels, Tpad]
    Teacher input: raw audio [B, Tpad_raw]
    """
    xs_mel = []
    ys = []
    raws = []

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
        raws.append(y)

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

    # pad mel
    max_t = max(x.shape[1] for x in xs_mel)
    x_pad = np.zeros((len(xs_mel), 1, n_mels, max_t), dtype=np.float32)
    for i, x in enumerate(xs_mel):
        x_pad[i, 0, :, : x.shape[1]] = x

    # pad raw
    max_raw = max(r.shape[0] for r in raws)
    raw_pad = np.zeros((len(raws), max_raw), dtype=np.float32)
    for i, r in enumerate(raws):
        raw_pad[i, : r.shape[0]] = r

    return (
        torch.from_numpy(x_pad),
        torch.tensor(ys, dtype=torch.long),
        torch.from_numpy(raw_pad),
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

    for xb_mel, yb, xb_raw in dl:
        xb_mel = xb_mel.to(device)
        yb = yb.to(device)
        xb_raw = xb_raw.to(device)

        with torch.no_grad():
            t_logits = teacher(xb_raw)

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
def evaluate(student, dl, device):
    student.eval()
    all_true, all_pred = [], []

    for xb_mel, yb, xb_raw in dl:
        xb_mel = xb_mel.to(device)
        yb = yb.to(device)

        logits = student(xb_mel)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
        all_pred.extend(preds)
        all_true.extend(yb.detach().cpu().numpy().tolist())

    m = classification_metrics(all_true, all_pred)
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=str, default="crnn", choices=["tinycnn", "crnn", "crnn_cbam"])

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n_mels", type=int, default=64)

    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--tau", type=float, default=5.0)

    parser.add_argument("--rnn_hidden", type=int, default=64)
    parser.add_argument("--rnn_layers", type=int, default=1)
    parser.add_argument("--cbam_reduction", type=int, default=8)
    parser.add_argument("--cbam_sa_kernel", type=int, default=7)

    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--out_csv", type=str, default="results/leaderboard.csv")
    parser.add_argument("--latency_T", type=int, default=400)
    parser.add_argument("--teacher_name", type=str, default="facebook/wav2vec2-base")
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

    set_seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = args.run_name if args.run_name else f"kd_{args.student}_a{args.alpha}_t{args.tau}_seed{args.seed}"
    rd = run_dir(run_name)
    write_json(rd / "args.json", vars(args) | {"run_name": run_name})
    write_json(rd / "env.json", env_snapshot())

    train_ds = load_dreamcatcher_hf_split("train", dataset_mode=args.dataset_mode, run_name=run_name, steps_csv=args.steps_csv)
    val_ds = load_dreamcatcher_hf_split("validation", dataset_mode=args.dataset_mode, run_name=run_name, steps_csv=args.steps_csv)
    test_ds = load_dreamcatcher_hf_split("test", dataset_mode=args.dataset_mode, run_name=run_name, steps_csv=args.steps_csv)

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
    teacher = Wav2Vec2Teacher(n_classes=len(LABELS), model_name=args.teacher_name).to(device)
    teacher.eval()

    opt = torch.optim.Adam(student.parameters(), lr=args.lr)

    best_val_f1 = -1.0
    best_val_metrics = None
    best_state = None

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_m = run_train_epoch(student, teacher, train_dl, opt, device, args.alpha, args.tau)
        va_m = evaluate(student, val_dl, device)

        if va_m.f1_macro > best_val_f1:
            best_val_f1 = va_m.f1_macro
            best_val_metrics = va_m
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}

        print(
            f"epoch={epoch} "
            f"train_loss={tr_loss:.4f} train_acc={tr_m.acc:.4f} train_f1={tr_m.f1_macro:.4f} | "
            f"val_acc={va_m.acc:.4f} val_f1={va_m.f1_macro:.4f}"
        )

    if best_state is not None:
        student.load_state_dict(best_state)

    te_m = evaluate(student, test_dl, device)
    print(f"test_acc={te_m.acc:.4f} test_f1={te_m.f1_macro:.4f}")

    # Benchmark logging
    param_count = count_params(student)
    model_size_mb = estimate_model_size_mb(student)
    lat_ms = measure_cpu_latency(student, input_shape=(1, 1, args.n_mels, args.latency_T))
    wall_time_s = time.time() - t_run0

    row = {
        "run_name": run_name,
        "task": "audio_event_label",
        "model": args.student,
        "teacher": args.teacher_name,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "sr": args.sr,
        "n_mels": args.n_mels,
        "rnn_hidden": args.rnn_hidden if args.student != "tinycnn" else "",
        "rnn_layers": args.rnn_layers if args.student != "tinycnn" else "",
        "cbam_reduction": args.cbam_reduction if args.student == "crnn_cbam" else "",
        "cbam_sa_kernel": args.cbam_sa_kernel if args.student == "crnn_cbam" else "",
        "alpha": args.alpha,
        "tau": args.tau,
        "dataset_mode": args.dataset_mode,
        "max_samples": args.max_samples if args.max_samples else "",
        "invalid_audio_policy": "",
        "best_val_f1": round(best_val_f1, 6),
        "best_val_acc": round(best_val_metrics.acc, 6) if best_val_metrics is not None else "",
        "best_val_precision_macro": round(best_val_metrics.precision_macro, 6) if best_val_metrics is not None else "",
        "best_val_recall_macro": round(best_val_metrics.recall_macro, 6) if best_val_metrics is not None else "",
        "best_val_balanced_acc": round(best_val_metrics.balanced_acc, 6) if best_val_metrics is not None else "",
        "test_f1": round(te_m.f1_macro, 6),
        "test_acc": round(te_m.acc, 6),
        "test_precision_macro": round(te_m.precision_macro, 6),
        "test_recall_macro": round(te_m.recall_macro, 6),
        "test_balanced_acc": round(te_m.balanced_acc, 6),
        "params": param_count,
        "model_size_mb": round(model_size_mb, 4),
        "cpu_latency_ms": round(lat_ms, 4),
        "wall_time_s": round(wall_time_s, 3),
    }

    append_to_leaderboard(args.out_csv, row)
    print(f"Logged to {args.out_csv}")
    write_json(
        rd / "metrics.json",
        {
            "run_name": run_name,
            "best_val": (best_val_metrics.__dict__ if best_val_metrics is not None else None),
            "test": te_m.__dict__,
            "params": param_count,
            "model_size_mb": model_size_mb,
            "cpu_latency_ms": lat_ms,
            "wall_time_s": wall_time_s,
        },
    )
    print("Done.")


if __name__ == "__main__":
    main()
