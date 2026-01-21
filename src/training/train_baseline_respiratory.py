from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from src.data.dreamcatcher_hf import DreamCatcherHFAudioConfig
from src.data.dreamcatcher_subset import DreamCatcherRespiratorySubset, RESPIRATORY_LABELS

# Use respiratory subset labels (3 classes)
LABELS = RESPIRATORY_LABELS
from src.evaluation.metrics import classification_metrics
from src.models.tinycnn import TinyCNN
from src.models.crnn import CRNN
from src.models.crnn_cbam import CRNN_CBAM
from src.utils.reproducibility import set_seed
from src.utils.benchmarking import (
    count_params,
    estimate_model_size_mb,
    measure_cpu_latency,
    append_to_leaderboard,
)
from src.utils.runlog import StepLogger
from src.utils.artifacts import env_snapshot, run_dir, write_json


def collate_fn(batch):
    """
    batch: list of (feat[n_mels, time], label_id)
    Pads time to max in batch.
    Returns:
        xb: [B, 1, n_mels, T]
        yb: [B]
    """
    xs, ys = zip(*batch)
    max_t = max(x.shape[1] for x in xs)
    n_mels = xs[0].shape[0]

    x_pad = np.zeros((len(xs), 1, n_mels, max_t), dtype=np.float32)
    for i, x in enumerate(xs):
        t = x.shape[1]
        x_pad[i, 0, :, :t] = x

    return torch.from_numpy(x_pad), torch.tensor(ys, dtype=torch.long)


def make_model(model_name: str, n_classes: int, args) -> torch.nn.Module:
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
    raise ValueError("Unknown model. Choose from: tinycnn, crnn, crnn_cbam")


def run_one_epoch(model, dl, opt, loss_fn, device):
    model.train()
    total_loss = 0.0
    all_true, all_pred = [], []

    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)

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
def evaluate(model, dl, loss_fn, device, *, return_preds: bool = False):
    model.eval()
    total_loss = 0.0
    all_true, all_pred = [], []

    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)

        total_loss += float(loss.item()) * xb.size(0)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
        all_pred.extend(preds)
        all_true.extend(yb.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(1, len(dl.dataset))
    m = classification_metrics(all_true, all_pred)
    if return_preds:
        return avg_loss, m, all_true, all_pred
    return avg_loss, m


def main():
    parser = argparse.ArgumentParser()

    # model + training
    parser.add_argument(
        "--model", type=str, default="tinycnn", choices=["tinycnn", "crnn", "crnn_cbam"]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
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

    # audio
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n_mels", type=int, default=64)
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="",
        help="Optional HF datasets cache dir override (use this if disk is full).",
    )
    parser.add_argument(
        "--invalid_audio_policy",
        type=str,
        default="skip",
        choices=["skip", "resample", "zero"],
        help="How to handle rare invalid/empty audio buffers in the dataset.",
    )

    # CRNN
    parser.add_argument("--rnn_hidden", type=int, default=64)
    parser.add_argument("--rnn_layers", type=int, default=1)

    # CBAM
    parser.add_argument("--cbam_reduction", type=int, default=8)
    parser.add_argument("--cbam_sa_kernel", type=int, default=7)
    parser.add_argument(
        "--att_mode",
        type=str,
        default="cbam",
        choices=["cbam", "ca", "sa"],
        help="Attention mode for crnn_cbam: cbam=CA+SA, ca=channel-only, sa=spatial-only.",
    )

    # logging
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--out_csv", type=str, default="results/leaderboard.csv")
    parser.add_argument("--latency_T", type=int, default=400)
    parser.add_argument("--dataset_mode", type=str, default="full", choices=["full", "smoke"])
    parser.add_argument("--steps_csv", type=str, default="results/run_steps.csv")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Optional cap per split for faster smoke runs (0 = no cap).",
    )

    args = parser.parse_args()
    t_run0 = time.time()
    run_started_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    if args.run_name:
        run_name = args.run_name
    else:
        suffix = f"_att{args.att_mode}" if args.model == "crnn_cbam" else ""
        run_name = f"{args.model}{suffix}_seed{args.seed}"
    slog = StepLogger(run_name=run_name, csv_path=args.steps_csv)
    rd = run_dir(run_name)
    # snapshot config for reproducibility (no secrets)
    write_json(rd / "args.json", vars(args) | {"run_name": run_name})
    write_json(rd / "env.json", env_snapshot())

    print(f"\n{'=' * 60}")
    print(f"Starting Training")
    print(f"  Model: {args.model}")
    print(f"  Run Name: {run_name}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Seed: {args.seed}")
    print(f"  Dataset Mode: {args.dataset_mode}")
    print(f"  Steps CSV: {args.steps_csv}")
    print(f"{'=' * 60}\n")

    slog.log(
        "run_start",
        detail=f"model={args.model} epochs={args.epochs} bs={args.batch_size} lr={args.lr} seed={args.seed}",
    )

    set_seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading datasets...")
    t_ds = time.time()
    slog.log("load_datasets_start", detail=f"mode={args.dataset_mode}")
    cfg = DreamCatcherHFAudioConfig(
        sample_rate=args.sr,
        n_mels=args.n_mels,
        invalid_audio_policy=args.invalid_audio_policy,
    )
    train_ds = DreamCatcherRespiratorySubset(
        split="train",
        cfg=cfg,
        dataset_mode=args.dataset_mode,
        run_name=run_name,
        steps_csv=args.steps_csv,
        cache_dir=(args.cache_dir or None),
        max_samples=args.max_samples if args.max_samples else 0,
    )
    val_ds = DreamCatcherRespiratorySubset(
        split="validation",
        cfg=cfg,
        dataset_mode=args.dataset_mode,
        run_name=run_name,
        steps_csv=args.steps_csv,
        cache_dir=(args.cache_dir or None),
        max_samples=args.max_samples if args.max_samples else 0,
    )
    test_ds = DreamCatcherRespiratorySubset(
        split="test",
        cfg=cfg,
        dataset_mode=args.dataset_mode,
        run_name=run_name,
        steps_csv=args.steps_csv,
        cache_dir=(args.cache_dir or None),
        max_samples=args.max_samples if args.max_samples else 0,
    )
    print(f"All datasets loaded: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}\n")
    slog.log(
        "load_datasets_done",
        t0=t_ds,
        detail=f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}",
    )

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = make_model(args.model, n_classes=len(LABELS), args=args).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

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
        t_ep = time.time()
        tr_loss, tr_m = run_one_epoch(model, train_dl, opt, loss_fn, device)
        va_loss, va_m = evaluate(model, val_dl, loss_fn, device)

        improved = va_m.f1_macro > (best_val_f1 + min_delta)
        if improved:
            best_val_f1 = va_m.f1_macro
            best_val_metrics = va_m
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"epoch={epoch} "
            f"train_loss={tr_loss:.4f} train_acc={tr_m.acc * 100:.2f}% train_f1={tr_m.f1_macro * 100:.2f}% | "
            f"val_loss={va_loss:.4f} val_acc={va_m.acc * 100:.2f}% val_f1={va_m.f1_macro * 100:.2f}%"
        )
        slog.log(
            "epoch_done",
            t0=t_ep,
            detail=f"epoch={epoch} train_loss={tr_loss:.4f} train_acc={tr_m.acc:.4f} train_f1={tr_m.f1_macro:.4f} val_loss={va_loss:.4f} val_acc={va_m.acc:.4f} val_f1={va_m.f1_macro:.4f}",
        )

        if patience > 0 and no_improve >= patience:
            msg = (
                f"early_stop at epoch={epoch} best_epoch={best_epoch} best_val_f1={best_val_f1:.6f}"
            )
            print(msg)
            slog.log("early_stop", detail=msg)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    t_test = time.time()
    slog.log("test_eval_start")
    te_loss, te_m, te_true, te_pred = evaluate(model, test_dl, loss_fn, device, return_preds=True)
    print(
        f"test_loss={te_loss:.4f} test_acc={te_m.acc * 100:.2f}% test_f1={te_m.f1_macro * 100:.2f}%"
    )
    slog.log(
        "test_eval_done",
        t0=t_test,
        detail=f"test_loss={te_loss:.4f} test_acc={te_m.acc:.4f} test_f1={te_m.f1_macro:.4f}",
    )

    # Save test confusion matrix as a per-run artifact (CSV).
    cm = confusion_matrix(te_true, te_pred, labels=list(range(len(LABELS))))
    cm_path = rd / "test_confusion_matrix.csv"
    with cm_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred", *LABELS])
        for i, row_vals in enumerate(cm.tolist()):
            w.writerow([LABELS[i], *row_vals])

    param_count = count_params(model)
    model_size_mb = estimate_model_size_mb(model)
    lat_ms = measure_cpu_latency(model, input_shape=(1, 1, args.n_mels, args.latency_T))
    wall_time_s = time.time() - t_run0
    run_finished_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    row = {
        "run_started_at_utc": run_started_at_utc,
        "run_finished_at_utc": run_finished_at_utc,
        "run_name": run_name,
        "task": "respiratory_subset",
        "model": args.model,
        "teacher": "",
        "seed": args.seed,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "epochs_ran": epochs_ran,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "sr": args.sr,
        "n_mels": args.n_mels,
        "rnn_hidden": args.rnn_hidden if args.model != "tinycnn" else "",
        "rnn_layers": args.rnn_layers if args.model != "tinycnn" else "",
        "cbam_reduction": args.cbam_reduction if args.model == "crnn_cbam" else "",
        "cbam_sa_kernel": args.cbam_sa_kernel if args.model == "crnn_cbam" else "",
        "att_mode": args.att_mode if args.model == "crnn_cbam" else "",
        "alpha": "",
        "tau": "",
        "dataset_mode": args.dataset_mode,
        "max_samples": "",
        "invalid_audio_policy": args.invalid_audio_policy,
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
    # write per-run summary (machine-readable)
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
    slog.log("leaderboard_written", detail=f"path={args.out_csv}")
    print("Done.")
    slog.log("run_done")


if __name__ == "__main__":
    main()
