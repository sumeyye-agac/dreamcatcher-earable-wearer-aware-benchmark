from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.cached_dataset import CachedDataset
from src.data.dreamcatcher_dataset import CLASS_LABELS as LABELS
from src.evaluation.metrics import classification_metrics, per_class_metrics
from src.models.crnn import CRNN
from src.models.tinycnn import TinyCNN
from src.training.defaults import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CBAM_REDUCTION,
    DEFAULT_CBAM_SA_KERNEL,
    DEFAULT_CLASS_WEIGHTS,
    DEFAULT_EARLY_STOP_MIN_DELTA,
    DEFAULT_EARLY_STOP_PATIENCE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_NUM_WORKERS,
    DEFAULT_RNN_HIDDEN,
    DEFAULT_RNN_LAYERS,
    DEFAULT_SEED,
    DEFAULT_WEIGHT_DECAY,
)
from src.utils.artifact_contract import ARTIFACT_CONTRACT_VERSION, enforce_artifact_contract
from src.utils.artifacts import env_snapshot, run_dir, write_json
from src.utils.benchmarking import (
    append_to_leaderboard,
    count_params,
    estimate_model_size_mb,
    measure_cpu_latency,
)
from src.utils.csv_schemas import (
    CLASS_METRICS_FIELDNAMES,
    TEST_METRICS_FIELDNAMES,
    TRAIN_EPOCH_METRICS_FIELDNAMES,
)
from src.utils.experiment_tracking import (
    config_hash,
    dataset_fingerprint_from_cached_dataset,
    git_snapshot,
    rng_state_dict,
    select_device,
    utc_now_iso,
    write_csv_row,
)
from src.utils.reproducibility import set_seed
from src.utils.runlog import StepLogger


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
    if model_name == "tinycnn_cbam":
        from src.models.tinycnn_cbam import TinyCNN_CBAM
        use_ca = args.att_mode in ("cbam", "ca")
        use_sa = args.att_mode in ("cbam", "sa")
        return TinyCNN_CBAM(
            n_classes=n_classes,
            cbam_reduction=args.cbam_reduction,
            cbam_sa_kernel=args.cbam_sa_kernel,
            use_ca=use_ca,
            use_sa=use_sa,
        )
    if model_name == "extremetinycnn":
        from src.models.extreme_tinycnn import ExtremeTinyCNN
        return ExtremeTinyCNN(n_classes=n_classes)
    if model_name == "extremetinycnn_cbam":
        from src.models.extreme_tinycnn_cbam import ExtremeTinyCNN_CBAM
        use_ca = args.att_mode in ("cbam", "ca")
        use_sa = args.att_mode in ("cbam", "sa")
        return ExtremeTinyCNN_CBAM(
            n_classes=n_classes,
            cbam_reduction=args.cbam_reduction,
            cbam_sa_kernel=args.cbam_sa_kernel,
            use_ca=use_ca,
            use_sa=use_sa,
        )
    if model_name == "crnn":
        return CRNN(n_classes=n_classes, rnn_hidden=args.rnn_hidden, rnn_layers=args.rnn_layers)
    raise ValueError(
        "Unknown model. Choose from: tinycnn, tinycnn_cbam, extremetinycnn, "
        "extremetinycnn_cbam, crnn"
    )


def run_one_epoch(model, dl, opt, loss_fn, device, grad_clip=0.0):
    model.train()
    total_loss = 0.0
    all_true, all_pred = [], []

    pbar = tqdm(dl, desc="Training", leave=False)
    for xb, yb in pbar:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)

        opt.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        total_loss += float(loss.item()) * xb.size(0)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
        all_pred.extend(preds)
        all_true.extend(yb.detach().cpu().numpy().tolist())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / max(1, len(dl.dataset))
    m = classification_metrics(all_true, all_pred)
    return avg_loss, m


@torch.no_grad()
def evaluate(model, dl, loss_fn, device, *, return_preds: bool = False):
    model.eval()
    total_loss = 0.0
    all_true, all_pred = [], []

    pbar = tqdm(dl, desc="Validating", leave=False)
    for xb, yb in pbar:
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


def write_per_class_metrics_csv(path: Path, split: str, y_true, y_pred) -> None:
    ts_utc = utc_now_iso()
    for row in per_class_metrics(y_true, y_pred, LABELS):
        write_csv_row(
            path,
            CLASS_METRICS_FIELDNAMES,
            {
                "ts_utc": ts_utc,
                "split": split,
                **row,
            },
        )


def main():
    parser = argparse.ArgumentParser()

    # model + training
    parser.add_argument(
        "--model",
        type=str,
        default="tinycnn",
        choices=[
            "tinycnn", "tinycnn_cbam",
            "extremetinycnn", "extremetinycnn_cbam",
            "crnn"
        ]
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--grad_clip", type=float, default=0.0, help="Gradient clipping value (0 disables)")
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=DEFAULT_EARLY_STOP_PATIENCE,
        help="Early stopping patience on val_f1 (0 disables early stopping).",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=DEFAULT_EARLY_STOP_MIN_DELTA,
        help="Minimum val_f1 improvement to reset patience.",
    )

    # audio (NOTE: sr, n_mels, invalid_audio_policy are not used when training from HDF5 cache)
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate (for leaderboard metadata only)")
    parser.add_argument("--n_mels", type=int, default=64, help="Mel bands (for leaderboard metadata only, must match cache)")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="",
        help="Optional HDF5 cache directory override (default: results/cache/spectrograms).",
    )
    parser.add_argument(
        "--invalid_audio_policy",
        type=str,
        default="skip",
        choices=["skip", "resample", "zero"],
        help="How to handle invalid audio (for leaderboard metadata only - not used with HDF5 cache).",
    )

    # CRNN
    parser.add_argument("--rnn_hidden", type=int, default=DEFAULT_RNN_HIDDEN)
    parser.add_argument("--rnn_layers", type=int, default=DEFAULT_RNN_LAYERS)

    # CBAM
    parser.add_argument("--cbam_reduction", type=int, default=DEFAULT_CBAM_REDUCTION)
    parser.add_argument("--cbam_sa_kernel", type=int, default=DEFAULT_CBAM_SA_KERNEL)
    parser.add_argument(
        "--att_mode",
        type=str,
        default="cbam",
        choices=["cbam", "ca", "sa"],
        help="Attention mode for CBAM models: cbam=CA+SA, ca=channel-only, sa=spatial-only.",
    )

    # logging
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--out_csv", type=str, default="results/leaderboard.csv")
    parser.add_argument("--latency_T", type=int, default=400, help="Time steps for latency benchmark (deprecated - now auto-detected from cache)")
    parser.add_argument("--dataset_mode", type=str, default="full", choices=["full", "smoke"], help="Dataset mode (for leaderboard metadata only - not used with HDF5 cache)")
    parser.add_argument("--steps_csv", type=str, default="results/run_steps.csv")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Max samples per split (for leaderboard metadata only - not used with HDF5 cache).",
    )
    parser.add_argument(
        "--class_weights",
        type=str,
        default=DEFAULT_CLASS_WEIGHTS,
        help="Comma-separated class weights (e.g. 1.0,1.5,5.5)",
    )

    args = parser.parse_args()
    t_run0 = time.time()
    run_started_at_utc = utc_now_iso()

    if args.run_name:
        run_name = args.run_name
    else:
        run_name = f"{args.model}_seed{args.seed}"

    resolved_config = vars(args) | {"run_name": run_name}
    cfg_hash = config_hash(resolved_config)
    run_id = f"{run_name}-s{args.seed}-{cfg_hash}"

    slog = StepLogger(run_name=run_name, csv_path=args.steps_csv)
    run_path = Path("results/runs") / run_name
    if run_path.exists():
        raise FileExistsError(
            f"Run directory already exists: {run_path}. Use a new --run_name for clean reproducible runs."
        )
    rd = run_dir(run_name)
    checkpoints_dir = rd / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot reproducibility metadata
    git_info = git_snapshot()
    write_json(rd / "resolved_config.json", resolved_config)
    write_json(rd / "env.json", env_snapshot())
    write_json(rd / "git.json", git_info)

    print(f"\n{'=' * 60}")
    print("Starting Training")
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

    # Canonical benchmark feature setup for cross-run consistency.
    if args.n_mels != 64:
        raise ValueError("This benchmark is pinned to n_mels=64 for reproducible comparisons.")

    print("Loading cached datasets...")
    t_ds = time.time()
    slog.log("load_datasets_start", detail=f"mode={args.dataset_mode}")
    cache_dir = args.cache_dir if args.cache_dir else "results/cache/spectrograms"
    train_ds = CachedDataset(split="train", cache_dir=cache_dir)
    val_ds = CachedDataset(split="validation", cache_dir=cache_dir)
    test_ds = CachedDataset(split="test", cache_dir=cache_dir)
    data_fingerprint = dataset_fingerprint_from_cached_dataset(train_ds)
    write_json(rd / "data_fingerprint.json", data_fingerprint)
    print(f"All datasets loaded: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}\n")
    slog.log(
        "load_datasets_done",
        t0=t_ds,
        detail=f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}",
    )

    # Device selection policy: CUDA -> MPS -> CPU
    device, device_name = select_device()

    nw = args.num_workers
    pin = device.type == "cuda"
    persist = nw > 0
    dl_kwargs = dict(collate_fn=collate_fn, num_workers=nw, pin_memory=pin, persistent_workers=persist)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **dl_kwargs)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **dl_kwargs)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **dl_kwargs)

    print("=" * 60)
    print(f"DEVICE: {device} ({device_name})")
    print("=" * 60)

    slog.log("device_selected", detail=f"device={device} name={device_name}")

    model = make_model(args.model, n_classes=len(LABELS), args=args).to(device)
    param_count = count_params(model)
    (rd / "model_architecture.txt").write_text(f"{model}\n", encoding="utf-8")
    slog.log("model_initialized", detail=f"model={args.model} params={param_count}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Weighted loss for class imbalance (quiet=1.0, breathe=1.5, snore=5.5)
    if hasattr(args, 'class_weights') and args.class_weights:
        weights = torch.tensor([float(w) for w in args.class_weights.split(',')]).to(device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        print(f"Using weighted loss: {weights.tolist()}")
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_val_metrics = None
    best_state = None
    best_epoch = 0
    epochs_ran = 0
    no_improve = 0
    stopped_epoch = 0
    early_stopped = False
    patience = int(args.early_stop_patience) if args.early_stop_patience else 0
    min_delta = float(args.early_stop_min_delta) if args.early_stop_min_delta else 0.0
    epoch_metrics_csv = rd / "epoch_metrics.csv"

    for epoch in range(1, args.epochs + 1):
        epochs_ran = epoch
        epoch_start_utc = utc_now_iso()
        t_ep = time.time()
        tr_loss, tr_m = run_one_epoch(model, train_dl, opt, loss_fn, device, args.grad_clip)
        va_loss, va_m = evaluate(model, val_dl, loss_fn, device)

        improved = va_m.f1_macro > (best_val_f1 + min_delta)
        if improved:
            best_val_f1 = va_m.f1_macro
            best_val_loss = va_loss
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
        epoch_end_utc = utc_now_iso()
        write_csv_row(
            epoch_metrics_csv,
            TRAIN_EPOCH_METRICS_FIELDNAMES,
            {
                "ts_utc": epoch_end_utc,
                "run_id": run_id,
                "epoch": epoch,
                "epoch_start_utc": epoch_start_utc,
                "epoch_end_utc": epoch_end_utc,
                "epoch_s": round(time.time() - t_ep, 6),
                "train_loss": tr_loss,
                "train_acc": tr_m.acc,
                "train_f1_macro": tr_m.f1_macro,
                "train_precision_macro": tr_m.precision_macro,
                "train_recall_macro": tr_m.recall_macro,
                "train_balanced_acc": tr_m.balanced_acc,
                "val_loss": va_loss,
                "val_acc": va_m.acc,
                "val_f1_macro": va_m.f1_macro,
                "val_precision_macro": va_m.precision_macro,
                "val_recall_macro": va_m.recall_macro,
                "val_balanced_acc": va_m.balanced_acc,
                "lr": opt.param_groups[0]["lr"],
                "is_best": int(improved),
            },
        )

        if patience > 0 and no_improve >= patience:
            msg = (
                f"early_stop at epoch={epoch} best_epoch={best_epoch} best_val_f1={best_val_f1:.6f}"
            )
            print(msg)
            slog.log("early_stop", detail=msg)
            early_stopped = True
            stopped_epoch = epoch
            break

    if not early_stopped:
        stopped_epoch = epochs_ran

    # Keep final-epoch weights for resumable checkpoints before restoring best model.
    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

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

    # Save reproducible checkpoint set
    best_ckpt = checkpoints_dir / "best_model.pth"
    last_ckpt = checkpoints_dir / "last_model.pth"
    optimizer_ckpt = checkpoints_dir / "optimizer_last.pth"
    rng_ckpt = checkpoints_dir / "rng_state.pth"
    torch.save(model.state_dict(), best_ckpt)
    torch.save(last_state, last_ckpt)
    torch.save(opt.state_dict(), optimizer_ckpt)
    torch.save(rng_state_dict(), rng_ckpt)
    print(f"Saved checkpoints under: {checkpoints_dir}")

    # Save confusion matrix and detailed class metrics
    cm = confusion_matrix(te_true, te_pred, labels=list(range(len(LABELS))))
    cm_path = rd / "test_confusion_matrix.csv"
    with cm_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred", *LABELS])
        for i, row_vals in enumerate(cm.tolist()):
            w.writerow([LABELS[i], *row_vals])

    class_metrics_csv = rd / "class_metrics.csv"
    val_loss_eval, val_m_eval, val_true, val_pred = evaluate(
        model, val_dl, loss_fn, device, return_preds=True
    )
    write_per_class_metrics_csv(class_metrics_csv, "validation", val_true, val_pred)
    write_per_class_metrics_csv(class_metrics_csv, "test", te_true, te_pred)

    test_metrics_csv = rd / "test_metrics.csv"
    write_csv_row(
        test_metrics_csv,
        TEST_METRICS_FIELDNAMES,
        {
            "ts_utc": utc_now_iso(),
            "run_id": run_id,
            "test_loss": te_loss,
            "test_acc": te_m.acc,
            "test_f1_macro": te_m.f1_macro,
            "test_precision_macro": te_m.precision_macro,
            "test_recall_macro": te_m.recall_macro,
            "test_balanced_acc": te_m.balanced_acc,
        },
    )

    model_size_mb = estimate_model_size_mb(model)
    sample_spec, _ = train_ds[0]
    input_shape = (1, 1, sample_spec.shape[0], sample_spec.shape[1])
    lat_ms = measure_cpu_latency(model, input_shape=input_shape)
    wall_time_s = time.time() - t_run0
    run_finished_at_utc = utc_now_iso()

    early_stop_payload = {
        "enabled": patience > 0,
        "patience": patience,
        "min_delta": min_delta,
        "early_stopped": early_stopped,
        "stopped_epoch": stopped_epoch,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
    }
    write_json(rd / "early_stop.json", early_stop_payload)

    metrics_payload = {
        "schema_version": "v2",
        "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
        "run_id": run_id,
        "run_name": run_name,
        "task_name": "sleep_event_classification",
        "ts_start_utc": run_started_at_utc,
        "ts_end_utc": run_finished_at_utc,
        "duration_s": wall_time_s,
        "seed": args.seed,
        "config_hash": cfg_hash,
        "device": str(device),
        "device_name": device_name,
        "model": args.model,
        "teacher_model": "",
        "optimizer": "Adam",
        "scheduler": "",
        "hyperparameters": resolved_config,
        "best_epoch": best_epoch,
        "epochs_ran": epochs_ran,
        "early_stop": early_stop_payload,
        "validation_best": {
            "loss": best_val_loss,
            "acc": best_val_metrics.acc if best_val_metrics is not None else None,
            "balanced_acc": best_val_metrics.balanced_acc if best_val_metrics is not None else None,
            "f1_macro": best_val_f1,
            "precision_macro": (
                best_val_metrics.precision_macro if best_val_metrics is not None else None
            ),
            "recall_macro": best_val_metrics.recall_macro if best_val_metrics is not None else None,
        },
        "validation_current": {
            "loss": val_loss_eval,
            "acc": val_m_eval.acc,
            "balanced_acc": val_m_eval.balanced_acc,
            "f1_macro": val_m_eval.f1_macro,
            "precision_macro": val_m_eval.precision_macro,
            "recall_macro": val_m_eval.recall_macro,
        },
        "test": {
            "loss": te_loss,
            "acc": te_m.acc,
            "balanced_acc": te_m.balanced_acc,
            "f1_macro": te_m.f1_macro,
            "precision_macro": te_m.precision_macro,
            "recall_macro": te_m.recall_macro,
        },
        "params": param_count,
        "model_size_mb": model_size_mb,
        "cpu_latency_ms": lat_ms,
        "data_fingerprint": data_fingerprint,
        "artifacts": {
            "model_architecture_txt": str(rd / "model_architecture.txt"),
            "epoch_metrics_csv": str(epoch_metrics_csv),
            "test_metrics_csv": str(test_metrics_csv),
            "class_metrics_csv": str(class_metrics_csv),
            "test_confusion_matrix_csv": str(cm_path),
            "best_checkpoint": str(best_ckpt),
            "last_checkpoint": str(last_ckpt),
            "optimizer_checkpoint": str(optimizer_ckpt),
            "scheduler_checkpoint": "",
            "rng_checkpoint": str(rng_ckpt),
        },
    }
    write_json(rd / "metrics.json", metrics_payload)

    leaderboard_row = {
        "schema_version": "v2",
        "run_id": run_id,
        "run_name": run_name,
        "task_name": "sleep_event_classification",
        "model": args.model,
        "teacher_model": "",
        "ts_start_utc": run_started_at_utc,
        "ts_end_utc": run_finished_at_utc,
        "duration_s": round(wall_time_s, 6),
        "seed": args.seed,
        "config_hash": cfg_hash,
        "git_commit_sha": git_info.get("commit_sha"),
        "git_dirty": git_info.get("dirty"),
        "device": str(device),
        "device_name": device_name,
        "epochs": args.epochs,
        "epochs_ran": epochs_ran,
        "early_stop_patience": patience,
        "early_stop_min_delta": min_delta,
        "early_stopped": int(early_stopped),
        "stopped_epoch": stopped_epoch,
        "best_epoch": best_epoch,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "optimizer": "Adam",
        "scheduler": "",
        "sr": args.sr,
        "n_mels": args.n_mels,
        "clip_seconds": 5.0,
        "class_weights": args.class_weights,
        "dataset_commit": data_fingerprint.get("dataset_commit", ""),
        "cache_fingerprint": config_hash(data_fingerprint) if data_fingerprint else "",
        "best_val_loss": round(best_val_loss, 6),
        "best_val_f1": round(best_val_f1, 6),
        "best_val_acc": round(best_val_metrics.acc, 6) if best_val_metrics is not None else "",
        "best_val_precision_macro": (
            round(best_val_metrics.precision_macro, 6) if best_val_metrics is not None else ""
        ),
        "best_val_recall_macro": (
            round(best_val_metrics.recall_macro, 6) if best_val_metrics is not None else ""
        ),
        "best_val_balanced_acc": (
            round(best_val_metrics.balanced_acc, 6) if best_val_metrics is not None else ""
        ),
        "test_loss": round(te_loss, 6),
        "test_f1": round(te_m.f1_macro, 6),
        "test_acc": round(te_m.acc, 6),
        "test_precision_macro": round(te_m.precision_macro, 6),
        "test_recall_macro": round(te_m.recall_macro, 6),
        "test_balanced_acc": round(te_m.balanced_acc, 6),
        "params": param_count,
        "teacher_params": "",
        "compression_ratio": "",
        "model_size_mb": round(model_size_mb, 6),
        "cpu_latency_ms": round(lat_ms, 6),
        "artifact_dir": str(rd),
        "metrics_json": str(rd / "metrics.json"),
        "epoch_metrics_csv": str(epoch_metrics_csv),
        "test_metrics_csv": str(test_metrics_csv),
        "class_metrics_csv": str(class_metrics_csv),
        "test_cm_csv": str(cm_path),
        "best_checkpoint": str(best_ckpt),
        "last_checkpoint": str(last_ckpt),
        "optimizer_checkpoint": str(optimizer_ckpt),
        "scheduler_checkpoint": "",
        "rng_checkpoint": str(rng_ckpt),
        "rnn_hidden": args.rnn_hidden if args.model == "crnn" else "",
        "rnn_layers": args.rnn_layers if args.model == "crnn" else "",
        "cbam_reduction": args.cbam_reduction if "_cbam" in args.model else "",
        "cbam_sa_kernel": args.cbam_sa_kernel if "_cbam" in args.model else "",
        "att_mode": args.att_mode if "_cbam" in args.model else "",
        "alpha": "",
        "tau": "",
    }
    enforce_artifact_contract(rd, rd / "metrics.json", leaderboard_row)
    append_to_leaderboard(args.out_csv, leaderboard_row)
    print(f"Logged to {args.out_csv}")

    slog.log("leaderboard_written", detail=f"path={args.out_csv}")
    print("Done.")
    slog.log("run_done")


if __name__ == "__main__":
    main()
