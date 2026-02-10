from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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
    KD_EPOCH_METRICS_FIELDNAMES,
    TEST_METRICS_FIELDNAMES,
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
    if model_name == "crnn":
        return CRNN(n_classes=n_classes, rnn_hidden=args.rnn_hidden, rnn_layers=args.rnn_layers)
    raise ValueError("Unknown model. Choose from: tinycnn, tinycnn_cbam, crnn")


def distillation_loss(
    student_logits,
    teacher_logits,
    labels,
    ce_loss_fn,
    temperature,
    alpha,
):
    """Knowledge distillation loss combining soft targets and hard targets."""
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    kd_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature**2)
    ce_loss = ce_loss_fn(student_logits, labels)
    total_loss = alpha * kd_loss + (1 - alpha) * ce_loss
    return total_loss, kd_loss, ce_loss


def run_one_epoch_kd(student, teacher, dl, opt, class_weights, temperature, alpha, device):
    """Train one epoch with knowledge distillation."""
    student.train()
    teacher.eval()
    ce_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    total_loss = 0.0
    all_true, all_pred = [], []

    pbar = tqdm(dl, desc="Training KD", leave=False)
    for xb, yb in pbar:
        xb, yb = xb.to(device), yb.to(device)

        student_logits = student(xb)
        with torch.no_grad():
            teacher_logits = teacher(xb)

        loss, kd_loss, ce_loss = distillation_loss(
            student_logits,
            teacher_logits,
            yb,
            ce_loss_fn,
            temperature,
            alpha,
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += float(loss.item()) * xb.size(0)
        preds = torch.argmax(student_logits, dim=1).detach().cpu().numpy().tolist()
        all_pred.extend(preds)
        all_true.extend(yb.detach().cpu().numpy().tolist())

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "kd": f"{kd_loss.item():.3f}", "ce": f"{ce_loss.item():.3f}"})

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
        preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        all_pred.extend(preds)
        all_true.extend(yb.cpu().numpy().tolist())

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


def train_with_kd(args, student, teacher, train_dl, val_dl, test_dl, device, logger, run_id, out_dir):
    """Train student model with knowledge distillation from teacher."""
    if args.class_weights:
        weights = torch.tensor([float(x) for x in args.class_weights.split(",")], device=device)
    else:
        weights = None

    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    val_loss_fn = torch.nn.CrossEntropyLoss(weight=weights) if weights is not None else torch.nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_val_metrics = None
    best_epoch = 0
    best_model_state = None
    epochs_ran = 0
    patience_counter = 0
    early_stopped = False
    stopped_epoch = 0
    patience = int(args.early_stop_patience) if args.early_stop_patience else 0
    min_delta = float(args.early_stop_min_delta) if args.early_stop_min_delta else 0.0

    epoch_metrics_csv = out_dir / "epoch_metrics.csv"

    for epoch in range(1, args.epochs + 1):
        epochs_ran = epoch
        epoch_start_utc = utc_now_iso()
        t0 = time.time()

        train_loss, train_m = run_one_epoch_kd(
            student,
            teacher,
            train_dl,
            optimizer,
            weights,
            args.temperature,
            args.alpha,
            device,
        )
        val_loss, val_m = evaluate(student, val_dl, val_loss_fn, device)

        improved = val_m.f1_macro > (best_val_f1 + min_delta)
        if improved:
            best_val_f1 = val_m.f1_macro
            best_val_loss = val_loss
            best_val_metrics = val_m
            best_epoch = epoch
            best_model_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        logger.log(
            "epoch_done",
            t0=t0,
            detail=(
                f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_m.acc:.4f} train_f1={train_m.f1_macro:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_m.acc:.4f} val_f1={val_m.f1_macro:.4f} alpha={args.alpha:.2f} tau={args.temperature:.2f}"
            ),
        )
        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} train_acc={train_m.acc * 100:.2f}% train_f1={train_m.f1_macro * 100:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_m.acc * 100:.2f}% val_f1={val_m.f1_macro * 100:.2f}%"
        )

        epoch_end_utc = utc_now_iso()
        write_csv_row(
            epoch_metrics_csv,
            KD_EPOCH_METRICS_FIELDNAMES,
            {
                "ts_utc": epoch_end_utc,
                "run_id": run_id,
                "epoch": epoch,
                "epoch_start_utc": epoch_start_utc,
                "epoch_end_utc": epoch_end_utc,
                "epoch_s": round(time.time() - t0, 6),
                "train_loss": train_loss,
                "train_acc": train_m.acc,
                "train_f1_macro": train_m.f1_macro,
                "train_precision_macro": train_m.precision_macro,
                "train_recall_macro": train_m.recall_macro,
                "train_balanced_acc": train_m.balanced_acc,
                "val_loss": val_loss,
                "val_acc": val_m.acc,
                "val_f1_macro": val_m.f1_macro,
                "val_precision_macro": val_m.precision_macro,
                "val_recall_macro": val_m.recall_macro,
                "val_balanced_acc": val_m.balanced_acc,
                "lr": optimizer.param_groups[0]["lr"],
                "alpha": args.alpha,
                "tau": args.temperature,
                "is_best": int(improved),
            },
        )

        if patience > 0 and patience_counter >= patience:
            msg = f"early_stop at epoch={epoch} best_epoch={best_epoch} best_val_f1={best_val_f1:.6f}"
            print(msg)
            logger.log("early_stop", detail=msg)
            early_stopped = True
            stopped_epoch = epoch
            break

    if not early_stopped:
        stopped_epoch = epochs_ran

    last_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}

    if best_model_state is not None:
        student.load_state_dict(best_model_state)

    val_loss_eval, val_m_eval, val_true, val_pred = evaluate(
        student,
        val_dl,
        val_loss_fn,
        device,
        return_preds=True,
    )
    test_loss, test_m, y_true, y_pred = evaluate(
        student,
        test_dl,
        val_loss_fn,
        device,
        return_preds=True,
    )

    return {
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "best_val_loss": best_val_loss,
        "best_val_metrics": best_val_metrics,
        "test_loss": test_loss,
        "test_metrics": test_m,
        "y_true": y_true,
        "y_pred": y_pred,
        "val_loss_eval": val_loss_eval,
        "val_metrics_eval": val_m_eval,
        "val_true": val_true,
        "val_pred": val_pred,
        "epochs_ran": epochs_ran,
        "early_stopped": early_stopped,
        "stopped_epoch": stopped_epoch,
        "patience": patience,
        "min_delta": min_delta,
        "last_state": last_state,
        "optimizer_state": optimizer.state_dict(),
        "epoch_metrics_csv": epoch_metrics_csv,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--student_model",
        type=str,
        default="tinycnn",
        choices=[
            "tinycnn",
            "tinycnn_cbam",
            "crnn",
        ],
    )
    parser.add_argument(
        "--teacher_model",
        type=str,
        default="crnn",
        choices=[
            "tinycnn",
            "tinycnn_cbam",
            "crnn",
        ],
    )
    parser.add_argument(
        "--teacher_checkpoint",
        type=str,
        required=True,
        help="Path to teacher checkpoint.",
    )

    parser.add_argument("--temperature", type=float, default=5.0, help="KD temperature.")
    parser.add_argument("--alpha", type=float, default=0.7, help="KD loss weight.")

    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--early_stop_patience", type=int, default=DEFAULT_EARLY_STOP_PATIENCE)
    parser.add_argument("--early_stop_min_delta", type=float, default=DEFAULT_EARLY_STOP_MIN_DELTA)
    parser.add_argument("--class_weights", type=str, default=DEFAULT_CLASS_WEIGHTS)

    parser.add_argument("--rnn_hidden", type=int, default=DEFAULT_RNN_HIDDEN)
    parser.add_argument("--rnn_layers", type=int, default=DEFAULT_RNN_LAYERS)
    parser.add_argument("--att_mode", type=str, default="cbam", choices=["cbam", "ca", "sa"])
    parser.add_argument("--cbam_reduction", type=int, default=DEFAULT_CBAM_REDUCTION)
    parser.add_argument("--cbam_sa_kernel", type=int, default=DEFAULT_CBAM_SA_KERNEL)

    parser.add_argument("--cache_dir", type=str, default="results/cache/spectrograms")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n_mels", type=int, default=64)
    parser.add_argument("--clip_seconds", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--steps_csv", type=str, default="results/run_steps.csv")
    parser.add_argument("--out_csv", type=str, default="results/leaderboard.csv")
    args = parser.parse_args()

    t_run0 = time.time()
    run_started_at_utc = utc_now_iso()
    run_name = args.run_name

    resolved_config = vars(args) | {"run_name": run_name}
    cfg_hash = config_hash(resolved_config)
    run_id = f"{run_name}-s{args.seed}-{cfg_hash}"

    run_path = Path("results/runs") / run_name
    if run_path.exists():
        raise FileExistsError(
            f"Run directory already exists: {run_path}. Use a new --run_name for clean reproducible runs."
        )
    out_dir = run_dir(run_name)
    checkpoints_dir = out_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    slog = StepLogger(run_name=run_name, csv_path=args.steps_csv)

    git_info = git_snapshot()
    write_json(out_dir / "resolved_config.json", resolved_config)
    write_json(out_dir / "env.json", env_snapshot())
    write_json(out_dir / "git.json", git_info)

    set_seed(args.seed)
    torch.manual_seed(args.seed)

    if args.n_mels != 64:
        raise ValueError("This benchmark is pinned to n_mels=64 for reproducible comparisons.")

    slog.log("load_datasets_start", detail=f"cache_dir={args.cache_dir}")
    train_ds = CachedDataset(split="train", cache_dir=args.cache_dir)
    val_ds = CachedDataset(split="validation", cache_dir=args.cache_dir)
    test_ds = CachedDataset(split="test", cache_dir=args.cache_dir)
    data_fingerprint = dataset_fingerprint_from_cached_dataset(train_ds)
    write_json(out_dir / "data_fingerprint.json", data_fingerprint)
    slog.log(
        "load_datasets_done",
        detail=f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}",
    )

    device, device_name = select_device()

    nw = args.num_workers
    pin = device.type == "cuda"
    persist = nw > 0
    dl_kwargs = dict(collate_fn=collate_fn, num_workers=nw, pin_memory=pin, persistent_workers=persist)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **dl_kwargs)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **dl_kwargs)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **dl_kwargs)
    slog.log("device_selected", detail=f"device={device} name={device_name}")
    print(f"Device: {device} ({device_name})")

    n_classes = len(LABELS)
    teacher = make_model(args.teacher_model, n_classes, args).to(device)
    teacher_ckpt = torch.load(args.teacher_checkpoint, map_location=device)
    teacher_state = teacher_ckpt["state_dict"] if isinstance(teacher_ckpt, dict) and "state_dict" in teacher_ckpt else teacher_ckpt
    teacher.load_state_dict(teacher_state)
    teacher.eval()
    teacher_params = count_params(teacher)
    (out_dir / "teacher_architecture.txt").write_text(f"{teacher}\n", encoding="utf-8")

    student = make_model(args.student_model, n_classes, args).to(device)
    student_params = count_params(student)
    (out_dir / "model_architecture.txt").write_text(f"{student}\n", encoding="utf-8")
    compression_ratio = teacher_params / max(1, student_params)
    slog.log(
        "models_initialized",
        detail=(
            f"teacher={args.teacher_model} teacher_params={teacher_params} "
            f"student={args.student_model} student_params={student_params}"
        ),
    )

    print("\n=== Knowledge Distillation Training ===")
    print(f"Teacher: {args.teacher_model} ({teacher_params:,} params)")
    print(f"Student: {args.student_model} ({student_params:,} params)")
    print(f"Temperature: {args.temperature}")
    print(f"Alpha: {args.alpha}")
    print(f"Run: {run_name}")
    print(f"Output: {out_dir}\n")
    slog.log(
        "run_start",
        detail=(
            f"teacher={args.teacher_model} student={args.student_model} "
            f"epochs={args.epochs} bs={args.batch_size} lr={args.lr} alpha={args.alpha} tau={args.temperature}"
        ),
    )

    results = train_with_kd(
        args,
        student,
        teacher,
        train_dl,
        val_dl,
        test_dl,
        device,
        slog,
        run_id,
        out_dir,
    )

    best_ckpt = checkpoints_dir / "best_model.pth"
    last_ckpt = checkpoints_dir / "last_model.pth"
    optimizer_ckpt = checkpoints_dir / "optimizer_last.pth"
    rng_ckpt = checkpoints_dir / "rng_state.pth"
    torch.save(student.state_dict(), best_ckpt)
    torch.save(results["last_state"], last_ckpt)
    torch.save(results["optimizer_state"], optimizer_ckpt)
    torch.save(rng_state_dict(), rng_ckpt)

    cm = confusion_matrix(results["y_true"], results["y_pred"], labels=list(range(len(LABELS))))
    cm_path = out_dir / "test_confusion_matrix.csv"
    with cm_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred", *LABELS])
        for i, row_vals in enumerate(cm.tolist()):
            w.writerow([LABELS[i], *row_vals])

    class_metrics_csv = out_dir / "class_metrics.csv"
    write_per_class_metrics_csv(
        class_metrics_csv,
        "validation",
        results["val_true"],
        results["val_pred"],
    )
    write_per_class_metrics_csv(
        class_metrics_csv,
        "test",
        results["y_true"],
        results["y_pred"],
    )

    test_m = results["test_metrics"]
    test_metrics_csv = out_dir / "test_metrics.csv"
    write_csv_row(
        test_metrics_csv,
        TEST_METRICS_FIELDNAMES,
        {
            "ts_utc": utc_now_iso(),
            "run_id": run_id,
            "test_loss": results["test_loss"],
            "test_acc": test_m.acc,
            "test_f1_macro": test_m.f1_macro,
            "test_precision_macro": test_m.precision_macro,
            "test_recall_macro": test_m.recall_macro,
            "test_balanced_acc": test_m.balanced_acc,
        },
    )

    model_size_mb = estimate_model_size_mb(student)
    sample_spec, _ = train_ds[0]
    input_shape = (1, 1, sample_spec.shape[0], sample_spec.shape[1])
    lat_ms = measure_cpu_latency(student, input_shape=input_shape)
    wall_time_s = time.time() - t_run0
    run_finished_at_utc = utc_now_iso()

    early_stop_payload = {
        "enabled": results["patience"] > 0,
        "patience": results["patience"],
        "min_delta": results["min_delta"],
        "early_stopped": results["early_stopped"],
        "stopped_epoch": results["stopped_epoch"],
        "best_epoch": results["best_epoch"],
        "best_val_f1": results["best_val_f1"],
    }
    write_json(out_dir / "early_stop.json", early_stop_payload)

    best_val_metrics = results["best_val_metrics"]
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
        "model": args.student_model,
        "teacher_model": args.teacher_model,
        "teacher_checkpoint": args.teacher_checkpoint,
        "optimizer": "Adam",
        "scheduler": "",
        "hyperparameters": resolved_config,
        "best_epoch": results["best_epoch"],
        "epochs_ran": results["epochs_ran"],
        "early_stop": early_stop_payload,
        "validation_best": {
            "loss": results["best_val_loss"],
            "acc": best_val_metrics.acc if best_val_metrics is not None else None,
            "balanced_acc": best_val_metrics.balanced_acc if best_val_metrics is not None else None,
            "f1_macro": results["best_val_f1"],
            "precision_macro": best_val_metrics.precision_macro if best_val_metrics is not None else None,
            "recall_macro": best_val_metrics.recall_macro if best_val_metrics is not None else None,
        },
        "validation_current": {
            "loss": results["val_loss_eval"],
            "acc": results["val_metrics_eval"].acc,
            "balanced_acc": results["val_metrics_eval"].balanced_acc,
            "f1_macro": results["val_metrics_eval"].f1_macro,
            "precision_macro": results["val_metrics_eval"].precision_macro,
            "recall_macro": results["val_metrics_eval"].recall_macro,
        },
        "test": {
            "loss": results["test_loss"],
            "acc": test_m.acc,
            "balanced_acc": test_m.balanced_acc,
            "f1_macro": test_m.f1_macro,
            "precision_macro": test_m.precision_macro,
            "recall_macro": test_m.recall_macro,
        },
        "params": student_params,
        "teacher_params": teacher_params,
        "compression_ratio": compression_ratio,
        "model_size_mb": model_size_mb,
        "cpu_latency_ms": lat_ms,
        "data_fingerprint": data_fingerprint,
        "artifacts": {
            "teacher_architecture_txt": str(out_dir / "teacher_architecture.txt"),
            "model_architecture_txt": str(out_dir / "model_architecture.txt"),
            "epoch_metrics_csv": str(results["epoch_metrics_csv"]),
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
    write_json(out_dir / "metrics.json", metrics_payload)

    leaderboard_row = {
        "schema_version": "v2",
        "run_id": run_id,
        "run_name": run_name,
        "task_name": "sleep_event_classification",
        "model": args.student_model,
        "teacher_model": args.teacher_model,
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
        "epochs_ran": results["epochs_ran"],
        "early_stop_patience": results["patience"],
        "early_stop_min_delta": results["min_delta"],
        "early_stopped": int(results["early_stopped"]),
        "stopped_epoch": results["stopped_epoch"],
        "best_epoch": results["best_epoch"],
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "optimizer": "Adam",
        "scheduler": "",
        "sr": args.sr,
        "n_mels": args.n_mels,
        "clip_seconds": args.clip_seconds,
        "class_weights": args.class_weights,
        "dataset_commit": data_fingerprint.get("dataset_commit", ""),
        "cache_fingerprint": config_hash(data_fingerprint) if data_fingerprint else "",
        "best_val_loss": round(results["best_val_loss"], 6),
        "best_val_f1": round(results["best_val_f1"], 6),
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
        "test_loss": round(results["test_loss"], 6),
        "test_f1": round(test_m.f1_macro, 6),
        "test_acc": round(test_m.acc, 6),
        "test_precision_macro": round(test_m.precision_macro, 6),
        "test_recall_macro": round(test_m.recall_macro, 6),
        "test_balanced_acc": round(test_m.balanced_acc, 6),
        "params": student_params,
        "teacher_params": teacher_params,
        "compression_ratio": round(compression_ratio, 6),
        "model_size_mb": round(model_size_mb, 6),
        "cpu_latency_ms": round(lat_ms, 6),
        "artifact_dir": str(out_dir),
        "metrics_json": str(out_dir / "metrics.json"),
        "epoch_metrics_csv": str(results["epoch_metrics_csv"]),
        "test_metrics_csv": str(test_metrics_csv),
        "class_metrics_csv": str(class_metrics_csv),
        "test_cm_csv": str(cm_path),
        "best_checkpoint": str(best_ckpt),
        "last_checkpoint": str(last_ckpt),
        "optimizer_checkpoint": str(optimizer_ckpt),
        "scheduler_checkpoint": "",
        "rng_checkpoint": str(rng_ckpt),
        "rnn_hidden": args.rnn_hidden if ("crnn" in args.student_model or "crnn" in args.teacher_model) else "",
        "rnn_layers": args.rnn_layers if ("crnn" in args.student_model or "crnn" in args.teacher_model) else "",
        "cbam_reduction": (
            args.cbam_reduction if ("_cbam" in args.student_model or "_cbam" in args.teacher_model) else ""
        ),
        "cbam_sa_kernel": (
            args.cbam_sa_kernel if ("_cbam" in args.student_model or "_cbam" in args.teacher_model) else ""
        ),
        "att_mode": args.att_mode if ("_cbam" in args.student_model or "_cbam" in args.teacher_model) else "",
        "alpha": args.alpha,
        "tau": args.temperature,
    }
    enforce_artifact_contract(out_dir, out_dir / "metrics.json", leaderboard_row)
    append_to_leaderboard(args.out_csv, leaderboard_row)

    print("\n=== Final Results (KD) ===")
    print(f"Test Accuracy: {test_m.acc:.4f}")
    print(f"Test F1-Macro: {test_m.f1_macro:.4f}")
    print(f"Test Balanced Acc: {test_m.balanced_acc:.4f}")
    print(f"Student params: {student_params:,}")
    print(f"Teacher params: {teacher_params:,}")
    print(f"Compression: {compression_ratio:.2f}x")
    print(f"Model size: {model_size_mb:.3f} MB")
    print(f"CPU latency: {lat_ms:.3f} ms")
    print(f"Results saved to: {out_dir}")
    print(f"Leaderboard updated: {args.out_csv}")

    slog.log("leaderboard_written", detail=f"path={args.out_csv}")
    slog.log("run_done")


if __name__ == "__main__":
    main()
