from __future__ import annotations

import argparse
import csv
import time
from datetime import UTC, datetime

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.cached_dataset import CachedDataset
from src.data.dreamcatcher_dataset import CLASS_LABELS as LABELS
from src.evaluation.metrics import classification_metrics
from src.models.crnn import CRNN
from src.models.crnn_cbam import CRNN_CBAM
from src.models.tinycnn import TinyCNN
from src.utils.artifacts import env_snapshot, run_dir, write_json
from src.utils.benchmarking import (
    append_to_leaderboard,
    count_params,
    estimate_model_size_mb,
    measure_cpu_latency,
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
    if model_name == "extremetinycnn":
        from src.models.extreme_tinycnn import ExtremeTinyCNN
        return ExtremeTinyCNN(n_classes=n_classes)
    if model_name == "ultraextremetinycnn":
        from src.models.ultra_extreme_tinycnn import UltraExtremeTinyCNN
        return UltraExtremeTinyCNN(n_classes=n_classes)
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
    raise ValueError("Unknown model. Choose from: tinycnn, extremetinycnn, ultraextremetinycnn, crnn, crnn_cbam")


def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    """
    Knowledge distillation loss combining soft targets and hard targets.

    Args:
        student_logits: [B, C] raw logits from student
        teacher_logits: [B, C] raw logits from teacher
        labels: [B] ground truth labels
        temperature: temperature for softening distributions
        alpha: weight for distillation loss vs hard loss

    Returns:
        Combined loss: alpha * KD_loss + (1-alpha) * CE_loss
    """
    # Soft targets from teacher
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)

    # KL divergence loss (scaled by temperature^2)
    kd_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)

    # Hard target cross-entropy loss
    ce_loss = F.cross_entropy(student_logits, labels)

    # Combined loss
    return alpha * kd_loss + (1 - alpha) * ce_loss


def run_one_epoch_kd(student, teacher, dl, opt, class_weights, temperature, alpha, device):
    """Train one epoch with knowledge distillation."""
    student.train()
    teacher.eval()

    total_loss = 0.0
    all_true, all_pred = [], []

    pbar = tqdm(dl, desc="Training KD", leave=False)
    for xb, yb in pbar:
        xb, yb = xb.to(device), yb.to(device)

        # Get student predictions
        student_logits = student(xb)

        # Get teacher predictions (no grad)
        with torch.no_grad():
            teacher_logits = teacher(xb)

        # Apply class weights to hard CE loss
        ce_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        ce_loss = ce_loss_fn(student_logits, yb)

        # Soft KD loss
        soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
        soft_student = F.log_softmax(student_logits / temperature, dim=1)
        kd_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)

        # Combined loss
        loss = alpha * kd_loss + (1 - alpha) * ce_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += float(loss.item()) * xb.size(0)
        preds = torch.argmax(student_logits, dim=1).detach().cpu().numpy().tolist()
        all_pred.extend(preds)
        all_true.extend(yb.detach().cpu().numpy().tolist())

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "kd": f"{kd_loss.item():.3f}",
            "ce": f"{ce_loss.item():.3f}"
        })

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


def train_with_kd(args, student, teacher, train_dl, val_dl, test_dl, device, logger):
    """Train student model with knowledge distillation from teacher."""

    # Class weights
    if args.class_weights:
        weights = torch.tensor([float(x) for x in args.class_weights.split(",")], device=device)
    else:
        weights = None

    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Loss function for evaluation (standard CE)
    if weights is not None:
        val_loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    else:
        val_loss_fn = torch.nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train with KD
        train_loss, train_m = run_one_epoch_kd(
            student, teacher, train_dl, optimizer,
            weights, args.temperature, args.alpha, device
        )

        # Validate
        val_loss, val_m = evaluate(student, val_dl, val_loss_fn, device)

        elapsed = time.time() - t0

        logger.log_step(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_m["acc"],
            train_f1=train_m["f1_macro"],
            val_loss=val_loss,
            val_acc=val_m["acc"],
            val_f1=val_m["f1_macro"],
            time_s=elapsed,
        )

        # Early stopping check
        if val_m["f1_macro"] > best_val_f1:
            best_val_f1 = val_m["f1_macro"]
            best_epoch = epoch
            best_model_state = student.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.early_stop_patience:
            print(f"\n[Early stop] No improvement for {args.early_stop_patience} epochs.")
            break

    # Load best model
    if best_model_state is not None:
        student.load_state_dict(best_model_state)

    # Final test evaluation
    test_loss, test_m, y_true, y_pred = evaluate(
        student, test_dl, val_loss_fn, device, return_preds=True
    )

    return {
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "test_metrics": test_m,
        "y_true": y_true,
        "y_pred": y_pred,
        "epochs_ran": epoch,
    }


def main():
    parser = argparse.ArgumentParser()

    # Student model
    parser.add_argument("--student_model", type=str, default="tinycnn",
                        help="Student model (tinycnn, extremetinycnn, ultraextremetinycnn, crnn)")

    # Teacher model
    parser.add_argument("--teacher_model", type=str, default="crnn_cbam",
                        help="Teacher model (crnn_cbam, crnn)")
    parser.add_argument("--teacher_checkpoint", type=str, required=True,
                        help="Path to teacher checkpoint (e.g., results/runs/crnn_cbam/best_model.pth)")

    # KD hyperparameters
    parser.add_argument("--temperature", type=float, default=5.0,
                        help="Temperature for softening distributions (default: 5.0)")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Weight for KD loss vs hard CE loss (default: 0.7)")

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--class_weights", type=str, default="1.0,1.5,5.5",
                        help="Class weights (comma-separated)")

    # Model args (for teacher)
    parser.add_argument("--rnn_hidden", type=int, default=64)
    parser.add_argument("--rnn_layers", type=int, default=2)
    parser.add_argument("--att_mode", type=str, default="cbam")
    parser.add_argument("--cbam_reduction", type=int, default=16)
    parser.add_argument("--cbam_sa_kernel", type=int, default=7)

    # Data
    parser.add_argument("--cache_dir", type=str, default="results/cache/spectrograms")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, required=True)

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load datasets
    print("Loading datasets...")
    train_ds = CachedDataset(f"{args.cache_dir}/train.h5")
    val_ds = CachedDataset(f"{args.cache_dir}/val.h5")
    test_ds = CachedDataset(f"{args.cache_dir}/test.h5")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                         num_workers=4, collate_fn=collate_fn, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=4, collate_fn=collate_fn, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate_fn, pin_memory=True)

    n_classes = len(LABELS)
    print(f"Classes: {LABELS}")

    # Create teacher model
    print(f"\nLoading teacher model: {args.teacher_model}")
    teacher = make_model(args.teacher_model, n_classes, args).to(device)
    teacher_ckpt = torch.load(args.teacher_checkpoint, map_location=device)
    teacher.load_state_dict(teacher_ckpt)
    teacher.eval()
    teacher_params = count_params(teacher)
    print(f"Teacher parameters: {teacher_params:,}")

    # Create student model
    print(f"\nCreating student model: {args.student_model}")
    student = make_model(args.student_model, n_classes, args).to(device)
    student_params = count_params(student)
    print(f"Student parameters: {student_params:,}")
    print(f"Compression ratio: {teacher_params / student_params:.2f}x")

    # Setup output directory
    out_dir = run_dir(args.run_name)
    logger = StepLogger(str(out_dir / "train.log"))

    print("\n=== Knowledge Distillation Training ===")
    print(f"Teacher: {args.teacher_model} ({teacher_params:,} params)")
    print(f"Student: {args.student_model} ({student_params:,} params)")
    print(f"Temperature: {args.temperature}")
    print(f"Alpha (KD weight): {args.alpha}")
    print(f"Run: {args.run_name}")
    print(f"Output: {out_dir}\n")

    start_time = datetime.now(UTC)

    # Train with KD
    results = train_with_kd(
        args, student, teacher, train_dl, val_dl, test_dl, device, logger
    )

    end_time = datetime.now(UTC)
    wall_time = (end_time - start_time).total_seconds()

    # Save best model
    torch.save(student.state_dict(), out_dir / "best_model.pth")

    # Confusion matrix
    cm = confusion_matrix(results["y_true"], results["y_pred"])
    cm_path = out_dir / "test_confusion_matrix.csv"
    with open(cm_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + LABELS)
        for i, label in enumerate(LABELS):
            w.writerow([label] + cm[i].tolist())

    # Benchmark metrics
    model_size = estimate_model_size_mb(student)
    latency = measure_cpu_latency(student, (1, 1, 128, 501), device=device)

    # Save metrics
    test_m = results["test_metrics"]
    metrics = {
        "run_name": args.run_name,
        "model": args.student_model,
        "teacher_model": args.teacher_model,
        "kd_temperature": args.temperature,
        "kd_alpha": args.alpha,
        "params": student_params,
        "teacher_params": teacher_params,
        "compression_ratio": round(teacher_params / student_params, 2),
        "model_size_mb": model_size,
        "cpu_latency_ms": latency,
        "epochs_ran": results["epochs_ran"],
        "best_epoch": results["best_epoch"],
        "best_val_f1": results["best_val_f1"],
        "test": {
            "acc": test_m["acc"],
            "balanced_acc": test_m["balanced_acc"],
            "f1_macro": test_m["f1_macro"],
            "precision_macro": test_m["precision_macro"],
            "recall_macro": test_m["recall_macro"],
        },
        "test_confusion_matrix_csv": str(cm_path),
        "task": "sleep_event_classification",
        "run_started_at_utc": start_time.isoformat(),
        "run_finished_at_utc": end_time.isoformat(),
        "wall_time_s": wall_time,
    }

    write_json(out_dir / "metrics.json", metrics)
    env_snapshot(out_dir / "environment.json")

    # Append to leaderboard
    append_to_leaderboard(metrics)

    print("\n=== Final Results (KD) ===")
    print(f"Test Accuracy: {test_m['acc']:.4f}")
    print(f"Test F1-Macro: {test_m['f1_macro']:.4f}")
    print(f"Test Balanced Acc: {test_m['balanced_acc']:.4f}")
    print(f"Student params: {student_params:,}")
    print(f"Compression: {teacher_params / student_params:.2f}x")
    print(f"Model size: {model_size:.3f} MB")
    print(f"CPU latency: {latency:.3f} ms")
    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
