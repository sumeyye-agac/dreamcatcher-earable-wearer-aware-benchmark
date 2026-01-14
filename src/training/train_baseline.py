from __future__ import annotations

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dreamcatcher_hf import DreamCatcherHFAudioDataset, DreamCatcherHFAudioConfig, LABELS
from src.evaluation.metrics import macro_f1
from src.models.tinycnn import TinyCNN
from src.models.crnn import CRNN
from src.models.crnn_cbam import CRNN_CBAM
from src.utils.reproducibility import set_seed
from src.utils.benchmarking import count_params, measure_cpu_latency, append_to_leaderboard


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
        return CRNN_CBAM(
            n_classes=n_classes,
            rnn_hidden=args.rnn_hidden,
            rnn_layers=args.rnn_layers,
            cbam_reduction=args.cbam_reduction,
            cbam_sa_kernel=args.cbam_sa_kernel,
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
    acc = float(np.mean(np.array(all_true) == np.array(all_pred)))
    f1 = macro_f1(all_true, all_pred, n_classes=len(LABELS))
    return avg_loss, acc, f1


@torch.no_grad()
def evaluate(model, dl, loss_fn, device):
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
    acc = float(np.mean(np.array(all_true) == np.array(all_pred)))
    f1 = macro_f1(all_true, all_pred, n_classes=len(LABELS))
    return avg_loss, acc, f1


def main():
    parser = argparse.ArgumentParser()

    # model + training
    parser.add_argument("--model", type=str, default="tinycnn", choices=["tinycnn", "crnn", "crnn_cbam"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)

    # audio
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n_mels", type=int, default=64)

    # CRNN
    parser.add_argument("--rnn_hidden", type=int, default=64)
    parser.add_argument("--rnn_layers", type=int, default=1)

    # CBAM
    parser.add_argument("--cbam_reduction", type=int, default=8)
    parser.add_argument("--cbam_sa_kernel", type=int, default=7)

    # logging
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--out_csv", type=str, default="results/leaderboard.csv")
    parser.add_argument("--latency_T", type=int, default=400)

    args = parser.parse_args()

    # Print training configuration
    print(f"\n{'='*60}")
    print(f"Starting Training")
    print(f"  Model: {args.model}")
    print(f"  Run Name: {args.run_name if args.run_name else 'default'}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Seed: {args.seed}")
    print(f"{'='*60}\n")

    set_seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading datasets...")
    cfg = DreamCatcherHFAudioConfig(sample_rate=args.sr, n_mels=args.n_mels)
    train_ds = DreamCatcherHFAudioDataset(split="train", cfg=cfg)
    val_ds = DreamCatcherHFAudioDataset(split="validation", cfg=cfg)
    test_ds = DreamCatcherHFAudioDataset(split="test", cfg=cfg)
    print(f"All datasets loaded: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}\n")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(args.model, n_classes=len(LABELS), args=args).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_f1 = run_one_epoch(model, train_dl, opt, loss_fn, device)
        va_loss, va_acc, va_f1 = evaluate(model, val_dl, loss_fn, device)

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"epoch={epoch} "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} train_f1={tr_f1:.4f} | "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} val_f1={va_f1:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    te_loss, te_acc, te_f1 = evaluate(model, test_dl, loss_fn, device)
    print(f"test_loss={te_loss:.4f} test_acc={te_acc:.4f} test_f1={te_f1:.4f}")

    # Benchmark logging
    param_count = count_params(model)
    lat_ms = measure_cpu_latency(model, input_shape=(1, 1, args.n_mels, args.latency_T))

    run_name = args.run_name if args.run_name else f"{args.model}_seed{args.seed}"

    row = {
        "run_name": run_name,
        "task": "audio_event_label",
        "model": args.model,
        "teacher": "",
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "sr": args.sr,
        "n_mels": args.n_mels,
        "rnn_hidden": args.rnn_hidden if args.model != "tinycnn" else "",
        "rnn_layers": args.rnn_layers if args.model != "tinycnn" else "",
        "cbam_reduction": args.cbam_reduction if args.model == "crnn_cbam" else "",
        "cbam_sa_kernel": args.cbam_sa_kernel if args.model == "crnn_cbam" else "",
        "alpha": "",
        "tau": "",
        "best_val_f1": round(best_val_f1, 6),
        "test_acc": round(te_acc, 6),
        "test_f1": round(te_f1, 6),
        "params": param_count,
        "cpu_latency_ms": round(lat_ms, 4),
    }

    append_to_leaderboard(args.out_csv, row)
    print(f"Logged to {args.out_csv}")
    print("Done.")


if __name__ == "__main__":
    main()
