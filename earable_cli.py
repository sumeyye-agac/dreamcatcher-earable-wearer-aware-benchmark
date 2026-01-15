from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _require_hf_token() -> None:
    try:
        from huggingface_hub import get_token
    except Exception:
        # If deps aren't installed yet, fail later when training modules import.
        return
    if not get_token():
        raise SystemExit(
            "ERROR: HuggingFace token not found. DreamCatcher is a gated dataset.\n"
            "Run: hf auth login\n"
            "Or set: export HF_TOKEN=\"hf_...\"; export HUGGINGFACE_HUB_TOKEN=\"$HF_TOKEN\""
        )


def _run(cmd: list[str]) -> None:
    print(f"[earable] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def suite_audio_smoke(args: argparse.Namespace) -> None:
    _require_hf_token()
    for model, extra in [
        ("tinycnn", []),
        ("crnn", ["--rnn_hidden", "64", "--rnn_layers", "1"]),
        ("crnn_cbam", ["--rnn_hidden", "64", "--rnn_layers", "1", "--cbam_reduction", "8", "--cbam_sa_kernel", "7"]),
    ]:
        run_name = f"{model}_smoke"
        _run(
            [
                sys.executable,
                "-m",
                "src.training.train_baseline",
                "--model",
                model,
                "--dataset_mode",
                "smoke",
                "--invalid_audio_policy",
                "skip",
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--run_name",
                run_name,
                *extra,
            ]
        )


def suite_audio_benchmark(args: argparse.Namespace) -> None:
    _require_hf_token()
    for model, extra in [
        ("tinycnn", []),
        ("crnn", ["--rnn_hidden", "64", "--rnn_layers", "1"]),
        ("crnn_cbam", ["--rnn_hidden", "64", "--rnn_layers", "1", "--cbam_reduction", "8", "--cbam_sa_kernel", "7"]),
    ]:
        run_name = f"{model}_baseline"
        _run(
            [
                sys.executable,
                "-m",
                "src.training.train_baseline",
                "--model",
                model,
                "--dataset_mode",
                "full",
                "--invalid_audio_policy",
                "skip",
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--run_name",
                run_name,
                *extra,
            ]
        )


def suite_kd_smoke(args: argparse.Namespace) -> None:
    _require_hf_token()
    for student, extra in [
        ("crnn", ["--rnn_hidden", "64", "--rnn_layers", "1"]),
        ("crnn_cbam", ["--rnn_hidden", "64", "--rnn_layers", "1", "--cbam_reduction", "8", "--cbam_sa_kernel", "7"]),
    ]:
        run_name = "crnn_rbkd_smoke" if student == "crnn" else "crnn_cbam_rbkdatt_smoke"
        _run(
            [
                sys.executable,
                "-m",
                "src.training.train_kd",
                "--student",
                student,
                "--dataset_mode",
                "smoke",
                "--epochs",
                "1",
                "--batch_size",
                "4",
                "--max_samples",
                "512",
                "--lr",
                "1e-3",
                "--alpha",
                "0.7",
                "--tau",
                "5",
                "--teacher_name",
                "facebook/wav2vec2-base",
                "--run_name",
                run_name,
                *extra,
            ]
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="earable")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_suite = sub.add_parser("suite", help="Run common experiment suites.")
    p_suite.add_argument(
        "name",
        choices=["audio-smoke", "audio-benchmark", "kd-smoke"],
        help="Which suite to run.",
    )
    p_suite.add_argument("--epochs", type=int, default=1)
    p_suite.add_argument("--batch_size", type=int, default=8)
    p_suite.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args(argv)

    # Ensure we run from repo root if invoked elsewhere.
    # (Best-effort: if src/ exists next to this file)
    try:
        root = Path(__file__).resolve().parent
        if (root / "src").exists():
            pass
    except Exception:
        pass

    if args.cmd == "suite":
        if args.name == "audio-smoke":
            suite_audio_smoke(args)
        elif args.name == "audio-benchmark":
            suite_audio_benchmark(args)
        elif args.name == "kd-smoke":
            suite_kd_smoke(args)
        else:
            raise SystemExit(f"Unknown suite: {args.name}")
        return 0

    raise SystemExit("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())

