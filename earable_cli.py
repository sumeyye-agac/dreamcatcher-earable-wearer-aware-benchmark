from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _csv_list(s: str, *, cast=str):
    vals = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(cast(x))
    return vals


def _fmt_runval(v) -> str:
    # file-name friendly: 0.7 -> 0p7, 1e-3 -> 1e-3
    s = str(v)
    return s.replace(".", "p")


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


def sweep_kd(args: argparse.Namespace) -> None:
    """
    Simple grid-search runner for KD.

    Example:
      earable sweep kd --students crnn,crnn_cbam --alpha 0.3,0.7 --tau 2,5 --lr 1e-3 --batch_size 4,8 --dataset_mode smoke --max_samples 512
    """
    # Allow dry-run without HF token (prints commands only).
    if not args.dry_run:
        _require_hf_token()

    students = _csv_list(args.students, cast=str)
    alphas = _csv_list(args.alpha, cast=float)
    taus = _csv_list(args.tau, cast=float)
    lrs = _csv_list(args.lr, cast=float)
    bss = _csv_list(args.batch_size, cast=int)

    if not students:
        raise SystemExit("Provide --students, e.g. --students crnn,crnn_cbam")
    if not alphas or not taus or not lrs or not bss:
        raise SystemExit("Provide non-empty --alpha/--tau/--lr/--batch_size comma lists.")

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_prefix = args.run_prefix or f"kd_sweep_{ts}"

    combos = list(itertools.product(students, alphas, taus, lrs, bss))
    if args.max_runs and args.max_runs > 0:
        combos = combos[: args.max_runs]

    print(f"[earable] KD sweep: {len(combos)} runs", flush=True)

    jobs = int(args.jobs) if args.jobs else 1
    if jobs < 1:
        jobs = 1

    runs: list[tuple[str, list[str]]] = []
    for student, alpha, tau, lr, bs in combos:
        extra = []
        if student in {"crnn", "crnn_cbam"}:
            extra += ["--rnn_hidden", str(args.rnn_hidden), "--rnn_layers", str(args.rnn_layers)]
        if student == "crnn_cbam":
            extra += ["--cbam_reduction", str(args.cbam_reduction), "--cbam_sa_kernel", str(args.cbam_sa_kernel)]

        run_name = (
            f"{run_prefix}_{student}"
            f"_a{_fmt_runval(alpha)}_t{_fmt_runval(tau)}"
            f"_lr{_fmt_runval(lr)}_bs{bs}"
            f"_{args.dataset_mode}"
        )
        cmd = [
            sys.executable,
            "-m",
            "src.training.train_kd",
            "--student",
            student,
            "--dataset_mode",
            args.dataset_mode,
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(bs),
            "--lr",
            str(lr),
            "--alpha",
            str(alpha),
            "--tau",
            str(tau),
            "--teacher_name",
            args.teacher_name,
            "--seed",
            str(args.seed),
            "--run_name",
            run_name,
        ]
        if args.max_samples and args.max_samples > 0:
            cmd += ["--max_samples", str(args.max_samples)]
        cmd += extra

        if args.dry_run:
            print(f"[earable] (dry-run) would run: {run_name}", flush=True)
            print(f"[earable] $ {' '.join(cmd)}", flush=True)
            continue

        runs.append((run_name, cmd))

    if args.dry_run:
        return

    if jobs == 1:
        for run_name, cmd in runs:
            print(f"[earable] run={run_name}", flush=True)
            _run(cmd)
        return

    print(f"[earable] Running in parallel: jobs={jobs}", flush=True)

    def _worker(rn: str, c: list[str]) -> str:
        print(f"[earable] run_start={rn}", flush=True)
        subprocess.run(c, check=True)
        print(f"[earable] run_done={rn}", flush=True)
        return rn

    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(_worker, rn, c): rn for rn, c in runs}
        for fut in as_completed(futs):
            rn = futs[fut]
            try:
                fut.result()
            except Exception as e:
                failures.append((rn, str(e)))

    if failures:
        print("[earable] Sweep finished with failures:", file=sys.stderr)
        for rn, err in failures:
            print(f"  - {rn}: {err}", file=sys.stderr)
        raise SystemExit(1)


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

    p_sweep = sub.add_parser("sweep", help="Run a simple grid-search style sweep.")
    p_sweep_sub = p_sweep.add_subparsers(dest="sweep_cmd", required=True)

    p_kd = p_sweep_sub.add_parser("kd", help="KD hyperparameter sweep (grid).")
    p_kd.add_argument("--students", type=str, default="crnn,crnn_cbam", help="Comma list, e.g. crnn,crnn_cbam")
    p_kd.add_argument("--alpha", type=str, required=True, help="Comma list, e.g. 0.3,0.7")
    p_kd.add_argument("--tau", type=str, required=True, help="Comma list, e.g. 2,5,10")
    p_kd.add_argument("--lr", type=str, required=True, help="Comma list, e.g. 1e-3,3e-4")
    p_kd.add_argument("--batch_size", type=str, required=True, help="Comma list, e.g. 4,8")
    p_kd.add_argument("--epochs", type=int, default=1)
    p_kd.add_argument("--seed", type=int, default=42)
    p_kd.add_argument("--dataset_mode", type=str, default="smoke", choices=["full", "smoke"])
    p_kd.add_argument("--max_samples", type=int, default=512, help="Per-split cap when dataset_mode=smoke")
    p_kd.add_argument("--teacher_name", type=str, default="facebook/wav2vec2-base")
    p_kd.add_argument("--run_prefix", type=str, default="", help="Prefix for run_name; defaults to timestamped prefix")
    p_kd.add_argument("--max_runs", type=int, default=0, help="Optional cap on number of runs (0 = no cap)")
    p_kd.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    p_kd.add_argument("--jobs", type=int, default=1, help="Number of parallel workers (default: 1)")
    p_kd.add_argument("--rnn_hidden", type=int, default=64)
    p_kd.add_argument("--rnn_layers", type=int, default=1)
    p_kd.add_argument("--cbam_reduction", type=int, default=8)
    p_kd.add_argument("--cbam_sa_kernel", type=int, default=7)

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
    if args.cmd == "sweep":
        if args.sweep_cmd == "kd":
            sweep_kd(args)
            return 0
        raise SystemExit(f"Unknown sweep: {args.sweep_cmd}")

    raise SystemExit("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())

