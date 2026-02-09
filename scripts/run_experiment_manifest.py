#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.utils.artifact_contract import validate_run_artifact_contract


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def fslug(x: float) -> str:
    s = f"{x:g}"
    return s.replace("-", "m").replace(".", "p")


def flatten_args(arg_map: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for k, v in arg_map.items():
        if v is None:
            continue
        out.extend([f"--{k}", str(v)])
    return out


def nested_get(d: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


@dataclass
class RunSpec:
    stage: str
    run_name: str
    model: str
    cmd: list[str]
    meta: dict[str, Any]


class ManifestRunner:
    def __init__(self, manifest_path: Path, resume: bool, dry_run: bool, fresh_start: bool):
        self.manifest_path = manifest_path
        self.resume = resume
        self.dry_run = dry_run
        self.fresh_start = fresh_start
        self.repo_root = Path.cwd()
        self.logs_dir = self.repo_root / "logs"
        self.results_dir = self.repo_root / "results"
        self.run_root = self.results_dir / "runs"
        self.status_csv = self.results_dir / "orchestration" / "manifest_status.csv"
        self.orch_log = self.logs_dir / f"manifest_orchestrator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.seed = int(self.manifest["seed"])
        self.common = self.manifest["common_train"]
        self.class_weights = self.manifest["class_weights"]
        self.cache_dir = self.manifest["cache_dir"]
        self.output_csv = self.manifest["output_csv"]
        self.steps_csv = self.manifest["steps_csv"]
        self.runtime_policy = self.manifest.get("runtime_policy", {})

        self.full_epochs = int(self.runtime_policy.get("full_epochs", self.common["epochs"]))
        self.kd_after_teacher_gate_only = bool(
            self.runtime_policy.get("kd_after_teacher_gap_gate_only", True)
        )

        self.run_index: dict[str, RunSpec] = {}

    def log(self, msg: str) -> None:
        line = f"[{utc_now_iso()}] {msg}"
        print(line, flush=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        with self.orch_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def append_status(
        self,
        *,
        stage: str,
        run_name: str,
        model: str,
        status: str,
        duration_s: float,
        return_code: int,
        note: str = "",
    ) -> None:
        self.status_csv.parent.mkdir(parents=True, exist_ok=True)
        exists = self.status_csv.exists() and self.status_csv.stat().st_size > 0
        row = {
            "ts_utc": utc_now_iso(),
            "stage": stage,
            "run_name": run_name,
            "model": model,
            "status": status,
            "duration_s": f"{duration_s:.2f}",
            "return_code": str(return_code),
            "note": note,
        }
        fieldnames = list(row.keys())
        with self.status_csv.open("a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                w.writeheader()
            w.writerow(row)
        self._refresh_progress_report()

    def _load_status_rows(self) -> list[dict[str, str]]:
        if not self.status_csv.exists():
            return []
        with self.status_csv.open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def _metrics_for_run(self, run_name: str) -> dict[str, Any]:
        p = self.run_root / run_name / "metrics.json"
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _refresh_progress_report(self) -> None:
        rows = self._load_status_rows()
        out_dir = self.results_dir / "orchestration"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "progress_table.csv"
        md_path = out_dir / "progress_table.md"

        table_rows: list[dict[str, str]] = []
        for r in rows:
            run_name = r.get("run_name", "")
            metrics = self._metrics_for_run(run_name)
            val_f1 = nested_get(metrics, ["validation_best", "f1_macro"], "")
            test_f1 = nested_get(metrics, ["test", "f1_macro"], "")
            test_acc = nested_get(metrics, ["test", "acc"], "")
            best_epoch = metrics.get("best_epoch", "")
            epochs_ran = metrics.get("epochs_ran", "")
            params = metrics.get("params", "")
            model = metrics.get("model", r.get("model", ""))
            table_rows.append(
                {
                    "ts_utc": r.get("ts_utc", ""),
                    "stage": r.get("stage", ""),
                    "run_name": run_name,
                    "status": r.get("status", ""),
                    "duration_min": (
                        f"{(float(r.get('duration_s', '0') or 0.0) / 60.0):.2f}"
                        if r.get("duration_s", "")
                        else ""
                    ),
                    "model": str(model),
                    "params": str(params),
                    "val_f1": (f"{float(val_f1):.4f}" if val_f1 != "" else ""),
                    "test_f1": (f"{float(test_f1):.4f}" if test_f1 != "" else ""),
                    "test_acc": (f"{float(test_acc):.4f}" if test_acc != "" else ""),
                    "best_epoch": str(best_epoch),
                    "epochs_ran": str(epochs_ran),
                }
            )

        fieldnames = [
            "ts_utc",
            "stage",
            "run_name",
            "status",
            "duration_min",
            "model",
            "params",
            "val_f1",
            "test_f1",
            "test_acc",
            "best_epoch",
            "epochs_ran",
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(table_rows)

        lines = [
            "# Manifest Progress Table",
            "",
            f"Last update: {utc_now_iso()}",
            "",
            "| ts_utc | stage | run_name | status | duration_min | model | params | val_f1 | test_f1 | test_acc | best_epoch | epochs_ran |",
            "|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
        for tr in table_rows[-200:]:
            lines.append(
                "| {ts_utc} | {stage} | {run_name} | {status} | {duration_min} | {model} | {params} | {val_f1} | {test_f1} | {test_acc} | {best_epoch} | {epochs_ran} |".format(
                    **{k: (v if v != "" else "-") for k, v in tr.items()}
                )
            )
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def clean_outputs(self) -> None:
        self.log("Cleaning old run artifacts (keeping cache)...")
        paths = [
            self.run_root,
            self.logs_dir,
            self.results_dir / "leaderboard.csv",
            self.results_dir / "leaderboard.csv.lock",
            self.results_dir / "run_steps.csv",
            self.results_dir / "run_steps.csv.lock",
            self.results_dir / "orchestration",
        ]
        for p in paths:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.exists():
                p.unlink()
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.run_root.mkdir(parents=True, exist_ok=True)
        self._refresh_progress_report()

    def _is_cache_split_valid(self, split: str, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            import h5py
        except Exception:
            return path.exists()
        try:
            with h5py.File(path, "r") as f:
                if "spectrograms" not in f or "labels" not in f:
                    return False
                specs = f["spectrograms"]
                labels = f["labels"]
                if specs.ndim != 3 or labels.ndim != 1:
                    return False
                if specs.shape[0] <= 0 or labels.shape[0] <= 0:
                    return False
                if specs.shape[0] != labels.shape[0]:
                    return False
                if int(f.attrs.get("n_samples", -1)) != specs.shape[0]:
                    return False
                if int(f.attrs.get("n_mels", -1)) != int(self.common["n_mels"]):
                    return False
                if int(f.attrs.get("sample_rate", -1)) != int(self.common["sr"]):
                    return False
                if str(f.attrs.get("split", "")) != split:
                    return False
                if int(f.attrs.get("max_samples", 0)) != 0:
                    return False
            return True
        except Exception:
            return False

    def ensure_cache(self) -> None:
        base = Path(self.cache_dir)
        required = {
            "train": base / "train.h5",
            "validation": base / "validation.h5",
            "test": base / "test.h5",
        }
        missing_splits = [
            split for split, path in required.items() if not self._is_cache_split_valid(split, path)
        ]
        if not missing_splits:
            self.log(f"Cache OK: {base}")
            self.append_status(
                stage="setup_cache",
                run_name="cache_preprocess",
                model="preprocess",
                status="skipped_cache_ok",
                duration_s=0.0,
                return_code=0,
            )
            return
        self.log(
            "Cache missing/invalid: "
            f"{[str(required[s]) for s in missing_splits]} "
            f"-> running preprocess for splits={missing_splits}"
        )
        cmd = [
            sys.executable,
            "scripts/preprocess.py",
            "--splits",
            ",".join(missing_splits),
            "--skip-existing",
            "--batch-size",
            str(int(self.runtime_policy.get("preprocess_batch_size", 128))),
            "--compression",
            str(self.runtime_policy.get("preprocess_compression", "lzf")),
            "--compression-level",
            str(int(self.runtime_policy.get("preprocess_compression_level", 1))),
        ]
        log_path = self.logs_dir / "preprocess_cache.log"
        t0 = time.time()
        rc = self._run_command(cmd, log_path)
        dt = time.time() - t0
        if rc != 0:
            self.append_status(
                stage="setup_cache",
                run_name="cache_preprocess",
                model="preprocess",
                status="failed",
                duration_s=dt,
                return_code=rc,
                note=f"log={log_path}",
            )
            raise RuntimeError(f"preprocess failed with code {rc}. See {log_path}")
        self.append_status(
            stage="setup_cache",
            run_name="cache_preprocess",
            model="preprocess",
            status="ok",
            duration_s=dt,
            return_code=0,
            note=f"log={log_path}",
        )

    def _run_command(self, cmd: list[str], log_path: Path) -> int:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"\n[{utc_now_iso()}] CMD: {' '.join(cmd)}\n")
            f.flush()
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=self.repo_root)
        return proc.returncode

    def _is_run_valid(self, run_name: str) -> bool:
        metrics_path = self.run_root / run_name / "metrics.json"
        if not metrics_path.exists():
            return False
        try:
            validate_run_artifact_contract(self.run_root / run_name, metrics_path)
            return True
        except Exception:
            return False

    @staticmethod
    def _stable_hash(payload: dict[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _spec_signature(self, spec: RunSpec) -> dict[str, Any]:
        payload = {
            "module": spec.meta.get("module", ""),
            "args": spec.meta.get("args", {}),
            "stage": spec.stage,
            "model": spec.model,
        }
        return {
            "hash": self._stable_hash(payload),
            "payload": payload,
        }

    @staticmethod
    def _value_equal(a: Any, b: Any) -> bool:
        if isinstance(a, bool) or isinstance(b, bool):
            return bool(a) == bool(b)
        try:
            return abs(float(a) - float(b)) < 1e-12
        except (TypeError, ValueError):
            pass
        return str(a) == str(b)

    def _resolved_config_matches_spec(self, run_name: str, spec: RunSpec) -> bool:
        resolved_path = self.run_root / run_name / "resolved_config.json"
        if not resolved_path.exists():
            return False
        try:
            resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        args = spec.meta.get("args", {})
        for key, expected in args.items():
            if key not in resolved:
                return False
            if not self._value_equal(resolved[key], expected):
                return False
        return True

    def _save_spec_signature(self, run_name: str, spec: RunSpec) -> None:
        out = self.run_root / run_name / "orchestrator_spec_signature.json"
        out.write_text(json.dumps(self._spec_signature(spec), indent=2), encoding="utf-8")

    def _stored_spec_signature_matches(self, run_name: str, spec: RunSpec) -> bool:
        path = self.run_root / run_name / "orchestrator_spec_signature.json"
        if not path.exists():
            return False
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return False
        return stored.get("hash") == self._spec_signature(spec)["hash"]

    def _is_run_reusable(self, spec: RunSpec) -> bool:
        if not self._is_run_valid(spec.run_name):
            return False
        # Primary check: resolved config must match current spec args.
        if not self._resolved_config_matches_spec(spec.run_name, spec):
            self.log(f"RESUME mismatch (resolved_config): rerun {spec.run_name}")
            return False
        # Secondary strict check when orchestrator signature exists.
        sig_path = self.run_root / spec.run_name / "orchestrator_spec_signature.json"
        if sig_path.exists() and not self._stored_spec_signature_matches(spec.run_name, spec):
            self.log(f"RESUME mismatch (orchestrator signature): rerun {spec.run_name}")
            return False
        return True

    def run_spec(self, spec: RunSpec) -> None:
        self.run_index[spec.run_name] = spec
        log_path = self.logs_dir / f"{spec.run_name}.log"
        if self.resume and self._is_run_reusable(spec):
            self.log(f"SKIP valid run: {spec.run_name}")
            self.append_status(
                stage=spec.stage,
                run_name=spec.run_name,
                model=spec.model,
                status="skipped_valid",
                duration_s=0.0,
                return_code=0,
            )
            return
        self.log(f"START {spec.stage} :: {spec.run_name}")
        if self.dry_run:
            self.log(f"DRY-RUN CMD: {' '.join(spec.cmd)}")
            self.append_status(
                stage=spec.stage,
                run_name=spec.run_name,
                model=spec.model,
                status="dry_run",
                duration_s=0.0,
                return_code=0,
            )
            return

        t0 = time.time()
        rc = self._run_command(spec.cmd, log_path)
        dt = time.time() - t0
        if rc != 0:
            self.append_status(
                stage=spec.stage,
                run_name=spec.run_name,
                model=spec.model,
                status="failed",
                duration_s=dt,
                return_code=rc,
                note=f"log={log_path}",
            )
            raise RuntimeError(f"Run failed: {spec.run_name} (code={rc}). See {log_path}")

        if not self._is_run_valid(spec.run_name):
            self.append_status(
                stage=spec.stage,
                run_name=spec.run_name,
                model=spec.model,
                status="invalid_artifacts",
                duration_s=dt,
                return_code=3,
                note=f"log={log_path}",
            )
            raise RuntimeError(f"Run completed but artifact contract failed: {spec.run_name}")

        self.append_status(
            stage=spec.stage,
            run_name=spec.run_name,
            model=spec.model,
            status="ok",
            duration_s=dt,
            return_code=0,
        )
        self._save_spec_signature(spec.run_name, spec)
        self.log(f"DONE {spec.run_name} ({dt/60.0:.1f} min)")

    def read_metrics(self, run_name: str) -> dict[str, Any]:
        p = self.run_root / run_name / "metrics.json"
        if not p.exists():
            raise FileNotFoundError(f"metrics missing for run {run_name}")
        return json.loads(p.read_text(encoding="utf-8"))

    def run_val_f1(self, run_name: str) -> float:
        metrics = self.read_metrics(run_name)
        f1 = nested_get(metrics, ["validation_best", "f1_macro"], None)
        if f1 is None:
            raise RuntimeError(f"Missing validation_best.f1_macro for run {run_name}")
        return float(f1)

    def build_train_spec(
        self,
        *,
        stage: str,
        run_name: str,
        model: str,
        lr: float,
        weight_decay: float,
        grad_clip: float,
        epochs: int,
        early_stop_patience: int,
        cbam_reduction: int | None = None,
        cbam_sa_kernel: int | None = None,
    ) -> RunSpec:
        args = {
            "model": model,
            "run_name": run_name,
            "seed": self.seed,
            "epochs": epochs,
            "batch_size": self.common["batch_size"],
            "num_workers": self.common["num_workers"],
            "lr": lr,
            "weight_decay": weight_decay,
            "grad_clip": grad_clip,
            "early_stop_patience": early_stop_patience,
            "early_stop_min_delta": self.common["early_stop_min_delta"],
            "class_weights": self.class_weights,
            "cache_dir": self.cache_dir,
            "out_csv": self.output_csv,
            "steps_csv": self.steps_csv,
            "sr": self.common["sr"],
            "n_mels": self.common["n_mels"],
            "rnn_hidden": self.common["rnn_hidden"],
            "rnn_layers": self.common["rnn_layers"],
            "att_mode": self.common["att_mode"],
        }
        if cbam_reduction is not None:
            args["cbam_reduction"] = cbam_reduction
        if cbam_sa_kernel is not None:
            args["cbam_sa_kernel"] = cbam_sa_kernel
        cmd = [sys.executable, "-m", "src.training.train", *flatten_args(args)]
        return RunSpec(
            stage=stage,
            run_name=run_name,
            model=model,
            cmd=cmd,
            meta={
                "module": "src.training.train",
                "args": args,
                "cbam_reduction": cbam_reduction,
                "cbam_sa_kernel": cbam_sa_kernel,
                "lr": lr,
                "weight_decay": weight_decay,
                "grad_clip": grad_clip,
                "epochs": epochs,
                "early_stop_patience": early_stop_patience,
                "is_cbam": model.endswith("_cbam"),
            },
        )

    def build_kd_spec(
        self,
        *,
        stage: str,
        run_name: str,
        student_model: str,
        teacher_model: str,
        teacher_checkpoint: str,
        alpha: float,
        temperature: float,
        epochs: int,
        early_stop_patience: int,
        cbam_reduction: int | None = None,
        cbam_sa_kernel: int | None = None,
    ) -> RunSpec:
        kd_cfg = self.manifest["kd"]
        args = {
            "run_name": run_name,
            "seed": self.seed,
            "student_model": student_model,
            "teacher_model": teacher_model,
            "teacher_checkpoint": teacher_checkpoint,
            "temperature": temperature,
            "alpha": alpha,
            "epochs": epochs,
            "batch_size": kd_cfg["batch_size"],
            "num_workers": kd_cfg["num_workers"],
            "lr": kd_cfg["lr"],
            "weight_decay": kd_cfg["weight_decay"],
            "early_stop_patience": early_stop_patience,
            "early_stop_min_delta": kd_cfg["early_stop_min_delta"],
            "class_weights": self.class_weights,
            "cache_dir": self.cache_dir,
            "out_csv": self.output_csv,
            "steps_csv": self.steps_csv,
            "sr": self.common["sr"],
            "n_mels": self.common["n_mels"],
            "rnn_hidden": self.common["rnn_hidden"],
            "rnn_layers": self.common["rnn_layers"],
            "att_mode": self.common["att_mode"],
        }
        if cbam_reduction is not None:
            args["cbam_reduction"] = cbam_reduction
        if cbam_sa_kernel is not None:
            args["cbam_sa_kernel"] = cbam_sa_kernel
        cmd = [sys.executable, "-m", "src.training.train_kd", *flatten_args(args)]
        return RunSpec(
            stage=stage,
            run_name=run_name,
            model=student_model,
            cmd=cmd,
            meta={
                "module": "src.training.train_kd",
                "args": args,
                "student_model": student_model,
                "alpha": alpha,
                "temperature": temperature,
                "cbam_reduction": cbam_reduction,
                "cbam_sa_kernel": cbam_sa_kernel,
                "epochs": epochs,
                "early_stop_patience": early_stop_patience,
            },
        )

    def build_phase1_full_specs(self) -> list[RunSpec]:
        specs: list[RunSpec] = []
        baseline = self.manifest["baseline"]
        lr = float(self.common["lr"])
        for model in baseline["models"]:
            run_name = f"p1_{model}_seed{self.seed}"
            specs.append(
                self.build_train_spec(
                    stage="phase1_full_base",
                    run_name=run_name,
                    model=model,
                    lr=lr,
                    weight_decay=float(self.common["weight_decay"]),
                    grad_clip=float(self.common["grad_clip"]),
                    epochs=self.full_epochs,
                    early_stop_patience=int(self.common["early_stop_patience"]),
                )
            )

        for model, rr, sk in itertools.product(
            baseline["cbam_models"],
            baseline["cbam_reduction_values"],
            baseline["cbam_sa_kernel_values"],
        ):
            run_name = f"p1_{model}_rr{rr}_sk{sk}_seed{self.seed}"
            specs.append(
                self.build_train_spec(
                    stage="phase1_full_cbam",
                    run_name=run_name,
                    model=model,
                    lr=float(self.common["lr"]),
                    weight_decay=float(self.common["weight_decay"]),
                    grad_clip=float(self.common["grad_clip"]),
                    epochs=self.full_epochs,
                    early_stop_patience=int(self.common["early_stop_patience"]),
                    cbam_reduction=int(rr),
                    cbam_sa_kernel=int(sk),
                )
            )
        return specs

    def select_phase1_full_specs(self, full_specs: list[RunSpec]) -> list[RunSpec]:
        return full_specs

    def pick_best_student_f1(self) -> tuple[str, float]:
        gate = self.manifest["teacher_gate"]
        candidates: list[tuple[str, float]] = []
        for run_dir in sorted(self.run_root.iterdir()):
            if not run_dir.is_dir():
                continue
            run_name = run_dir.name
            if not run_name.startswith("p1_"):
                continue
            metrics_path = run_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            model = metrics.get("model", "")
            if model not in set(gate["student_pool_models"]):
                continue
            f1 = nested_get(metrics, ["validation_best", "f1_macro"], None)
            if f1 is None:
                continue
            candidates.append((run_name, float(f1)))
        if not candidates:
            raise RuntimeError("No full phase1 student candidate metrics found for teacher gap check.")
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0]

    def pick_teacher_f1(self, run_name: str) -> float:
        return self.run_val_f1(run_name)

    def run_teacher_tuning_if_needed(self, best_student_f1: float) -> tuple[str, float, bool]:
        gate = self.manifest["teacher_gate"]
        teacher_run = gate["teacher_base_run_name"]
        teacher_f1 = self.pick_teacher_f1(teacher_run)
        min_gap = float(gate["min_f1_gap"])
        gap = teacher_f1 - best_student_f1
        self.log(
            f"Teacher gap check: teacher={teacher_run} f1={teacher_f1:.4f} "
            f"best_student_f1={best_student_f1:.4f} gap={gap:.4f} target={min_gap:.4f}"
        )
        if gap >= min_gap or not gate.get("tune_teacher_if_gap_below", False):
            return teacher_run, teacher_f1, gap >= min_gap

        tuning = gate["teacher_tuning"]
        combos = list(
            itertools.product(
                tuning["lr_values"],
                tuning["weight_decay_values"],
                tuning["grad_clip_values"],
            )
        )
        for lr, wd, gc in combos:
            run_name = (
                f"p1_crnn_tune_lr{fslug(float(lr))}_wd{fslug(float(wd))}_gc{fslug(float(gc))}_seed{self.seed}"
            )
            spec = self.build_train_spec(
                stage="phase1_teacher_tune",
                run_name=run_name,
                model=gate["teacher_model"],
                lr=float(lr),
                weight_decay=float(wd),
                grad_clip=float(gc),
                epochs=self.full_epochs,
                early_stop_patience=int(self.common["early_stop_patience"]),
            )
            self.run_spec(spec)
            tuned_f1 = self.pick_teacher_f1(run_name)
            if tuned_f1 > teacher_f1:
                teacher_f1 = tuned_f1
                teacher_run = run_name
            gap = teacher_f1 - best_student_f1
            self.log(f"Teacher tuning update: best_teacher={teacher_run} f1={teacher_f1:.4f} gap={gap:.4f}")
            if gap >= min_gap:
                return teacher_run, teacher_f1, True
        return teacher_run, teacher_f1, False

    def select_best_cbam_params(self, model_name: str) -> tuple[int, int]:
        best: tuple[float, int, int] | None = None
        for run_dir in sorted(self.run_root.iterdir()):
            if not run_dir.is_dir():
                continue
            run_name = run_dir.name
            if not run_name.startswith("p1_"):
                continue
            metrics_path = run_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            if m.get("model") != model_name:
                continue
            hp = m.get("hyperparameters", {})
            rr = hp.get("cbam_reduction")
            sk = hp.get("cbam_sa_kernel")
            f1 = nested_get(m, ["validation_best", "f1_macro"], None)
            if rr is None or sk is None or f1 is None:
                continue
            cand = (float(f1), int(rr), int(sk))
            if best is None or cand[0] > best[0]:
                best = cand
        if best is None:
            raise RuntimeError(f"No completed full CBAM baseline found for {model_name}")
        return best[1], best[2]

    def build_phase2_kd_full_specs(self, teacher_checkpoint: str) -> list[RunSpec]:
        kd_cfg = self.manifest["kd"]
        specs: list[RunSpec] = []
        for student in kd_cfg["students"]:
            rr, sk = (None, None)
            if student.endswith("_cbam"):
                rr, sk = self.select_best_cbam_params(student)
                self.log(f"KD fixed CBAM params for {student}: rr={rr}, sk={sk}")
            for alpha, tau in itertools.product(kd_cfg["alphas"], kd_cfg["temperatures"]):
                if rr is None:
                    run_name = f"p2_kd_{student}_a{fslug(float(alpha))}_t{fslug(float(tau))}_seed{self.seed}"
                else:
                    run_name = (
                        f"p2_kd_{student}_rr{rr}_sk{sk}_a{fslug(float(alpha))}_t{fslug(float(tau))}_seed{self.seed}"
                    )
                specs.append(
                    self.build_kd_spec(
                        stage="phase2_full_kd",
                        run_name=run_name,
                        student_model=student,
                        teacher_model=kd_cfg["teacher_model"],
                        teacher_checkpoint=teacher_checkpoint,
                        alpha=float(alpha),
                        temperature=float(tau),
                        epochs=int(kd_cfg["epochs"]),
                        early_stop_patience=int(kd_cfg["early_stop_patience"]),
                        cbam_reduction=rr,
                        cbam_sa_kernel=sk,
                    )
                )
        return specs

    def select_phase2_kd_full_specs(self, kd_full_specs: list[RunSpec]) -> list[RunSpec]:
        self.log(f"KD policy: running all {len(kd_full_specs)} KD combinations.")
        return kd_full_specs

    def print_plan_summary(self) -> None:
        baseline = self.manifest["baseline"]
        kd_cfg = self.manifest["kd"]
        n_base = len(baseline["models"])
        n_cbam = (
            len(baseline["cbam_models"])
            * len(baseline["cbam_reduction_values"])
            * len(baseline["cbam_sa_kernel_values"])
        )
        n_kd = len(kd_cfg["students"]) * len(kd_cfg["alphas"]) * len(kd_cfg["temperatures"])
        n_tune_max = (
            len(self.manifest["teacher_gate"]["teacher_tuning"]["lr_values"])
            * len(self.manifest["teacher_gate"]["teacher_tuning"]["weight_decay_values"])
            * len(self.manifest["teacher_gate"]["teacher_tuning"]["grad_clip_values"])
        )
        self.log(
            "Planned runs (single-stage): "
            f"phase1_full={n_base+n_cbam}, phase2_full={n_kd}, teacher_tuning_max={n_tune_max}"
        )

    def run(self) -> int:
        if self.fresh_start:
            self.clean_outputs()
        else:
            self._refresh_progress_report()
        self.print_plan_summary()

        if self.dry_run:
            phase1_full_specs = self.build_phase1_full_specs()
            for spec in phase1_full_specs[:3]:
                self.log(f"DRY-RUN CMD: {' '.join(spec.cmd)}")
            self.log("DRY-RUN finished. Cache checks, gate checks, and executions were skipped.")
            return 0

        self.ensure_cache()

        phase1_full_specs = self.build_phase1_full_specs()
        phase1_selected = self.select_phase1_full_specs(phase1_full_specs)
        for spec in phase1_selected:
            self.run_spec(spec)

        best_student_run, best_student_f1 = self.pick_best_student_f1()
        self.log(f"Best student candidate after phase1: {best_student_run} f1={best_student_f1:.4f}")
        teacher_run, teacher_f1, gate_ok = self.run_teacher_tuning_if_needed(best_student_f1)
        self.log(f"Teacher selected: {teacher_run} f1={teacher_f1:.4f} gate_ok={gate_ok}")
        if self.kd_after_teacher_gate_only and not gate_ok:
            self.log("KD gate not satisfied (teacher-student gap below threshold). KD stage aborted.")
            return 2

        teacher_checkpoint = str(self.run_root / teacher_run / "checkpoints" / "best_model.pth")
        if not Path(teacher_checkpoint).exists():
            raise FileNotFoundError(f"Teacher checkpoint missing: {teacher_checkpoint}")

        kd_full_specs = self.build_phase2_kd_full_specs(teacher_checkpoint)
        kd_selected = self.select_phase2_kd_full_specs(kd_full_specs)
        for spec in kd_selected:
            self.run_spec(spec)

        self.log("All phases completed successfully.")
        return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=str,
        default="experiments/manifest_repro_v1.json",
        help="Path to experiment manifest JSON.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs that already satisfy artifact contract.",
    )
    parser.add_argument(
        "--fresh-start",
        action="store_true",
        help="Delete old logs/results (except cache) before running.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan without executing runs.")
    args = parser.parse_args()

    runner = ManifestRunner(
        manifest_path=Path(args.manifest),
        resume=args.resume,
        dry_run=args.dry_run,
        fresh_start=args.fresh_start,
    )
    code = runner.run()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
