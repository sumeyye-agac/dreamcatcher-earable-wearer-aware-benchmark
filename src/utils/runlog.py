from __future__ import annotations

import csv
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass(frozen=True)
class StepLogger:
    """
    Minimal run timeline logger.

    - Prints to stderr (so it shows in terminal even when stdout is used for other logs).
    - Appends a row to a CSV (default: results/run_steps.csv).
    """

    run_name: str
    csv_path: str = "results/run_steps.csv"

    def log(self, stage: str, detail: str = "", t0: float | None = None) -> None:
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)

        dt_s: str = ""
        if t0 is not None:
            dt_s = f"{time.time() - t0:.2f}"

        line = f"[{_now_iso()}] stage={stage}"
        if self.run_name:
            line += f" run={self.run_name}"
        if dt_s:
            line += f" dt_s={dt_s}"
        if detail:
            line += f" | {detail}"
        print(line, file=sys.stderr, flush=True)

        file_exists = os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["ts", "run_name", "stage", "dt_s", "detail"],
            )
            if not file_exists:
                w.writeheader()
            w.writerow(
                {
                    "ts": _now_iso(),
                    "run_name": self.run_name,
                    "stage": stage,
                    "dt_s": dt_s,
                    "detail": detail,
                }
            )
