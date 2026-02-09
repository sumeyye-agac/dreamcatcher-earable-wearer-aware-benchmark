from __future__ import annotations

import csv
import os
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime

from src.utils.benchmarking import _file_lock
from src.utils.csv_schemas import RUN_STEPS_FIELDNAMES


def _now_utc_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class StepLogger:
    """
    Minimal run timeline logger.

    - Prints to stderr (so it shows in terminal even when stdout is used for other logs).
    - Appends a row to a CSV (default: results/run_steps.csv) with file locking for concurrent safety.
    """

    run_name: str
    csv_path: str = "results/run_steps.csv"

    def log(self, stage: str, detail: str = "", t0: float | None = None) -> None:
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)

        dt_s: str = ""
        if t0 is not None:
            dt_s = f"{time.time() - t0:.2f}"

        line = f"[{_now_utc_iso()}] stage={stage}"
        if self.run_name:
            line += f" run={self.run_name}"
        if dt_s:
            line += f" dt_s={dt_s}"
        if detail:
            line += f" | {detail}"
        print(line, file=sys.stderr, flush=True)

        # Use file lock for concurrent-safe CSV writing
        lock_path = self.csv_path + ".lock"
        with _file_lock(lock_path):
            file_exists = os.path.exists(self.csv_path)
            with open(self.csv_path, "a", newline="") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=RUN_STEPS_FIELDNAMES,
                )
                if not file_exists:
                    w.writeheader()
                w.writerow(
                    {
                        "ts_utc": _now_utc_iso(),
                        "run_name": self.run_name,
                        "stage": stage,
                        "dt_s": dt_s,
                        "detail": detail,
                    }
                )
