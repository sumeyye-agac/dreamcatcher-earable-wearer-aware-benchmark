#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "experiments" / "manifest_repro_v1.json"
README_PATH = REPO_ROOT / "README.md"
DECISION_LOG_PATH = REPO_ROOT / "docs" / "decision_log.md"


def fmt_policy_float(x: float) -> str:
    f = float(x)
    if f.is_integer():
        return f"{f:.1f}"
    return f"{f:g}"


def normalize_csv_number_list(raw: str) -> str:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return ",".join(parts)


def build_manifest_policy(manifest: dict) -> dict[str, str]:
    baseline = manifest["baseline"]
    kd = manifest["kd"]

    phase1_total = len(baseline["models"]) + (
        len(baseline["cbam_models"])
        * len(baseline["cbam_reduction_values"])
        * len(baseline["cbam_sa_kernel_values"])
    )
    kd_total = len(kd["students"]) * len(kd["alphas"]) * len(kd["temperatures"])
    kd_alphas = ",".join(fmt_policy_float(x) for x in kd["alphas"])
    kd_taus = ",".join(fmt_policy_float(x) for x in kd["temperatures"])

    return {
        "phase1_total": str(phase1_total),
        "kd_total": str(kd_total),
        "kd_alphas": kd_alphas,
        "kd_taus": kd_taus,
    }


def parse_readme_consistency_block(readme_text: str) -> dict[str, str]:
    m = re.search(r"<!--\s*consistency:\s*(.*?)\s*-->", readme_text)
    if not m:
        raise ValueError("README consistency marker not found.")

    out: dict[str, str] = {}
    tokens = [t.strip() for t in m.group(1).split() if t.strip()]
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def extract_current_pinned_policy_block(decision_log_text: str) -> str:
    header = "## Current Pinned Policy"
    start = decision_log_text.find(header)
    if start < 0:
        raise ValueError("Decision log 'Current Pinned Policy' section not found.")

    next_header = decision_log_text.find("\n## ", start + 1)
    if next_header < 0:
        return decision_log_text[start:]
    return decision_log_text[start:next_header]


def parse_decision_log_policy(block_text: str) -> dict[str, str]:
    phase1_m = re.search(r"=\s*(\d+)\s*runs", block_text)
    if not phase1_m:
        raise ValueError("Phase-1 total not found in decision log pinned policy.")

    kd_m = re.search(
        r"alpha=\[([^\]]+)\].*temperature=\[([^\]]+)\].*(?:->|→)\s*(\d+)\s+combinations",
        block_text,
        flags=re.S,
    )
    if not kd_m:
        raise ValueError("KD grid line not found in decision log pinned policy.")

    return {
        "phase1_total": phase1_m.group(1).strip(),
        "kd_total": kd_m.group(3).strip(),
        "kd_alphas": normalize_csv_number_list(kd_m.group(1)),
        "kd_taus": normalize_csv_number_list(kd_m.group(2)),
    }


def compare_policy(
    manifest_policy: dict[str, str],
    readme_policy: dict[str, str],
    decision_policy: dict[str, str],
) -> list[str]:
    issues: list[str] = []
    for key, expected in manifest_policy.items():
        rv = readme_policy.get(key)
        dv = decision_policy.get(key)
        if rv != expected:
            issues.append(f"README mismatch for {key}: expected={expected}, found={rv}")
        if dv != expected:
            issues.append(f"decision_log mismatch for {key}: expected={expected}, found={dv}")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Check manifest/docs policy consistency.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero on any mismatch.",
    )
    args = parser.parse_args()

    issues: list[str] = []

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    readme_text = README_PATH.read_text(encoding="utf-8")
    decision_text = DECISION_LOG_PATH.read_text(encoding="utf-8")

    manifest_policy = build_manifest_policy(manifest)
    readme_policy = parse_readme_consistency_block(readme_text)
    decision_block = extract_current_pinned_policy_block(decision_text)
    decision_policy = parse_decision_log_policy(decision_block)

    if "0.0" in [fmt_policy_float(a) for a in manifest["kd"]["alphas"]]:
        issues.append("Manifest KD alpha grid still contains 0.0.")

    if "Phase-2) are in progress" in readme_text:
        issues.append("README still says Phase-2 is in progress.")

    issues.extend(compare_policy(manifest_policy, readme_policy, decision_policy))

    print("Manifest policy:", manifest_policy)
    print("README policy:", readme_policy)
    print("Decision policy:", decision_policy)

    if issues:
        print("\nConsistency check: FAIL")
        for issue in issues:
            print(f"- {issue}")
        return 1 if args.strict else 0

    print("\nConsistency check: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
