from __future__ import annotations

import argparse
import pandas as pd


def to_md_table(df: pd.DataFrame) -> str:
    # Pandas markdown requires tabulate in some setups; avoid dependency.
    # We'll render a simple GitHub-flavored markdown table manually.
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join([header, sep] + rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="results/leaderboard.csv")
    parser.add_argument("--out", type=str, default="results/top5.md")
    parser.add_argument("--metric", type=str, default="test_f1")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # Ensure numeric
    df[args.metric] = pd.to_numeric(df[args.metric], errors="coerce")
    df = df.dropna(subset=[args.metric]).sort_values(args.metric, ascending=False)

    # Select columns that matter most for a quick snapshot
    preferred_cols = [
        "run_name",
        "model",
        "teacher",
        "test_f1",
        "test_acc",
        "params",
        "cpu_latency_ms",
    ]
    cols = [c for c in preferred_cols if c in df.columns]

    top = df.head(args.topk)[cols].copy()

    # Make floats readable
    for c in ["test_f1", "test_acc", "cpu_latency_ms"]:
        if c in top.columns:
            top[c] = top[c].map(lambda x: f"{float(x):.4f}")

    md = to_md_table(top)

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md + "\n")

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
