from __future__ import annotations

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="results/leaderboard.csv")
    parser.add_argument("--metric", type=str, default="test_f1")
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df[args.metric] = pd.to_numeric(df[args.metric], errors="coerce")
    df = df.dropna(subset=[args.metric]).sort_values(args.metric, ascending=False)

    cols = [
        "run_name", "model", "teacher", "test_f1", "test_acc", "params", "cpu_latency_ms",
        "alpha", "tau", "cbam_reduction", "cbam_sa_kernel"
    ]
    cols = [c for c in cols if c in df.columns]

    print(df.head(args.topk)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
