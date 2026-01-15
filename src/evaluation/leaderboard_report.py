from __future__ import annotations

import argparse
import pandas as pd


LEADERBOARD_COLUMNS = [
    "run_name",
    "task",
    "model",
    "teacher",
    "seed",
    "epochs",
    "batch_size",
    "lr",
    "sr",
    "n_mels",
    "rnn_hidden",
    "rnn_layers",
    "cbam_reduction",
    "cbam_sa_kernel",
    "alpha",
    "tau",
    "best_val_f1",
    "test_acc",
    "test_f1",
    "params",
    "cpu_latency_ms",
]


def read_leaderboard_csv(path: str) -> pd.DataFrame:
    """
    Read `results/leaderboard.csv`.

    Supports both:
    - headered CSVs (recommended)
    - legacy/headerless CSVs with the exact schema written by `append_to_leaderboard`
    """
    df = pd.read_csv(path)
    if "run_name" not in df.columns and df.shape[1] == len(LEADERBOARD_COLUMNS):
        df.columns = LEADERBOARD_COLUMNS
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="results/leaderboard.csv")
    parser.add_argument("--metric", type=str, default="test_f1")
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    df = read_leaderboard_csv(args.csv)
    if args.metric not in df.columns:
        raise KeyError(
            f"Metric column '{args.metric}' not found in {args.csv}. "
            f"Columns: {list(df.columns)}"
        )
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
