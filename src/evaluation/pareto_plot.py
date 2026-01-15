from __future__ import annotations

import argparse
import pandas as pd
import matplotlib.pyplot as plt


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


def pareto_front(df, x_col: str, y_col: str, maximize_y: bool = True):
    """
    Computes Pareto front for (x=cost, y=score).
    For each x (lower is better), keep points not dominated in y.
    """
    df = df.dropna(subset=[x_col, y_col]).copy()
    df = df.sort_values(x_col, ascending=True)

    best_y = None
    keep = []
    for _, row in df.iterrows():
        y = row[y_col]
        if best_y is None:
            best_y = y
            keep.append(True)
        else:
            if maximize_y:
                if y >= best_y:
                    best_y = y
                    keep.append(True)
                else:
                    keep.append(False)
            else:
                if y <= best_y:
                    best_y = y
                    keep.append(True)
                else:
                    keep.append(False)

    return df[keep]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="results/leaderboard.csv")
    parser.add_argument("--x", type=str, default="cpu_latency_ms", help="cost axis (lower is better)")
    parser.add_argument("--y", type=str, default="test_f1", help="score axis (higher is better)")
    parser.add_argument("--out", type=str, default="results/plots/pareto_front.png")
    parser.add_argument("--title", type=str, default="Pareto Frontier: F1 vs Latency")
    args = parser.parse_args()

    df = read_leaderboard_csv(args.csv)

    # Ensure numeric
    df[args.x] = pd.to_numeric(df[args.x], errors="coerce")
    df[args.y] = pd.to_numeric(df[args.y], errors="coerce")

    pf = pareto_front(df, x_col=args.x, y_col=args.y, maximize_y=True)

    # Plot
    plt.figure()
    plt.scatter(df[args.x], df[args.y])
    plt.scatter(pf[args.x], pf[args.y])

    # annotate Pareto points
    for _, r in pf.iterrows():
        plt.annotate(str(r.get("run_name", "")), (r[args.x], r[args.y]))

    plt.xlabel(args.x)
    plt.ylabel(args.y)
    plt.title(args.title)

    # Make sure output directory exists
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    plt.savefig(args.out, bbox_inches="tight", dpi=200)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
