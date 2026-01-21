"""
Visualization script for class distribution in the DreamCatcher dataset.
Generates bar chart and pie chart showing how data is distributed across classes.
Uses confusion matrix and run_steps.csv to derive dataset split sizes.
"""

import csv
import re
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


# Class labels from DreamCatcher
LABELS = [
    "quiet",
    "non_wearer",
    "bruxism",
    "swallow",
    "somniloquy",
    "breathe",
    "cough",
    "snore",
    "movements",
]


def load_class_counts_from_confusion_matrix(cm_path):
    """
    Load class counts from confusion matrix CSV.
    Counts are taken from the row sums (total samples per class).
    """
    class_names = []
    class_counts = []

    with open(cm_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header

        for row in reader:
            if len(row) > 0:
                class_name = row[0]
                # Sum all predictions for this true class
                true_count = sum(int(x) for x in row[1:] if x.strip().isdigit())

                if true_count > 0:  # Only include if there are samples
                    class_names.append(class_name)
                    class_counts.append(true_count)

    return class_names, class_counts


def extract_split_sizes_from_runsteps(runsteps_csv):
    """
    Extract train/val/test split sizes from run_steps.csv
    Looks for lines like: "dataset_split_loaded,0.09,split=test len=76991 max_samples="
    """
    splits = {}

    with open(runsteps_csv, "r") as f:
        reader = csv.DictReader(f, fieldnames=["ts", "run_name", "stage", "dt_s", "detail"])

        for row in reader:
            if row["stage"] == "dataset_split_loaded":
                detail = row["detail"]
                # Extract split name and length
                split_match = re.search(r"split=(\w+)", detail)
                len_match = re.search(r"len=(\d+)", detail)

                if split_match and len_match:
                    split_name = split_match.group(1)
                    split_len = int(len_match.group(1))

                    # Normalize split names
                    if split_name == "validation":
                        split_name = "val"
                    splits[split_name] = split_len

    return splits


def estimate_class_distribution_for_splits(test_class_counts, test_class_names, split_sizes):
    """
    Estimate total class distribution across all splits.
    Assumes similar class distribution across splits.

    Returns:
        dict with total distribution
    """
    test_total = sum(test_class_counts)
    total_counts = Counter()

    # Add test counts
    for name, count in zip(test_class_names, test_class_counts):
        total_counts[name] += count

    # Estimate train counts based on split ratio
    if "train" in split_sizes:
        train_total = split_sizes["train"]
        train_ratio = train_total / test_total
        for name, count in zip(test_class_names, test_class_counts):
            total_counts[name] += int(count * train_ratio)

    # Estimate val counts based on split ratio
    if "val" in split_sizes:
        val_total = split_sizes["val"]
        val_ratio = val_total / test_total
        for name, count in zip(test_class_names, test_class_counts):
            total_counts[name] += int(count * val_ratio)

    return {"total": dict(total_counts)}


def create_visualizations(distributions, output_dir, split_name="total"):
    """
    Create bar chart and pie chart visualizations.

    Args:
        distributions: Dict with class distributions
        output_dir: Directory to save PNG files
        split_name: Which split to visualize ("train", "val", "test", "total")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts_dict = distributions[split_name]

    # Sort by class name in the order defined in LABELS
    class_names = [name for name in LABELS if name in counts_dict]
    class_counts = [counts_dict[name] for name in class_names]

    total_samples = sum(class_counts)
    percentages = [100 * count / total_samples for count in class_counts]

    # Create bar chart
    fig, ax = plt.subplots(figsize=(14, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    bars = ax.bar(
        range(len(class_names)), class_counts, color=colors, edgecolor="black", linewidth=1.5
    )

    ax.set_ylabel("Number of Samples", fontsize=14, fontweight="bold")
    ax.set_title(
        f"DreamCatcher Dataset - Class Distribution (All Splits Combined)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Add value labels on bars - only counts, no class names
    for bar, count, pct in zip(bars, class_counts, percentages):
        height = bar.get_height()
        label_text = f"{int(count):,}\n({pct:.2f}%)"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            label_text,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Use numbers on x-axis, legend on the side
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels([str(i + 1) for i in range(len(class_names))], fontsize=12)
    ax.set_xlabel("Class #", fontsize=14, fontweight="bold")

    # Add legend on the right side with class names
    legend_labels = [f"{i + 1}. {name}" for i, name in enumerate(class_names)]
    ax.legend(
        bars,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=11,
        framealpha=0.95,
        title="Classes",
        title_fontsize=12,
    )

    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0, max(class_counts) * 1.12)
    plt.yticks(fontsize=11)
    plt.tight_layout()

    bar_filename = f"class_distribution_bar_{split_name}.png"
    bar_chart_path = output_dir / bar_filename
    plt.savefig(bar_chart_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Bar chart saved: {bar_chart_path}")
    plt.close()

    # Create pie chart
    fig, ax = plt.subplots(figsize=(14, 10))

    # Sort by count for better visualization
    sorted_indices = np.argsort(class_counts)[::-1]
    sorted_names = [class_names[i] for i in sorted_indices]
    sorted_counts = [class_counts[i] for i in sorted_indices]
    sorted_percentages = [percentages[i] for i in sorted_indices]

    colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_names)))

    # Create pie without labels or percentages on chart
    wedges, texts = ax.pie(
        sorted_counts,
        startangle=90,
        colors=colors,
        radius=0.7,  # Make pie larger
    )

    # Add legend below the pie with percentages and counts
    legend_labels = [
        f"{i + 1}. {name}: {pct:.1f}% ({count:,})"
        for i, (name, count, pct) in enumerate(
            zip(sorted_names, sorted_counts, [100 * c / sum(sorted_counts) for c in sorted_counts])
        )
    ]
    ax.legend(
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=3,
        fontsize=10,
        framealpha=0.95,
        title="Classes",
        title_fontsize=11,
    )

    ax.set_title(
        f"DreamCatcher Dataset - Class Distribution (%)", fontsize=16, fontweight="bold", pad=20
    )

    pie_filename = f"class_distribution_pie_{split_name}.png"
    pie_chart_path = output_dir / pie_filename
    plt.savefig(pie_chart_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Pie chart saved: {pie_chart_path}")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize DreamCatcher class distribution")
    parser.add_argument(
        "--confusion-matrix", default=None, help="Path to confusion matrix CSV file"
    )
    parser.add_argument("--run-steps", default=None, help="Path to run_steps.csv file")
    parser.add_argument("--output-dir", default=None, help="Output directory for visualizations")

    args = parser.parse_args()

    # Determine default paths
    repo_root = Path(__file__).parent.parent.parent
    confusion_matrix_path = (
        Path(args.confusion_matrix)
        if args.confusion_matrix
        else repo_root / "results" / "runs" / "crnn_baseline" / "test_confusion_matrix.csv"
    )
    run_steps_path = (
        Path(args.run_steps) if args.run_steps else repo_root / "results" / "run_steps.csv"
    )
    output_directory = (
        Path(args.output_dir) if args.output_dir else repo_root / "results" / "visualizations"
    )

    print("\n" + "=" * 70)
    print("DreamCatcher Dataset Class Distribution Visualization")
    print("=" * 70)

    if not confusion_matrix_path.exists():
        print(f"\n✗ Error: Confusion matrix not found at {confusion_matrix_path}")
        exit(1)

    # Load test data distribution from confusion matrix
    print(f"\nLoading test split from: {confusion_matrix_path}")
    test_class_names, test_class_counts = load_class_counts_from_confusion_matrix(
        confusion_matrix_path
    )
    print(f"  ✓ Loaded {sum(test_class_counts):,} test samples")

    # Load split sizes from run_steps
    split_sizes = {}
    if run_steps_path.exists():
        print(f"\nLoading split sizes from: {run_steps_path}")
        split_sizes = extract_split_sizes_from_runsteps(run_steps_path)
        if split_sizes:
            print(f"  ✓ Found split sizes: {split_sizes}")
        else:
            print(f"  ⚠ Could not extract split sizes from run_steps.csv")

    # Estimate total distribution across all splits
    if split_sizes:
        distributions = estimate_class_distribution_for_splits(
            test_class_counts, test_class_names, split_sizes
        )
        print("\nTotal class distribution (train + val + test):")
        for class_name in sorted(
            distributions["total"].keys(), key=lambda x: distributions["total"][x], reverse=True
        ):
            count = distributions["total"][class_name]
            pct = 100 * count / sum(distributions["total"].values())
            print(f"  {class_name:20s}: {count:8,} samples ({pct:6.2f}%)")
    else:
        # Only use test data
        distributions = {
            "total": {name: count for name, count in zip(test_class_names, test_class_counts)}
        }
        print("\n⚠ No split size information available. Using test set only.")

    # Create visualizations (only total - 2 PNGs)
    print("\nGenerating visualizations...")
    create_visualizations(distributions, output_directory, split_name="total")

    print(f"\n✓ Visualizations saved to: {output_directory}")
    print(f"  - class_distribution_bar_total.png")
    print(f"  - class_distribution_pie_total.png")
