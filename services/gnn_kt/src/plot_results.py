"""
Temporal Prerequisite-Aware Graph Knowledge Tracing (TPA-GKT)
Publication Figures Generator (plot_results.py)

Generates clean, 300 DPI publication-ready figures from experimental results:
  1. roc_comparison.png: Baseline vs. TPA-GKT ROC-AUC and Macro-F1 comparison.
  2. retention_by_delay_bucket.png: Delayed-retention AUC and Brier score across
     [<1h, 1h-24h, 1d-7d, >7d] time intervals.
  3. calibration_reliability.png: 10-bin ECE Reliability Diagram.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Add project root to sys.path
_project_root = str(Path(__file__).resolve().parents[3])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Styling configuration
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


def plot_baseline_comparison(
    table3_csv_path: Path,
    output_png_path: Path,
) -> None:
    """Plot comparative bar chart of baselines vs TPA-GKT."""
    if not table3_csv_path.exists():
        logger.warning("Table 3 CSV not found at %s. Skipping plot.", table3_csv_path)
        return

    models: list[str] = []
    auc_means: list[float] = []
    f1_means: list[float] = []

    with open(table3_csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            models.append(row["Model"])

            # Parse mean value (e.g., '0.7393' or '0.7393 ± 0.0050')
            auc_str = row["Next-Response AUC"].split("±")[0].strip()
            auc_means.append(float(auc_str) if auc_str != "N/A" else 0.0)

            # Try Macro-F1 or Binary-F1
            f1_col = (
                "Next-Response Macro-F1"
                if "Next-Response Macro-F1" in row
                else "Next-Response F1"
            )
            f1_str = row[f1_col].split("±")[0].strip()
            f1_means.append(float(f1_str) if f1_str != "N/A" else 0.0)

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    rects1 = ax.bar(
        x - width / 2, auc_means, width, label="Next-Response ROC-AUC", color="#2563EB"
    )
    rects2 = ax.bar(
        x + width / 2, f1_means, width, label="Next-Response Macro-F1", color="#10B981"
    )

    ax.set_ylabel("Metric Value")
    ax.set_title("Model Benchmarking on Junyi Academy 2020", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right")

    # Add numeric labels above bars
    def autolabel(rects: Any) -> None:
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png_path)
    plt.close(fig)
    logger.info("Saved baseline comparison plot to %s", output_png_path)


def plot_retention_by_bucket(
    buckets_csv_path: Path,
    output_png_path: Path,
) -> None:
    """Plot delayed retention performance across delay buckets."""
    if not buckets_csv_path.exists():
        logger.warning("Retention buckets CSV not found at %s. Skipping plot.", buckets_csv_path)
        return

    buckets: list[str] = []
    aucs: list[float] = []
    briers: list[float] = []
    counts: list[int] = []

    with open(buckets_csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            buckets.append(row["Bucket"])
            auc_str = row["TPA-GKT Retention AUC"].split("±")[0].strip()
            aucs.append(float(auc_str) if auc_str != "N/A" else 0.0)

            brier_str = row["Retention Brier"].split("±")[0].strip()
            briers.append(float(brier_str) if brier_str != "N/A" else 0.0)

            counts.append(int(row["Sample Count"]))

    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)

    color_auc = "#2563EB"
    ax1.set_xlabel("Retention Delay Bucket (Study Interval)")
    ax1.set_ylabel("Retention ROC-AUC", color=color_auc, fontweight="bold")
    line1 = ax1.plot(
        buckets, aucs, color=color_auc, marker="o", linewidth=2.5, markersize=8,
        label="Retention ROC-AUC"
    )
    ax1.tick_params(axis="y", labelcolor=color_auc)
    ax1.set_ylim(0.5, 0.9)

    # Secondary axis for Brier score
    ax2 = ax1.twinx()
    color_brier = "#DC2626"
    ax2.set_ylabel("Brier Score (Lower is Better)", color=color_brier, fontweight="bold")
    line2 = ax2.plot(
        buckets, briers, color=color_brier, marker="s", linestyle="--", linewidth=2.0,
        markersize=7, label="Brier Score"
    )
    ax2.tick_params(axis="y", labelcolor=color_brier)
    ax2.set_ylim(0.0, 0.3)

    # Title & annotations
    plt.title("TPA-GKT Delayed Retention Modeling Across Time Gaps", fontweight="bold")
    for i, count in enumerate(counts):
        ax1.annotate(
            f"N={count:,}\nAUC={aucs[i]:.3f}",
            xy=(i, aucs[i]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
        )

    lines = line1 + line2
    labels = [line_obj.get_label() for line_obj in lines]
    ax1.legend(lines, labels, loc="center left")

    fig.tight_layout()
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png_path)
    plt.close(fig)
    logger.info("Saved retention bucket plot to %s", output_png_path)


def plot_calibration_reliability(
    summary_json_path: Path,
    output_png_path: Path,
) -> None:
    """Plot 10-bin reliability diagram from evaluation summary."""
    if not summary_json_path.exists():
        logger.warning("Summary JSON not found at %s. Skipping plot.", summary_json_path)
        return

    with open(summary_json_path, encoding="utf-8") as f:
        data = json.load(f)

    # Extract TPA-GKT full results (supports both flat and nested 'runs' format)
    runs = data.get("runs", data)
    tpa_full = runs.get("tpa_gkt_full", {})
    if not tpa_full:
        logger.warning("TPA-GKT full results not found in summary JSON.")
        return

    first_seed = next(iter(tpa_full.values()))
    bins = first_seed.get("reliability_diagram", [])
    ece = first_seed.get("response_ece", 0.0)

    if not bins:
        logger.warning("Reliability diagram data empty. Skipping plot.")
        return

    confs = [b["confidence"] for b in bins if b["count"] > 0]
    accs = [b["accuracy"] for b in bins if b["count"] > 0]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    # Perfect calibration reference diagonal
    ax.plot([0, 1], [0, 1], linestyle="--", color="#64748B", label="Perfect Calibration")

    # Empirical reliability curve
    label_txt = f"TPA-GKT (ECE = {ece:.3f})"
    ax.plot(confs, accs, marker="o", color="#2563EB", linewidth=2.5, markersize=7, label=label_txt)
    ax.bar(confs, accs, width=0.06, alpha=0.2, color="#3B82F6")

    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Empirical Accuracy")
    ax.set_title("Next-Response Calibration (Reliability Diagram)", fontweight="bold")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right")

    fig.tight_layout()
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png_path)
    plt.close(fig)
    logger.info("Saved calibration reliability plot to %s", output_png_path)


def generate_all_plots(
    results_dir: str | Path = "services/gnn_kt/outputs/results",
) -> None:
    """Generate the full suite of publication figures."""
    r_dir = Path(results_dir)
    fig_dir = r_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_baseline_comparison(
        table3_csv_path=r_dir / "table3_results.csv",
        output_png_path=fig_dir / "roc_comparison.png",
    )
    plot_retention_by_bucket(
        buckets_csv_path=r_dir / "retention_buckets.csv",
        output_png_path=fig_dir / "retention_by_delay_bucket.png",
    )
    plot_calibration_reliability(
        summary_json_path=r_dir / "all_results_summary.json",
        output_png_path=fig_dir / "calibration_reliability.png",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TPA-GKT Publication Plotter")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="services/gnn_kt/outputs/results",
        help="Path to directory containing CSVs and summary JSON",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_all_plots(results_dir=args.results_dir)
