"""Make a static scientific summary figure from the sweep CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .sweep import CONDITIONS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    path = args.directory / "summary_aggregate.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    arms = ["cx_fixed", "direct_cx", "mb_cx", "mb_shuffled"]
    colors = ["#8d99ae", "#4c78a8", "#e45756", "#9d755d"]
    x = np.arange(len(CONDITIONS))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    metrics = [
        ("heading_mae_sectors", "Heading error (sectors)", None),
        ("goal_accuracy", "Goal accuracy", (0, 1.03)),
        ("steering_accuracy", "Turn decision accuracy", (0, 1.03)),
    ]
    for ax, (metric, ylabel, ylim) in zip(axes, metrics):
        plotted_arms = [("cx_fixed", "#333333")] if metric == "heading_mae_sectors" else list(zip(arms, colors))
        for arm, color in plotted_arms:
            selected = [next(r for r in rows if r["condition"] == condition and r["arm"] == arm)
                        for condition in CONDITIONS]
            means = [float(r[f"{metric}_mean"]) for r in selected]
            sds = [float(r[f"{metric}_sd_across_seeds"]) for r in selected]
            ax.errorbar(x, means, yerr=sds, marker="o", linewidth=1.5,
                        color=color, capsize=2, label="shared EB" if metric == "heading_mae_sectors" else arm)
        ax.set_xticks(x, [s.replace("_", "\n") for s in CONDITIONS], fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(axis="y", alpha=0.25)
        if ylim:
            ax.set_ylim(*ylim)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("Minimal spiking MB/CX synthetic pilot — mean ± SD across seeds", fontsize=11)
    output = args.directory / "baseline_summary.png"
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(output.resolve())


if __name__ == "__main__":
    main()
