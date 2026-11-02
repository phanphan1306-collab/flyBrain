"""Plot validation sparsity curves from the four-family comparison CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


EXPANSION_ARMS = ("mb_sparse_t5_cx", "mb_sparse_t4_cx",
                  "mb_sparse_t3_cx", "dense_expansion_cx")
CONDITIONS = ("standard", "cue_corruption")
COLORS = {"standard": "#2672a8", "cue_corruption": "#d65a31"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    with (args.directory / "summary_aggregate.csv").open(newline="", encoding="utf-8") as stream:
        records = {(row["condition"], row["arm"]): row for row in csv.DictReader(stream)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, (metric, title) in zip(axes, (("goal_accuracy", "Goal accuracy"),
                                          ("steering_accuracy", "Turn accuracy"))):
        for condition in CONDITIONS:
            points = [records[(condition, arm)] for arm in EXPANSION_ARMS]
            x = [100 * float(row["active_input_fraction_mean"]) for row in points]
            y = [float(row[f"{metric}_mean"]) for row in points]
            sd = [float(row[f"{metric}_sd_across_seeds"]) for row in points]
            color = COLORS[condition]
            ax.errorbar(x, y, yerr=sd, color=color, marker="o", capsize=2,
                        label=f"Expansion: {condition}")
            direct = records[(condition, "direct_cx")]
            ax.axhline(float(direct[f"{metric}_mean"]), color=color, alpha=0.55,
                       linestyle="--", label=f"Direct: {condition}")
            if condition == "standard":
                for xx, yy, arm in zip(x, y, EXPANSION_ARMS):
                    label = "t1" if arm == "dense_expansion_cx" else "t" + arm.split("_t")[1].split("_")[0]
                    offset = (-20, 5) if arm == "dense_expansion_cx" else (3, 5)
                    ax.annotate(label, (xx, yy), xytext=offset,
                                textcoords="offset points", fontsize=8)
        ax.set(xlabel="Active expansion neurons (%)", ylabel="Accuracy", title=title,
               xlim=(-2, 100), ylim=(0, 1.05))
        ax.grid(alpha=0.2)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Validation seeds 200–204: mean ± SD across seeds", fontsize=11)
    path = args.directory / "validation_sparsity_curves.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(path.resolve())


if __name__ == "__main__":
    main()
