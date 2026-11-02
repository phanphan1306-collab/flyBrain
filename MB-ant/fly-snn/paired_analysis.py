"""Seed-paired MB-vs-direct differences; descriptive pilot intervals only."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .benchmark import write_csv
from .sweep import CONDITIONS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    with (args.directory / "summary_by_seed.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    results = []
    for condition in CONDITIONS:
        group = [r for r in rows if r["condition"] == condition]
        mb = {int(r["seed"]): r for r in group if r["arm"] == "mb_cx"}
        direct = {int(r["seed"]): r for r in group if r["arm"] == "direct_cx"}
        if set(mb) != set(direct):
            raise ValueError(f"Unpaired seeds in {condition}")
        for metric in ("goal_accuracy", "steering_accuracy"):
            differences = np.asarray([float(mb[s][metric]) - float(direct[s][metric])
                                      for s in sorted(mb)], dtype=float)
            rng = np.random.default_rng(6_500 + list(CONDITIONS).index(condition))
            samples = rng.choice(differences, size=(5_000, len(differences)), replace=True)
            ci = np.quantile(samples.mean(axis=1), [0.025, 0.975])
            results.append({
                "condition": condition, "metric": metric,
                "comparison": "mb_cx_minus_direct_cx", "n_seeds": len(differences),
                "mean_paired_delta": float(differences.mean()),
                "bootstrap_95_low": float(ci[0]),
                "bootstrap_95_high": float(ci[1]),
            })
    output = args.directory / "paired_deltas.csv"
    write_csv(output, results)
    print(output.resolve())


if __name__ == "__main__":
    main()
