"""Predeclared robustness conditions for the minimal synthetic baseline."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .benchmark import run_seed, summarize, write_csv
from .model import Config


CONDITIONS = {
    "clean": dict(cue_flip_probability=0.0, landmark_dropout=0.0,
                  landmark_noise_probability=0.0, turn_noise_probability=0.0),
    "standard": dict(cue_flip_probability=0.08, landmark_dropout=0.35,
                     landmark_noise_probability=0.10, turn_noise_probability=0.10),
    "cue_corruption": dict(cue_flip_probability=0.22, landmark_dropout=0.35,
                           landmark_noise_probability=0.10, turn_noise_probability=0.10),
    "missing_landmarks": dict(cue_flip_probability=0.08, landmark_dropout=0.8,
                              landmark_noise_probability=0.10, turn_noise_probability=0.20),
    "combined": dict(cue_flip_probability=0.22, landmark_dropout=0.8,
                     landmark_noise_probability=0.20, turn_noise_probability=0.20),
}


def run_sweep(seeds: list[int]) -> tuple[list[dict], list[dict]]:
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("seeds must be nonempty and unique")
    raw: list[dict] = []
    per_seed: list[dict] = []
    for condition, overrides in CONDITIONS.items():
        cfg = replace(Config(), **overrides)
        for seed in seeds:
            rows = run_seed(cfg, seed)
            for row in rows:
                row["condition"] = condition
            raw.extend(rows)
            for item in summarize(rows):
                item["condition"] = condition
                per_seed.append(item)
    return raw, per_seed


def aggregate(per_seed: list[dict]) -> list[dict]:
    output = []
    for condition in CONDITIONS:
        for arm in ("cx_fixed", "direct_cx", "mb_cx", "mb_shuffled"):
            group = [x for x in per_seed if x["condition"] == condition and x["arm"] == arm]
            record = {"condition": condition, "arm": arm, "n_seeds": len(group)}
            for metric in ("heading_mae_sectors", "goal_accuracy", "steering_accuracy"):
                values = np.asarray([x[metric] for x in group], dtype=float)
                record[f"{metric}_mean"] = float(values.mean())
                record[f"{metric}_sd_across_seeds"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            output.append(record)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/fly_snn_sweep"))
    parser.add_argument("--first-seed", type=int, default=100)
    parser.add_argument("--n-seeds", type=int, default=20)
    args = parser.parse_args()
    if args.n_seeds < 2:
        parser.error("n-seeds must be >= 2")
    seeds = list(range(args.first_seed, args.first_seed + args.n_seeds))
    args.output.mkdir(parents=True, exist_ok=False)
    raw, per_seed = run_sweep(seeds)
    aggregate_rows = aggregate(per_seed)
    write_csv(args.output / "trials.csv", raw)
    write_csv(args.output / "summary_by_seed.csv", per_seed)
    write_csv(args.output / "summary_aggregate.csv", aggregate_rows)
    (args.output / "manifest.json").write_text(json.dumps({
        "status": "synthetic_pilot_not_paper_result",
        "conditions": CONDITIONS,
        "seeds": seeds,
        "config_other_fields": Config().to_dict(),
        "comparison": "paired arms per episode; independent seeds are aggregation units",
        "note": "No hyperparameter selection on these outputs; do not claim published-model replication.",
    }, indent=2), encoding="utf-8")
    print(json.dumps(aggregate_rows, indent=2))
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
