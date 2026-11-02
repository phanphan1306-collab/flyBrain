"""Four-family paired memory comparison on the existing synthetic episodes."""

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
    "missing_landmarks": dict(cue_flip_probability=0.08, landmark_dropout=0.80,
                              landmark_noise_probability=0.10, turn_noise_probability=0.10),
    "turn_corruption": dict(cue_flip_probability=0.08, landmark_dropout=0.35,
                            landmark_noise_probability=0.10, turn_noise_probability=0.20),
    "combined": dict(cue_flip_probability=0.22, landmark_dropout=0.80,
                     landmark_noise_probability=0.20, turn_noise_probability=0.20),
}


ARMS = ("direct_cx", "dense_expansion_cx", "mb_sparse_t3_cx",
        "mb_sparse_t4_cx", "mb_sparse_t5_cx", "direct_budget_cx")


def architecture(cfg: Config, arm: str) -> dict:
    if arm == "direct_cx":
        return {"family": "direct", "threshold": None, "input_synapses": 0,
                "trainable_synapses": cfg.cue_bits * cfg.sectors}
    if arm == "direct_budget_cx":
        return {"family": "direct_parameter_budget", "threshold": None,
                "input_synapses": cfg.kenyon_cells,
                "trainable_synapses": cfg.kenyon_cells * cfg.sectors}
    threshold = 1 if arm == "dense_expansion_cx" else int(arm.split("_t")[1].split("_")[0])
    return {"family": "dense_expansion" if threshold == 1 else "sparse_expansion",
            "threshold": threshold,
            "input_synapses": cfg.kenyon_cells * cfg.kc_connections,
            "trainable_synapses": cfg.kenyon_cells * cfg.sectors}


def run_comparison(seeds: list[int]) -> tuple[list[dict], list[dict], list[dict]]:
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be nonempty and unique")
    raw: list[dict] = []
    summaries: list[dict] = []
    aggregates: list[dict] = []
    for condition, overrides in CONDITIONS.items():
        cfg = replace(Config(), **overrides)
        for seed in seeds:
            rows = run_seed(cfg, seed, comparison=True)
            for row in rows:
                row["condition"] = condition
            raw.extend(rows)
            for item in summarize(rows):
                item["condition"] = condition
                summaries.append(item)
        for arm in ARMS:
            group = [r for r in summaries if r["condition"] == condition and r["arm"] == arm]
            record = {"condition": condition, "arm": arm, "n_seeds": len(group),
                      **architecture(cfg, arm)}
            for metric in ("goal_accuracy", "steering_accuracy", "heading_mae_sectors",
                           "goal_tie_rate", "heading_tie_rate"):
                values = np.asarray([r[metric] for r in group], dtype=float)
                record[f"{metric}_mean"] = float(values.mean())
                record[f"{metric}_sd_across_seeds"] = (
                    float(values.std(ddof=1)) if len(values) > 1 else 0.0)
            active = np.asarray([r["active_kc"] for r in raw
                                 if r["condition"] == condition and r["arm"] == arm
                                 and r["step"] > 0], dtype=float)
            record["active_input_fraction_mean"] = float(active.mean() / (
                cfg.cue_bits if arm == "direct_cx" else cfg.kenyon_cells))
            aggregates.append(record)
    return raw, summaries, aggregates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-seed", type=int, default=100)
    parser.add_argument("--n-seeds", type=int, default=20)
    args = parser.parse_args()
    if args.n_seeds < 2:
        parser.error("n-seeds must be >= 2")
    seeds = list(range(args.first_seed, args.first_seed + args.n_seeds))
    args.output.mkdir(parents=True, exist_ok=False)
    raw, summaries, aggregate = run_comparison(seeds)
    write_csv(args.output / "trials.csv", raw)
    write_csv(args.output / "summary_by_seed.csv", summaries)
    write_csv(args.output / "summary_aggregate.csv", aggregate)
    (args.output / "manifest.json").write_text(json.dumps({
        "status": "synthetic_development_comparison_not_paper_result",
        "seeds": seeds, "conditions": CONDITIONS, "config": Config().to_dict(),
        "arms": {arm: architecture(Config(), arm) for arm in ARMS},
        "budget_definition": "Allocated trainable cue/expansion-to-goal synapses; shared CX and steering excluded.",
        "warning": "Direct budget repeats cue channels; equal parameter count does not imply equal information, expressivity, or SOP.",
    }, indent=2), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
