"""Paired synthetic benchmark; no images, camera, robot, or real fly recordings."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from .model import (BudgetMatchedDirectMemorySNN, Config, ContextMemorySNN, DirectMemorySNN,
                    HeadingRingSNN, SteeringSNN, circular_error, decode_spikes)


def cue_with_noise(clean: np.ndarray, probability: float, rng: np.random.Generator) -> np.ndarray:
    return np.logical_xor(clean, rng.random(clean.size) < probability).astype(np.int8)


def reference_turn(goal: int, heading: int, sectors: int) -> int | None:
    delta = (goal - heading) % sectors
    if delta == 0:
        return 0
    if delta == sectors // 2:
        return None  # Exactly opposite: left and right are equally short.
    return 1 if delta < sectors // 2 else -1


def run_seed(cfg: Config, seed: int, *, comparison: bool = False) -> list[dict]:
    cfg.validate()
    rng = np.random.default_rng(seed)
    train_rng = np.random.default_rng(seed + 1_000_000)
    prototypes = np.zeros((cfg.contexts, cfg.cue_bits), dtype=np.int8)
    for prototype in prototypes:
        prototype[rng.choice(cfg.cue_bits, cfg.cue_bits // 3, replace=False)] = 1
    goals = rng.choice(cfg.sectors, cfg.contexts, replace=False) if cfg.contexts <= cfg.sectors else rng.integers(0, cfg.sectors, cfg.contexts)
    fixed_goal = Counter(goals.tolist()).most_common(1)[0][0]
    shuffled_goals = goals[train_rng.permutation(cfg.contexts)]

    if comparison:
        # All expansion thresholds share exactly the same PN-KC wiring.
        memories = {
            "direct_cx": DirectMemorySNN(cfg),
            "direct_budget_cx": BudgetMatchedDirectMemorySNN(cfg, seed + 20_000),
            **{name: ContextMemorySNN(replace(cfg, kc_threshold=threshold), seed + 10_000)
               for name, threshold in (("dense_expansion_cx", 1),
                                       ("mb_sparse_t3_cx", 3),
                                       ("mb_sparse_t4_cx", 4),
                                       ("mb_sparse_t5_cx", 5))},
        }
    else:
        memories = {
            "cx_fixed": None,
            "direct_cx": DirectMemorySNN(cfg),
            "mb_cx": ContextMemorySNN(cfg, seed + 10_000),
            "mb_shuffled": ContextMemorySNN(cfg, seed + 10_000),
        }
    for _ in range(cfg.train_repeats):
        for context in train_rng.permutation(cfg.contexts):
            for name, memory in memories.items():
                if memory is not None:
                    target = shuffled_goals[context] if name == "mb_shuffled" else goals[context]
                    memory.train(prototypes[context], int(target))

    ring = HeadingRingSNN(cfg)
    steering = SteeringSNN(cfg)
    rows: list[dict] = []
    for context in range(cfg.contexts):
        turn_rng = np.random.default_rng(seed + 2_000_000 + context)
        landmark_rng = np.random.default_rng(seed + 3_000_000 + context)
        cue_rng = np.random.default_rng(seed + 4_000_000 + context)
        turn_noise_rng = np.random.default_rng(seed + 5_000_000 + context)
        ring.reset()
        true_heading = int(turn_rng.integers(cfg.sectors))
        goal = int(goals[context])
        # Same episode and sensory realization for all comparison arms.
        for step in range(cfg.episode_steps):
            turn = 0 if step == 0 else int(turn_rng.choice([-1, 0, 1], p=[0.25, 0.5, 0.25]))
            true_heading = (true_heading + turn) % cfg.sectors
            turn_corrupted = turn_noise_rng.random() < cfg.turn_noise_probability
            turn_perturbation = int(turn_noise_rng.choice([-1, 1]))
            observed_turn = (int(np.clip(turn + turn_perturbation, -1, 1))
                             if step > 0 and turn_corrupted else turn)
            dropped = landmark_rng.random() < cfg.landmark_dropout
            noisy = landmark_rng.random() < cfg.landmark_noise_probability
            noise_direction = int(landmark_rng.choice([-1, 1]))
            if step == 0:
                landmark = true_heading  # common initialization, not a test score
            elif dropped:
                landmark = None
            else:
                landmark = true_heading
                if noisy:
                    landmark = (landmark + noise_direction) % cfg.sectors
            heading_counts = ring.update(observed_turn, landmark)
            heading_prediction, heading_tie = decode_spikes(heading_counts)
            cue = cue_with_noise(prototypes[context], cfg.cue_flip_probability, cue_rng)
            reference = reference_turn(goal, true_heading, cfg.sectors)
            for arm, memory in memories.items():
                if arm == "cx_fixed":
                    goal_counts = np.zeros(cfg.sectors, dtype=np.int32)
                    goal_counts[fixed_goal] = cfg.memory_ticks
                    active_kc = 0
                else:
                    goal_counts, active_kc = memory.recall(cue)
                goal_prediction, goal_tie = decode_spikes(goal_counts)
                decision, motor_spikes = steering.decide(goal_counts, heading_counts)
                rows.append({
                    "seed": seed, "context": context, "step": step, "arm": arm,
                    "true_heading": true_heading, "goal": goal, "true_turn": turn,
                    "observed_turn": observed_turn,
                    "landmark_cue": "" if landmark is None else landmark,
                    "heading_prediction": heading_prediction,
                    "heading_tie": int(heading_tie),
                    "heading_error_sectors": circular_error(heading_prediction, true_heading, cfg.sectors),
                    "goal_prediction": goal_prediction, "goal_tie": int(goal_tie),
                    "goal_error_sectors": circular_error(goal_prediction, goal, cfg.sectors),
                    "active_kc": active_kc, "decision": decision,
                    "reference_decision": "" if reference is None else reference,
                    "motor_left_spikes": motor_spikes[0],
                    "motor_right_spikes": motor_spikes[1],
                    "steering_evaluable": int(reference is not None),
                    "steering_correct": "" if reference is None else int(decision == reference),
                })
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["seed"], row["arm"])].append(row)
    output = []
    for (seed, arm), group in sorted(grouped.items()):
        # Step zero is a guaranteed landmark used only to initialize the ring.
        # Keep it in trials.csv, but never include it in benchmark metrics.
        evaluated = [r for r in group if r["step"] > 0]
        valid = [r for r in evaluated if r["steering_evaluable"]]
        output.append({
            "seed": seed, "arm": arm, "n": len(evaluated),
            "heading_mae_sectors": float(np.mean([r["heading_error_sectors"] for r in evaluated])),
            "heading_tie_rate": float(np.mean([r["heading_tie"] for r in evaluated])),
            "goal_accuracy": float(np.mean([r["goal_prediction"] == r["goal"] for r in evaluated])),
            "goal_tie_rate": float(np.mean([r["goal_tie"] for r in evaluated])),
            "steering_accuracy": float(np.mean([r["steering_correct"] for r in valid])) if valid else None,
            "n_steering": len(valid),
        })
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/fly_snn_pilot"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--contexts", type=int, default=8)
    parser.add_argument("--cue-noise", type=float, default=0.08)
    parser.add_argument("--landmark-dropout", type=float, default=0.35)
    parser.add_argument("--landmark-noise", type=float, default=0.10)
    parser.add_argument("--turn-noise", type=float, default=0.10)
    args = parser.parse_args()
    cfg = replace(Config(), contexts=args.contexts,
                  cue_flip_probability=args.cue_noise,
                  landmark_dropout=args.landmark_dropout,
                  landmark_noise_probability=args.landmark_noise,
                  turn_noise_probability=args.turn_noise)
    cfg.validate()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seed values must be unique")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = [row for seed in args.seeds for row in run_seed(cfg, seed)]
    summary = summarize(rows)
    write_csv(args.output / "trials.csv", rows)
    write_csv(args.output / "summary_by_seed.csv", summary)
    (args.output / "manifest.json").write_text(json.dumps({
        "status": "synthetic_pilot_not_paper_result",
        "model_provenance": "new minimal SNN modules, not paper reproduction",
        "config": cfg.to_dict(), "seeds": args.seeds,
        "python": platform.python_version(), "numpy": np.__version__,
        "n_trial_rows": len(rows),
        "split": "clean prototype training; independent noisy test episodes",
    }, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
