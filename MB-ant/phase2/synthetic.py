"""Controlled synthetic observations for checking the fusion experiment code.

These are not MB network outputs and are never presented as ant/robot results.
"""

from __future__ import annotations

import numpy as np

from heading_fusion import wrap


SCENARIOS = {
    "drift_alias": {"gyro_bias": 0.25, "alias_prob": 0.22, "dropout_prob": 0.03},
    "drift_dropout": {"gyro_bias": 0.35, "alias_prob": 0.04, "dropout_prob": 0.22},
    "strong_drift": {"gyro_bias": 0.45, "alias_prob": 0.13, "dropout_prob": 0.06},
}


def make_run(*, split: str, scenario: str, replicate: int, seed: int,
             steps: int = 120) -> list[dict]:
    if scenario not in SCENARIOS or split not in {"validation", "test"}:
        raise ValueError("unknown synthetic split or scenario")
    if steps < 20:
        raise ValueError("steps must be >=20")
    rng = np.random.default_rng(seed)
    cfg = SCENARIOS[scenario]
    run_id = f"synthetic_{split}_{scenario}_{replicate}"
    truth = 35.0
    dead_reckoned = truth
    records = []
    for frame in range(steps):
        if frame == 0:
            gyro_delta = 0.0
        else:
            true_delta = 1.2 * np.sin(frame / 11) + 0.3 * np.cos(frame / 5)
            truth = wrap(truth + true_delta)
            gyro_delta = true_delta + cfg["gyro_bias"] + rng.normal(0, 0.35)
            dead_reckoned = wrap(dead_reckoned + gyro_delta)

        draw = rng.random()
        if draw < cfg["dropout_prob"]:
            event = "dropout"
            angles: list[float] = []
            scores: list[float] = []
            context_ok = False
            odom_place = None
            matched_place = None
        else:
            event = "alias" if draw < cfg["dropout_prob"] + cfg["alias_prob"] else "normal"
            if event == "normal" and rng.random() < 0.05:
                event = "flat"
            angles_array = np.array([wrap(dead_reckoned + offset)
                                     for offset in np.arange(-60, 61, 2)], dtype=float)
            if event == "flat":
                scores_array = 5 + rng.normal(0, 0.02, len(angles_array))
            else:
                true_distance = np.abs(np.array([wrap(a - truth) for a in angles_array]))
                scores_array = 2 + 0.20 * true_distance ** 1.5
                if event == "alias":
                    false_heading = wrap(dead_reckoned + rng.choice([-1, 1]) * rng.uniform(35, 55))
                    alias_distance = np.abs(np.array([wrap(a - false_heading) for a in angles_array]))
                    scores_array = np.minimum(scores_array,
                                              0.5 + 0.20 * alias_distance ** 1.5)
                scores_array = scores_array + rng.normal(0, 0.3, len(angles_array))
            angles = angles_array.tolist()
            scores = scores_array.tolist()

            # A separate, noisy place-progress channel. It uses no heading
            # truth when the fusion method runs. Some aliases fool it too.
            odom_place = frame / 5 + rng.normal(0, 0.35)
            matched_place = frame / 5 + rng.normal(0, 0.35)
            if event == "alias" and rng.random() < 0.7:
                matched_place += rng.choice([-1, 1]) * rng.uniform(6, 12)
            context_ok = bool(abs(matched_place - odom_place) <= 2)

        records.append({
            "schema_version": 1,
            "run_id": run_id,
            "route_id": run_id,
            "split": split,
            "origin": "synthetic",
            "frame": frame,
            "timestamp_s": round(frame * 0.1, 9),
            "gyro_delta_deg": float(gyro_delta),
            "heading_truth_deg": float(truth),
            "sector_angles_deg": angles,
            "familiarity_scores": scores,
            "visual_source": "synthetic_familiarity",
            "context_ok": context_ok,
            "context_source": "noisy_independent_place_progress",
            "odom_place_id": None if odom_place is None else float(odom_place),
            "matched_place_id": None if matched_place is None else float(matched_place),
            "event_type": event,
            "condition": scenario,
            "image_path": None,
        })
    return records


def make_dataset(seed: int = 20261027, steps: int = 120,
                 replicates_per_split: int = 2) -> list[dict]:
    records = []
    for split_index, split in enumerate(("validation", "test")):
        for scenario_index, scenario in enumerate(SCENARIOS):
            for replicate in range(replicates_per_split):
                run_seed = seed + split_index * 1000 + scenario_index * 100 + replicate
                records.extend(make_run(split=split, scenario=scenario,
                                        replicate=replicate, seed=run_seed, steps=steps))
    return records
