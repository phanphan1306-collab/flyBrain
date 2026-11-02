"""Calibrate on validation routes, then evaluate five heading methods on test.

Usage: python -m phase2.benchmark phase2/data/synthetic_heading.jsonl
       python -m phase2.benchmark my_synchronized_log.jsonl --output my_results
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from heading_fusion import HeadingFusion, wrap
from phase2.log_schema import load_log


METHODS = ("gyro_only", "visual_only", "ungated", "confidence", "full_gate")
VISUAL_GAIN = 0.5
MAX_CORRECTION_DEG = 8.0
INITIAL_OFFSET_DEG = 5.0


def grouped_runs(records: list[dict]) -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        runs[record["run_id"]].append(record)
    return dict(runs)


def replay_run(rows: list[dict], method: str, *, min_confidence: float = 0.08,
               direct_gate_deg: float = 35.0,
               confidence_reference: float = 0.1) -> list[dict]:
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    if any(r["heading_truth_deg"] is None for r in rows):
        raise ValueError("benchmark requires heading_truth_deg on every frame")
    # All methods receive the same independently calibrated initial heading.
    # Truth is used only here and for evaluation, never inside later updates.
    estimate = wrap(rows[0]["heading_truth_deg"] + INITIAL_OFFSET_DEG)
    fusion = None
    if method in {"confidence", "full_gate"}:
        fusion = HeadingFusion(
            estimate, min_confidence=min_confidence,
            direct_gate_deg=180.0 if method == "confidence" else direct_gate_deg,
            visual_gain=VISUAL_GAIN, max_correction_deg=MAX_CORRECTION_DEG,
            confidence_reference=confidence_reference,
        )
    trace = []
    for index, record in enumerate(rows):
        truth = float(record["heading_truth_deg"])
        if index == 0:
            predicted, correction, confidence, reason = estimate, 0.0, 0.0, "initial_calibration"
        else:
            angles = record["sector_angles_deg"]
            scores = record["familiarity_scores"]
            predicted = wrap(estimate + record["gyro_delta_deg"])
            correction, confidence, reason = 0.0, 0.0, "gyro_only"
            if method == "gyro_only":
                estimate = predicted
            elif method == "visual_only":
                predicted = estimate  # no gyro is used by this comparator
                if angles:
                    estimate = wrap(angles[int(np.argmin(scores))])
                    correction = wrap(estimate - predicted)
                    reason = "visual_direct"
                else:
                    reason = "visual_missing_hold"
            elif method == "ungated":
                if angles:
                    visual = wrap(angles[int(np.argmin(scores))])
                    innovation = wrap(visual - predicted)
                    correction = float(np.clip(VISUAL_GAIN * innovation,
                                               -MAX_CORRECTION_DEG, MAX_CORRECTION_DEG))
                    reason = "visual_unconditional"
                estimate = wrap(predicted + correction)
            else:
                assert fusion is not None
                result = fusion.update(
                    record["gyro_delta_deg"], angles if angles else None,
                    scores if scores else None,
                    context_ok=bool(record["context_ok"]) if method == "full_gate" else False,
                )
                predicted, estimate = result.predicted_deg, result.heading_deg
                correction, confidence, reason = (result.correction_deg,
                                                  result.confidence, result.reason)
        error = abs(wrap(estimate - truth))
        predicted_error = abs(wrap(predicted - truth))
        trace.append({
            "origin": record["origin"], "split": record["split"],
            "condition": record["condition"], "run_id": record["run_id"],
            "frame": record["frame"], "event_type": record["event_type"],
            "method": method, "truth_deg": truth, "predicted_deg": predicted,
            "estimate_deg": estimate, "absolute_error_deg": error,
            "correction_deg": correction, "confidence": confidence,
            "reason": reason,
            "harmful_correction_gt5deg": int(
                abs(correction) > 1e-9 and error - predicted_error > 5),
        })
    return trace


def objective(runs: list[list[dict]], method: str, params: dict) -> float:
    errors = np.array([row["absolute_error_deg"]
                       for run in runs for row in replay_run(run, method, **params)])
    return float(np.mean(errors) + 0.25 * np.percentile(errors, 95))


def tune(validation_runs: list[list[dict]]) -> dict[str, dict]:
    chosen = {}
    for method in ("confidence", "full_gate"):
        candidates = []
        for minimum in (0.02, 0.05, 0.08, 0.12):
            gates = (180.0,) if method == "confidence" else (20.0, 35.0, 50.0)
            for gate in gates:
                for reference in (0.05, 0.1, 0.2):
                    params = {"min_confidence": minimum, "direct_gate_deg": gate,
                              "confidence_reference": reference}
                    candidates.append((objective(validation_runs, method, params), params))
        _, chosen[method] = min(candidates, key=lambda pair: pair[0])
    return chosen


def summarize(trace: list[dict], condition: str) -> dict:
    rows = trace if condition == "all" else [r for r in trace if r["condition"] == condition]
    errors = np.array([r["absolute_error_deg"] for r in rows])
    return {
        "origin": rows[0]["origin"], "condition": condition,
        "method": rows[0]["method"], "frames": len(rows),
        "circular_mae_deg": float(np.mean(errors)),
        "p95_absolute_error_deg": float(np.percentile(errors, 95)),
        "error_gt45_fraction": float(np.mean(errors > 45)),
        "visual_corrections": sum(abs(r["correction_deg"]) > 1e-9 for r in rows),
        "harmful_corrections_gt5deg": sum(r["harmful_correction_gt5deg"] for r in rows),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(log_path: Path, output_dir: Path) -> list[dict]:
    records = load_log(log_path)
    runs = grouped_runs(records)
    validation = [rows for rows in runs.values() if rows[0]["split"] == "validation"]
    test = [rows for rows in runs.values() if rows[0]["split"] == "test"]
    if not validation or not test:
        raise ValueError("benchmark needs disjoint validation and test runs")
    origins = {r["origin"] for r in records}
    if len(origins) != 1:
        raise ValueError("do not pool synthetic, simulator, and robot data")
    tuned = tune(validation)
    trace = []
    for method in METHODS:
        params = tuned.get(method, {})
        for run in test:
            trace.extend(replay_run(run, method, **params))
    conditions = sorted({r["condition"] for r in records if r["split"] == "test"})
    summaries = [summarize([r for r in trace if r["method"] == method], condition)
                 for method in METHODS for condition in ["all", *conditions]]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "test_metrics.csv", summaries)
    write_csv(output_dir / "test_trace.csv", trace)
    config = {
        "log_file": str(log_path.resolve()), "origin": origins.pop(),
        "validation_run_ids": sorted(r[0]["run_id"] for r in validation),
        "test_run_ids": sorted(r[0]["run_id"] for r in test),
        "selection_objective": "validation circular MAE + 0.25 * validation p95 error",
        "initial_heading": "truth at first frame + 5 degrees, shared by all methods",
        "visual_gain": VISUAL_GAIN, "max_correction_deg": MAX_CORRECTION_DEG,
        "tuned_on_validation": tuned,
    }
    (output_dir / "benchmark_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, default=Path("phase2/results"))
    args = parser.parse_args()
    rows = run_benchmark(args.log, args.output)
    print(f"Wrote {len(rows)} metric rows to {args.output / 'test_metrics.csv'}")
    for row in rows:
        if row["condition"] == "all":
            print(f"{row['method']}: MAE {row['circular_mae_deg']:.2f}°, "
                  f"p95 {row['p95_absolute_error_deg']:.2f}°")


if __name__ == "__main__":
    main()
