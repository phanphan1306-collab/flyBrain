"""Versioned JSONL contract for heading experiments.

One record is one synchronized time step. `heading_truth_deg` is for evaluation
only: estimators must never read it after initialization. Sector scores use
lower = more familiar. Missing vision is represented by two empty lists.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path


REQUIRED = {
    "schema_version", "run_id", "route_id", "split", "origin", "frame",
    "timestamp_s", "gyro_delta_deg", "heading_truth_deg",
    "sector_angles_deg", "familiarity_scores", "visual_source",
    "context_ok", "context_source", "event_type", "condition",
}
SPLITS = {"validation", "test", "deployment"}
ORIGINS = {"synthetic", "simulator", "robot"}


def _finite_number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def validate_record(record: dict) -> None:
    missing = REQUIRED - record.keys()
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")
    if record["schema_version"] != 1:
        raise ValueError("unsupported schema_version")
    for key in ("run_id", "route_id", "visual_source", "context_source", "event_type", "condition"):
        if not isinstance(record[key], str) or not record[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    if record["split"] not in SPLITS or record["origin"] not in ORIGINS:
        raise ValueError("invalid split or origin")
    if isinstance(record["frame"], bool) or not isinstance(record["frame"], int) or record["frame"] < 0:
        raise ValueError("frame must be a nonnegative integer")
    _finite_number(record["timestamp_s"], "timestamp_s")
    _finite_number(record["gyro_delta_deg"], "gyro_delta_deg")
    truth = record["heading_truth_deg"]
    if truth is None:
        if record["split"] != "deployment":
            raise ValueError("validation/test records require heading_truth_deg")
    else:
        _finite_number(truth, "heading_truth_deg")
    if not isinstance(record["context_ok"], bool):
        raise ValueError("context_ok must be boolean")
    angles, scores = record["sector_angles_deg"], record["familiarity_scores"]
    if not isinstance(angles, list) or not isinstance(scores, list) or len(angles) != len(scores):
        raise ValueError("sector angles and scores must be same-length lists")
    if len(angles) not in (0,) and len(angles) < 3:
        raise ValueError("visual scan needs at least 3 sectors")
    for i, value in enumerate(angles):
        _finite_number(value, f"sector_angles_deg[{i}]")
    for i, value in enumerate(scores):
        _finite_number(value, f"familiarity_scores[{i}]")
    if not angles and record["context_ok"]:
        raise ValueError("context_ok cannot be true without a visual scan")
    image_path = record.get("image_path")
    if image_path is not None and not isinstance(image_path, str):
        raise ValueError("image_path must be a string or null")
    for key in ("odom_place_id", "matched_place_id"):
        if record.get(key) is not None:
            _finite_number(record[key], key)


def load_log(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError("record must be an object")
                validate_record(item)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            records.append(item)
    if not records:
        raise ValueError("empty heading log")
    grouped: dict[str, list[dict]] = defaultdict(list)
    route_split: dict[str, str] = {}
    for item in records:
        grouped[item["run_id"]].append(item)
        old = route_split.setdefault(item["route_id"], item["split"])
        if old != item["split"]:
            raise ValueError(f"route_id {item['route_id']} appears in multiple splits")
    for run_id, rows in grouped.items():
        fixed = {(r["route_id"], r["split"], r["origin"], r["condition"],
                  r["visual_source"]) for r in rows}
        if len(fixed) != 1:
            raise ValueError(f"run_id {run_id} changes route, split, origin, condition, or visual source")
        frames = [r["frame"] for r in rows]
        times = [r["timestamp_s"] for r in rows]
        if frames != list(range(len(rows))) or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError(f"run_id {run_id} has noncontiguous frames or timestamps")
        if abs(float(rows[0]["gyro_delta_deg"])) > 1e-12:
            raise ValueError(f"run_id {run_id}: first gyro delta must be zero")
    return records


def write_log(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for item in records:
            validate_record(item)
            stream.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    load_log(path)
