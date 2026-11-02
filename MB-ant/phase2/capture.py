"""Append synchronized observations from a robot or simulator loop.

The caller supplies sensor values; this module does not connect to hardware.
Create one JSONL file per run, then validate it with phase2.validate_log.
"""

from __future__ import annotations

import json
from pathlib import Path

from phase2.log_schema import validate_record


class HeadingLogWriter:
    def __init__(self, path: Path, *, run_id: str, route_id: str, split: str,
                 origin: str, condition: str, visual_source: str,
                 context_source: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("x", encoding="utf-8", newline="\n")
        self.metadata = {
            "schema_version": 1, "run_id": run_id, "route_id": route_id,
            "split": split, "origin": origin, "condition": condition,
            "visual_source": visual_source, "context_source": context_source,
        }
        self.frame = 0
        self.last_time: float | None = None

    def append(self, *, timestamp_s: float, gyro_delta_deg: float,
               heading_truth_deg: float | None,
               sector_angles_deg: list[float], familiarity_scores: list[float],
               context_ok: bool, event_type: str = "observed",
               image_path: str | None = None,
               odom_place_id: float | None = None,
               matched_place_id: float | None = None) -> None:
        if self.stream.closed:
            raise ValueError("writer is closed")
        if self.last_time is not None and timestamp_s <= self.last_time:
            raise ValueError("timestamp_s must increase within a run")
        if self.frame == 0 and abs(gyro_delta_deg) > 1e-12:
            raise ValueError("first gyro_delta_deg must be zero")
        record = {
            **self.metadata, "frame": self.frame,
            "timestamp_s": timestamp_s, "gyro_delta_deg": gyro_delta_deg,
            "heading_truth_deg": heading_truth_deg,
            "sector_angles_deg": sector_angles_deg,
            "familiarity_scores": familiarity_scores,
            "context_ok": context_ok, "event_type": event_type,
            "image_path": image_path, "odom_place_id": odom_place_id,
            "matched_place_id": matched_place_id,
        }
        validate_record(record)
        self.stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.stream.flush()
        self.frame += 1
        self.last_time = timestamp_s

    def close(self) -> None:
        self.stream.close()

    def __enter__(self) -> "HeadingLogWriter":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
