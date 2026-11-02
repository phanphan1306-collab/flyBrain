"""Export legacy one-EN angular scans as heading-sector evidence.

This reverses the scan grid in ant_route_following_test.m using the recorded
chosen angle and MATLAB's first-minimum rule. No gyro or heading truth is
inferred from this export.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from audit_dataset import read_struct
from heading_fusion import wrap


def export_one(nav_path: Path, ant: int, writer: csv.writer) -> int:
    nav = read_struct(nav_path, "navigation_result")
    steps = np.asarray(nav.step_record, dtype=float)
    activities = np.asarray(nav.EN_response.activity).reshape(-1)
    if steps.ndim != 2 or steps.shape[0] != 3 or len(activities) > steps.shape[1]:
        raise ValueError(f"ant {ant}: invalid scan/step dimensions")
    count = 0
    for t, item in enumerate(activities):
        scores = np.asarray(item.scan, dtype=float).reshape(-1)
        if not scores.size:
            continue
        if scores.size != 61 or not np.all(np.isfinite(scores)):
            raise ValueError(f"ant {ant}, step {t+1}: expected 61 finite EN scores")
        selected = steps[0, t]
        best = int(np.argmin(scores))
        sector_angles = selected + 2 * (best - np.arange(scores.size))
        if abs(wrap(sector_angles[best] - selected)) > 1e-9:
            raise AssertionError("scan reconstruction failed")
        for sector, (angle, score) in enumerate(zip(sector_angles, scores)):
            writer.writerow((ant, t + 1, sector, wrap(angle), score,
                             int(sector == best)))
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("source_data/result"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/legacy_sector_scores.csv"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("ant", "step", "sector", "heading_deg", "en_spikes",
                         "selected_by_legacy_min"))
        total = sum(export_one(args.data / f"ant{ant}route1.mat", ant, writer)
                    for ant in range(1, 16))
    print(f"Wrote {total} scans to {args.output}")


if __name__ == "__main__":
    main()
