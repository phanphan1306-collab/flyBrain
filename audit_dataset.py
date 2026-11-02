"""Audit the downloaded MB_ant records without rerunning the GPU simulation.

Usage: python audit_dataset.py [--data source_data/result] [--output outputs]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def read_struct(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    obj = loadmat(path, variable_names=[name], squeeze_me=True,
                  struct_as_record=False).get(name)
    if obj is None or not hasattr(obj, "_fieldnames"):
        raise ValueError(f"{path}: missing {name} struct")
    return obj


def vector(obj, name: str) -> np.ndarray:
    values = np.asarray(getattr(obj, name), dtype=float).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name}: nonfinite values")
    return values


def distance_to_polyline(points: np.ndarray, route: np.ndarray) -> np.ndarray:
    """Euclidean distance from each point to the nearest route segment."""
    if len(points) == 0:
        return np.empty(0)
    a, b = route[:-1], route[1:]
    ab = b - a
    denom = np.sum(ab * ab, axis=1)
    safe = np.where(denom == 0, 1, denom)
    rel = points[:, None, :] - a[None, :, :]
    fraction = np.clip(np.sum(rel * ab[None, :, :], axis=2) / safe, 0, 1)
    projected = a[None, :, :] + fraction[:, :, None] * ab[None, :, :]
    return np.linalg.norm(points[:, None, :] - projected, axis=2).min(axis=1)


def audit_one(data: Path, ant: int) -> dict:
    raw = read_struct(data / f"Raw_images_ant{ant}_route1.mat", "Raw_images")
    record = read_struct(data / f"Record{ant}_route1.mat", "Record")
    nav = read_struct(data / f"ant{ant}route1.mat", "navigation_result")
    n = vector(raw, "heading").size
    route = np.asarray(raw.img_pos, dtype=float)
    images = np.asarray(raw.position).reshape(-1)
    if route.shape != (n + 1, 2) or len(images) != n:
        raise ValueError(f"ant {ant}: Raw_images dimensions mismatch")
    if not np.all(np.isfinite(route)):
        raise ValueError(f"ant {ant}: nonfinite route coordinates")
    en = vector(record, "EN_activity")
    kc = vector(record, "KC_activity")
    pn = vector(record, "PN_activity")
    if en.size != 3 * n or kc.size != 3 * n or pn.size != 3 * n:
        raise ValueError(f"ant {ant}: activity length != 3 * route images")
    if np.any(en < 0) or np.any(kc < 0) or np.any(kc > 20000):
        raise ValueError(f"ant {ant}: invalid neuron activity")
    en_pre, _, en_post = en.reshape(3, n)
    kc_phases = kc.reshape(3, n)

    positions = np.asarray(nav.current_position, dtype=float).reshape(-1, 2)
    steps = np.asarray(nav.step_record, dtype=float)
    if steps.shape != (3, len(positions)):
        raise ValueError(f"ant {ant}: step_record/position mismatch")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(steps)):
        raise ValueError(f"ant {ant}: nonfinite navigation log")
    if not np.all(np.isin(steps[2], [0, 1])):
        raise ValueError(f"ant {ant}: invalid scan flags")
    scans = []
    for item in np.asarray(nav.EN_response.activity).reshape(-1):
        values = np.asarray(item.scan, dtype=float).reshape(-1)
        if values.size:
            scans.append(values)
    scan_margins = []
    tied = 0
    for values in scans:
        sorted_values = np.sort(values)
        scan_margins.append(sorted_values[1] - sorted_values[0])
        tied += sorted_values[1] == sorted_values[0]

    perf = vector(nav, "perf_measure")
    if not np.all(np.isin(perf, [0, 1])):
        raise ValueError(f"ant {ant}: invalid reset flags")
    transition_is_scan = steps[2, :-1] == 1
    commanded = np.diff(positions, axis=0)[transition_is_scan]
    reset = np.diff(positions, axis=0)[~transition_is_scan]
    reset_count = int(np.sum(perf))
    error_locations = np.asarray(nav.error_location, dtype=float)
    if error_locations.size % 2 or error_locations.size // 2 != reset_count:
        raise ValueError(f"ant {ant}: reset count/error locations mismatch")
    recorded_error_rate = float(nav.error_rate)
    recomputed_error_rate = reset_count / (len(positions) - 1)
    if abs(recorded_error_rate - recomputed_error_rate) > 1e-10:
        raise ValueError(f"ant {ant}: recorded error_rate mismatch")
    threshold = float(nav.dis_threshold)
    if threshold <= 0:
        raise ValueError(f"ant {ant}: invalid off-route threshold")
    off_route = distance_to_polyline(positions[1:][transition_is_scan], route)
    if len(scans) != len(commanded) or len(reset) != reset_count:
        raise ValueError(f"ant {ant}: scans/movements/resets mismatch")
    return {
        "ant": ant,
        "route_images": n,
        "kc_active_fraction_mean": np.mean(kc_phases / 20000),
        "en_pre_mean": np.mean(en_pre),
        "en_post_mean": np.mean(en_post),
        "en_post_zero_fraction": np.mean(en_post == 0),
        "en_post_zero_count": int(np.sum(en_post == 0)),
        "en_post_lt_pre_fraction": np.mean(en_post < en_pre),
        "recorded_scans": len(scans),
        "scan_tie_count": int(tied),
        "scan_tie_fraction": tied / len(scans) if scans else np.nan,
        "scan_margin_median_spikes": np.median(scan_margins) if scans else np.nan,
        "assisted_resets": reset_count,
        "recorded_error_rate": recorded_error_rate,
        "commanded_steps": len(commanded),
        "resets_per_commanded_step": reset_count / len(commanded),
        "commanded_distance_m": np.linalg.norm(commanded, axis=1).sum(),
        "reset_jump_distance_m": np.linalg.norm(reset, axis=1).sum(),
        "mean_off_route_commanded_endpoint_m": np.mean(off_route),
        "max_off_route_commanded_endpoint_m": np.max(off_route),
        "last_position_to_nest_m": np.linalg.norm(positions[-1] - route[-1]),
        "last_position_within_nest_radius": int(np.linalg.norm(positions[-1] - route[-1]) <= threshold),
        "kc_fired_total_shape": "x".join(map(str, np.shape(record.KC_fired_total))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("source_data/result"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    rows = [audit_one(args.data, ant) for ant in range(1, 16)]
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "dataset_audit.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}; {len(rows)} ants")
    print(f"Median post EN zero fraction: {np.median([r['en_post_zero_fraction'] for r in rows]):.3f}")
    print(f"Total assisted resets: {sum(r['assisted_resets'] for r in rows)}")


if __name__ == "__main__":
    main()
