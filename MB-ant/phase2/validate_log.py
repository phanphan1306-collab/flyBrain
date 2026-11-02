"""Validate a synchronized heading JSONL before calibration/evaluation.

Usage: python -m phase2.validate_log path/to/log.jsonl
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from phase2.log_schema import load_log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    records = load_log(args.log)
    counts = Counter(r["split"] for r in records)
    print(f"Valid log: {len(records)} frames, {len(set(r['run_id'] for r in records))} runs")
    print("Splits:", ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))


if __name__ == "__main__":
    main()
