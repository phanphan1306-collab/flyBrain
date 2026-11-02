"""Generate a reproducible synthetic log and exercise the full benchmark.

Usage: python -m phase2.run_demo --seed 20261027
"""

from __future__ import annotations

import argparse
from pathlib import Path

from phase2.benchmark import run_benchmark
from phase2.log_schema import write_log
from phase2.synthetic import make_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--output", type=Path, default=Path("phase2/results"))
    args = parser.parse_args()
    records = make_dataset(seed=args.seed, steps=args.steps)
    log_path = args.output / "synthetic_heading.jsonl"
    write_log(log_path, records)
    metrics = run_benchmark(log_path, args.output)
    print(f"Synthetic log: {log_path} ({len(records)} frames)")
    print("These metrics check code behavior only; they are not ant/robot results.")
    for row in metrics:
        if row["condition"] == "all":
            print(f"{row['method']}: MAE {row['circular_mae_deg']:.2f}°, "
                  f"p95 {row['p95_absolute_error_deg']:.2f}°")


if __name__ == "__main__":
    main()
