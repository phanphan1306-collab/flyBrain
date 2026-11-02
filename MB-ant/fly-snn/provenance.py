"""Record checksums for code and result files used by a benchmark run."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    module_dir = Path(__file__).resolve().parent
    code = {path.name: sha256(path) for path in sorted(module_dir.glob("*.py"))}
    results = {name: sha256(args.directory / name) for name in
               ("trials.csv", "summary_by_seed.csv", "summary_aggregate.csv",
                "paired_deltas.csv", "baseline_summary.png", "manifest.json")}
    output = args.directory / "provenance.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps({
        "sha256_code": code, "sha256_results": results,
        "python": platform.python_version(), "numpy": np.__version__,
        "platform": platform.platform(),
        "literature": {
            "Stentiford_2024": "https://doi.org/10.1371/journal.pcbi.1011913",
            "Ardin_2016": "https://doi.org/10.1371/journal.pcbi.1004683",
            "Goulard_2023": "https://doi.org/10.1371/journal.pcbi.1011480",
        },
    }, indent=2), encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
