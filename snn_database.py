"""Build a provenance-aware SQLite database from Ardin SNN result exports.

The archived MATLAB exports and a new SNN scan are separate datasets. This
script stores PN/KC/EN results only; virtual gyro and robot fields are ignored.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY,
    label TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL CHECK (kind IN ('archived_learning', 'archived_scan',
                                      'archived_figures', 'new_snn_scan')),
    source_path TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    imported_at_utc TEXT NOT NULL,
    notes TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS learning_images (
    dataset_id INTEGER NOT NULL REFERENCES datasets(id),
    ant INTEGER NOT NULL,
    image_index INTEGER NOT NULL,
    en_pre_spikes REAL NOT NULL,
    en_training_spikes REAL NOT NULL,
    en_post_spikes REAL NOT NULL,
    kc_pre_fraction REAL NOT NULL,
    kc_training_fraction REAL NOT NULL,
    kc_post_fraction REAL NOT NULL,
    PRIMARY KEY (dataset_id, ant, image_index)
);
CREATE TABLE IF NOT EXISTS learning_summary (
    dataset_id INTEGER NOT NULL REFERENCES datasets(id),
    measure TEXT NOT NULL,
    phase TEXT NOT NULL,
    n INTEGER NOT NULL,
    mean REAL NOT NULL,
    median REAL NOT NULL,
    std_sample REAL NOT NULL,
    PRIMARY KEY (dataset_id, measure, phase)
);
CREATE TABLE IF NOT EXISTS archived_sector_scores (
    dataset_id INTEGER NOT NULL REFERENCES datasets(id),
    ant INTEGER NOT NULL,
    step INTEGER NOT NULL,
    sector INTEGER NOT NULL,
    heading_deg REAL NOT NULL,
    en_spikes REAL NOT NULL,
    selected_by_legacy_min INTEGER NOT NULL CHECK (selected_by_legacy_min IN (0, 1)),
    PRIMARY KEY (dataset_id, ant, step, sector)
);
CREATE INDEX IF NOT EXISTS idx_archived_scan ON archived_sector_scores(ant, step);
CREATE TABLE IF NOT EXISTS snn_frames (
    dataset_id INTEGER NOT NULL REFERENCES datasets(id),
    run_id TEXT NOT NULL,
    frame INTEGER NOT NULL,
    route_id TEXT NOT NULL,
    split TEXT NOT NULL,
    origin TEXT NOT NULL,
    heading_truth_deg REAL,
    best_heading_deg REAL NOT NULL,
    minimum_ties INTEGER NOT NULL,
    circular_error_deg REAL,
    PRIMARY KEY (dataset_id, run_id, frame)
);
CREATE TABLE IF NOT EXISTS snn_sector_scores (
    dataset_id INTEGER NOT NULL,
    run_id TEXT NOT NULL,
    frame INTEGER NOT NULL,
    sector INTEGER NOT NULL,
    heading_deg REAL NOT NULL,
    en_spikes REAL NOT NULL,
    PRIMARY KEY (dataset_id, run_id, frame, sector),
    FOREIGN KEY (dataset_id, run_id, frame)
        REFERENCES snn_frames(dataset_id, run_id, frame)
);
CREATE TABLE IF NOT EXISTS figures (
    dataset_id INTEGER NOT NULL REFERENCES datasets(id),
    filename TEXT NOT NULL,
    source_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    PRIMARY KEY (dataset_id, filename)
);
CREATE TABLE IF NOT EXISTS archived_route_figures (
    dataset_id INTEGER NOT NULL REFERENCES datasets(id),
    ant INTEGER NOT NULL,
    jpg_path TEXT NOT NULL,
    jpg_sha256 TEXT NOT NULL,
    fig_path TEXT NOT NULL,
    fig_sha256 TEXT NOT NULL,
    PRIMARY KEY (dataset_id, ant)
);
"""


def initialize(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    definition = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='datasets'"
    ).fetchone()[0]
    if "archived_figures" not in definition:
        # Databases created by the first version need one additional source kind.
        # Keep all rows and child tables rather than rebuilding the user's data.
        conn.execute("PRAGMA foreign_keys = OFF")
        try:
            with conn:
                conn.execute("""CREATE TABLE datasets_new (
                    id INTEGER PRIMARY KEY,
                    label TEXT NOT NULL UNIQUE,
                    kind TEXT NOT NULL CHECK (kind IN (
                        'archived_learning', 'archived_scan',
                        'archived_figures', 'new_snn_scan')),
                    source_path TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL,
                    imported_at_utc TEXT NOT NULL,
                    notes TEXT NOT NULL)""")
                conn.execute("INSERT INTO datasets_new SELECT * FROM datasets")
                conn.execute("DROP TABLE datasets")
                conn.execute("ALTER TABLE datasets_new RENAME TO datasets")
        finally:
            conn.execute("PRAGMA foreign_keys = ON")
    if conn.execute("PRAGMA foreign_key_check").fetchall():
        raise ValueError("Database has broken foreign keys")


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: str | int | float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def read_csv(path: Path, expected: set[str]) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or []) != expected:
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def register(conn: sqlite3.Connection, label: str, kind: str, path: Path,
             digest: str, notes: str) -> tuple[int, bool]:
    existing = conn.execute(
        "SELECT id, source_sha256 FROM datasets WHERE label = ?", (label,)
    ).fetchone()
    if existing:
        if existing[1] != digest:
            raise ValueError(f"Dataset {label!r} already exists with different source data")
        return int(existing[0]), False
    cursor = conn.execute(
        "INSERT INTO datasets(label, kind, source_path, source_sha256, imported_at_utc, notes) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (label, kind, str(path.resolve()), digest,
         datetime.now(timezone.utc).isoformat(), notes),
    )
    return int(cursor.lastrowid), True


def import_learning(conn: sqlite3.Connection, folder: Path, ant: int) -> None:
    per_image = folder / "learning_per_image.csv"
    summary = folder / "learning_summary.csv"
    validation = folder / "learning_validation.txt"
    files = (per_image, summary, validation)
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    digest = hashlib.sha256("".join(checksum(p) for p in files).encode()).hexdigest()
    label = f"archived_ardin_ant{ant}_learning"
    dataset_id, fresh = register(
        conn, label, "archived_learning", folder, digest,
        "Derived from archived Record and Raw_images files; source seed and "
        "pre-training weight snapshot are unavailable. Not a newly verified MATLAB run.",
    )
    if not fresh:
        print(f"Skipped unchanged {label}")
        return
    image_rows = read_csv(per_image, {
        "image_index", "en_pre_spikes", "en_training_spikes", "en_post_spikes",
        "en_delta_post_minus_pre", "kc_pre_fraction", "kc_training_fraction",
        "kc_post_fraction", "en_zero_both",
    })
    for row in image_rows:
        pre = finite(row["en_pre_spikes"], "en_pre_spikes")
        post = finite(row["en_post_spikes"], "en_post_spikes")
        if not math.isclose(post - pre, finite(row["en_delta_post_minus_pre"], "EN delta")):
            raise ValueError("Inconsistent EN delta")
        conn.execute(
            "INSERT INTO learning_images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (dataset_id, ant, int(row["image_index"]), pre,
             finite(row["en_training_spikes"], "en_training_spikes"), post,
             finite(row["kc_pre_fraction"], "kc_pre_fraction"),
             finite(row["kc_training_fraction"], "kc_training_fraction"),
             finite(row["kc_post_fraction"], "kc_post_fraction")),
        )
    summary_rows = read_csv(summary, {"measure", "phase", "n", "mean", "median", "std_sample"})
    for row in summary_rows:
        conn.execute(
            "INSERT INTO learning_summary VALUES (?, ?, ?, ?, ?, ?, ?)",
            (dataset_id, row["measure"], row["phase"], int(row["n"]),
             finite(row["mean"], "mean"), finite(row["median"], "median"),
             finite(row["std_sample"], "std_sample")),
        )
    for figure in sorted(folder.glob("*.png")):
        conn.execute("INSERT INTO figures VALUES (?, ?, ?, ?)",
                     (dataset_id, figure.name, str(figure.resolve()), checksum(figure)))
    print(f"Imported {label}: {len(image_rows)} images, {len(summary_rows)} summary measures")


def import_archived_scores(conn: sqlite3.Connection, path: Path) -> None:
    label = "archived_ardin_15_route_sector_scores"
    dataset_id, fresh = register(
        conn, label, "archived_scan", path, checksum(path),
        "EN scores reconstructed from archived MATLAB route records. One EN is "
        "scanned over sectors; heading angles are reconstructed, not independent truth.",
    )
    if not fresh:
        print(f"Skipped unchanged {label}")
        return
    rows = read_csv(path, {"ant", "step", "sector", "heading_deg", "en_spikes",
                           "selected_by_legacy_min"})
    for row in rows:
        conn.execute(
            "INSERT INTO archived_sector_scores VALUES (?, ?, ?, ?, ?, ?, ?)",
            (dataset_id, int(row["ant"]), int(row["step"]), int(row["sector"]),
             finite(row["heading_deg"], "heading_deg"),
             finite(row["en_spikes"], "en_spikes"),
             int(row["selected_by_legacy_min"])),
        )
    print(f"Imported {label}: {len(rows)} sector scores")


def import_route_figures(conn: sqlite3.Connection, folder: Path) -> None:
    pairs = []
    for jpg in sorted(folder.glob("ant*route1.jpg")):
        text = jpg.stem.removeprefix("ant").removesuffix("route1")
        if not text.isdigit():
            continue
        fig = jpg.with_suffix(".fig")
        if not fig.is_file():
            raise FileNotFoundError(fig)
        pairs.append((int(text), jpg, fig, checksum(jpg), checksum(fig)))
    if not pairs:
        raise ValueError(f"No ant route JPG/FIG pairs in {folder}")
    ant_ids = [pair[0] for pair in pairs]
    if len(ant_ids) != len(set(ant_ids)):
        raise ValueError("Duplicate ant route figures")
    digest = hashlib.sha256("".join(
        f"{ant}:{jpg_hash}:{fig_hash}" for ant, _, _, jpg_hash, fig_hash in pairs
    ).encode()).hexdigest()
    label = "archived_ardin_route_figures"
    dataset_id, fresh = register(
        conn, label, "archived_figures", folder, digest,
        "JPG route plots and editable MATLAB FIG files in source_data/result. "
        "The user reports running Ardin MATLAB on another GPU-equipped "
        "machine and saving result images here. These files do not contain "
        "a run ID or execution metadata linking them to one specific run.",
    )
    if not fresh:
        print(f"Skipped unchanged {label}")
        return
    conn.executemany(
        "INSERT INTO archived_route_figures VALUES (?, ?, ?, ?, ?, ?)",
        ((dataset_id, ant, str(jpg.resolve()), jpg_hash,
          str(fig.resolve()), fig_hash)
         for ant, jpg, fig, jpg_hash, fig_hash in pairs),
    )
    print(f"Imported {label}: {len(pairs)} JPG/FIG pairs")


def import_snn_jsonl(conn: sqlite3.Connection, path: Path, label: str) -> None:
    dataset_id, fresh = register(
        conn, label, "new_snn_scan", path, checksum(path),
        "SNN EN scan from MATLAB JSONL. Only visual SNN outputs and simulator "
        "heading truth are stored; virtual gyro and robot fields are excluded.",
    )
    if not fresh:
        print(f"Skipped unchanged {label}")
        return
    count = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("visual_source") != "snn_en_scan" or record.get("origin") != "simulator":
                raise ValueError(f"Line {line_number}: expected simulator SNN EN scan")
            angles = record["sector_angles_deg"]
            scores = record["familiarity_scores"]
            if len(angles) != len(scores) or len(scores) < 3:
                raise ValueError(f"Line {line_number}: invalid scan arrays")
            numeric_scores = [finite(v, "EN score") for v in scores]
            numeric_angles = [finite(v, "heading_deg") for v in angles]
            minimum = min(numeric_scores)
            best = numeric_scores.index(minimum)  # MATLAB's first-minimum convention
            ties = sum(value == minimum for value in numeric_scores)
            truth = record.get("heading_truth_deg")
            truth_value = None if truth is None else finite(truth, "heading_truth_deg")
            best_angle = numeric_angles[best]
            error = None if truth_value is None else abs((best_angle - truth_value + 180) % 360 - 180)
            run_id, frame = str(record["run_id"]), int(record["frame"])
            conn.execute(
                "INSERT INTO snn_frames VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (dataset_id, run_id, frame, str(record["route_id"]),
                 str(record["split"]), "simulator", truth_value, best_angle,
                 ties, error),
            )
            conn.executemany(
                "INSERT INTO snn_sector_scores VALUES (?, ?, ?, ?, ?, ?)",
                ((dataset_id, run_id, frame, i, numeric_angles[i], numeric_scores[i])
                 for i in range(len(numeric_scores))),
            )
            count += 1
    if not count:
        raise ValueError(f"No observations in {path}")
    print(f"Imported {label}: {count} SNN frames")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--db", type=Path, default=root / "outputs" / "snn_results.sqlite")
    parser.add_argument("--learning-dir", type=Path, default=root / "outputs" / "ant1_learning")
    parser.add_argument("--legacy-scores", type=Path,
                        default=root / "outputs" / "legacy_sector_scores.csv")
    parser.add_argument("--route-figures-dir", type=Path,
                        default=root / "source_data" / "result")
    parser.add_argument("--snn-jsonl", type=Path, help="Optional fresh run_snn_heading_scan output")
    parser.add_argument("--run-label", help="Unique label required with --snn-jsonl")
    args = parser.parse_args()
    if bool(args.snn_jsonl) != bool(args.run_label):
        parser.error("--snn-jsonl and --run-label must be supplied together")
    args.db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(args.db) as conn:
        initialize(conn)
        with conn:
            if args.learning_dir.is_dir():
                import_learning(conn, args.learning_dir, ant=1)
            if args.legacy_scores.is_file():
                import_archived_scores(conn, args.legacy_scores)
            if args.route_figures_dir.is_dir():
                import_route_figures(conn, args.route_figures_dir)
            if args.snn_jsonl:
                import_snn_jsonl(conn, args.snn_jsonl, args.run_label)
        for table in ("datasets", "learning_images", "archived_sector_scores",
                      "snn_frames", "snn_sector_scores", "figures",
                      "archived_route_figures"):
            print(f"{table}: {conn.execute('SELECT count(*) FROM ' + table).fetchone()[0]}")
    print(f"Database: {args.db.resolve()}")


if __name__ == "__main__":
    main()
