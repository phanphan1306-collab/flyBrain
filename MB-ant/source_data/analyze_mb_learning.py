"""Offline analysis of one MB_ant learning record; never runs the simulation.

Place this file beside the ``result`` directory and run ``py analyze_mb_learning.py``.
Requires numpy, scipy and matplotlib. Outputs go to analysis_outputs/ant1.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


NUM_KC = 20_000  # ant_route_following_test.m:82


def mat_struct(path: Path, variable: str):
    if not path.is_file():
        raise ValueError(f"Missing input: {path}")
    try:
        data = loadmat(path, variable_names=[variable], struct_as_record=False, squeeze_me=True)
    except NotImplementedError as exc:
        raise ValueError(f"Cannot read {path}; MATLAB v7.3 requires a different loader") from exc
    except Exception as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    value = data.get(variable)
    if value is None or not hasattr(value, "_fieldnames"):
        raise ValueError(f"{variable} struct missing from {path}")
    return value


def field(struct, name: str):
    if name not in struct._fieldnames:
        raise ValueError(f"Missing field: {name}")
    return getattr(struct, name)


def numeric_vector(struct, name: str) -> np.ndarray:
    try:
        values = np.asarray(field(struct, name), dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not numeric") from exc
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"{name} contains NaN, Inf or negative values")
    return values


def sample_std(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if values.size > 1 else 0.0


def summary_row(measure: str, phase: str, values: np.ndarray) -> list:
    return [measure, phase, values.size, float(np.mean(values)),
            float(np.median(values)), sample_std(values)]


def analyze(record_path: Path, raw_path: Path, output_dir: Path) -> None:
    record = mat_struct(record_path, "Record")
    raw = mat_struct(raw_path, "Raw_images")
    headings = np.asarray(field(raw, "heading")).reshape(-1)
    positions = np.asarray(field(raw, "img_pos"))
    n = headings.size
    if n < 1 or positions.shape != (n + 1, 2):
        raise ValueError("Raw_images heading/img_pos dimensions do not match")

    pn = numeric_vector(record, "PN_activity")
    kc = numeric_vector(record, "KC_activity")
    en = numeric_vector(record, "EN_activity")
    if any(values.size != 3 * n for values in (pn, kc, en)):
        raise ValueError("PN/KC/EN activity lengths must each equal 3*numel(heading)")
    if np.any(kc > NUM_KC):
        raise ValueError("KC_activity exceeds configured numKC=20000")

    pre, training, post = np.split(en, 3)
    delta = post - pre
    zero_both = (pre == 0) & (post == 0)
    kc_sparse = (kc / NUM_KC).reshape(3, n)
    image_index = np.arange(1, n + 1)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(image_index, pre, "b.-", label="Pre-training")
    ax.plot(image_index, post, "r.-", label="Post-training")
    ax.set(xlabel="Image index on trained route", ylabel="EN spikes / trial",
           title="EN response to the same images before and after learning")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "en_before_after.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for phase, color, values in zip(
        ("Pre-training", "Training", "Post-training"),
        ("b", "k", "r"), kc_sparse
    ):
        ax.plot(image_index, values, color=color, label=phase)
    ax.set(xlabel="Image index on trained route",
           ylabel="Fraction of KC neurons active / trial",
           title="KC sparsity: KC activity divided by 20000")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "kc_sparsity.png", dpi=150)
    plt.close(fig)

    with (output_dir / "learning_per_image.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_index", "en_pre_spikes", "en_training_spikes",
                         "en_post_spikes", "en_delta_post_minus_pre",
                         "kc_pre_fraction", "kc_training_fraction", "kc_post_fraction",
                         "en_zero_both"])
        for i in range(n):
            writer.writerow([i + 1, pre[i], training[i], post[i], delta[i],
                             kc_sparse[0, i], kc_sparse[1, i], kc_sparse[2, i],
                             int(zero_both[i])])

    with (output_dir / "learning_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["measure", "phase", "n", "mean", "median", "std_sample"])
        for measure, values in (("PN_active_neurons", pn),
                                ("KC_active_neurons", kc), ("EN_spikes", en)):
            for phase, phase_values in zip(("pre", "training", "post"), np.split(values, 3)):
                writer.writerow(summary_row(measure, phase, phase_values))
        for phase, values in zip(("pre", "training", "post"), kc_sparse):
            writer.writerow(summary_row("KC_active_fraction", phase, values))
        writer.writerow(summary_row("EN_delta_post_minus_pre", "paired", delta))
        writer.writerow(summary_row("EN_zero_both_fraction", "paired", zero_both.astype(float)))

    (output_dir / "learning_validation.txt").write_text(
        f"record={record_path}\nraw_images={raw_path}\nimages={n}\n"
        f"activity_length_each={en.size} (expected {3*n})\n"
        f"EN_zero_both_count={int(np.sum(zero_both))}\n"
        "numKC=20000 from ant_route_following_test.m:82\n"
        "KC_activity counts neurons with >=1 spike, not total KC spikes.\n"
        "KC_fired_total excluded: original code has inconsistent indexing.\n"
        "Seed and pre-training weight snapshot are not stored in Record.\n"
        "Lower EN spike count is a model familiarity proxy, not proof of learning by itself.\n",
        encoding="utf-8",
    )
    print(f"Wrote analysis to {output_dir.resolve()}")
    print(f"Images: {n}; mean EN pre: {np.mean(pre):.4f}; post: {np.mean(post):.4f}; "
          f"paired post-pre: {np.mean(delta):.4f}")


def main() -> None:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=base / "result" / "Record1_route1.mat")
    parser.add_argument("--raw", type=Path, default=base / "result" / "Raw_images_ant1_route1.mat")
    parser.add_argument("--output", type=Path, default=base / "analysis_outputs" / "ant1")
    args = parser.parse_args()
    analyze(args.record, args.raw, args.output)


if __name__ == "__main__":
    main()
