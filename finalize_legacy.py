"""Freeze and reproduce the original MB_ant dataset baseline.

Run from any working directory: python MB_ant/finalize_legacy.py
Use --refresh-manifest only after intentionally replacing source_data files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from audit_dataset import audit_one
from export_sector_scores import export_one


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source_data"
OUTPUT = ROOT / "outputs"
MANIFEST = OUTPUT / "source_manifest.csv"

ROOT_FILES = {
    "Readme.txt", "Read_me_other_models.txt", "analyze_mb_learning.m",
    "analyze_mb_learning.py", "analyze_navigation_results.m",
    "ant_route_following_test.m", "box_plot.mat", "connection_generator.m",
    "EN_neuron.m", "GPU_network_fun.m", "ImgGrabber.m",
    "infomax_navigator.m", "KC_EN_synapse.m", "KC_neuron.m",
    "perfect_memory_flow_field.m", "perfect_memory_navigator.m",
    "PN_KC_synapse.m", "PN_neuron.m", "random_navigator.m", "STDP.m",
    "synapse.m",
}
ANTVIEW_FILES = {"AntData.mat", "world5000_gray.mat"}


def expected_files() -> set[str]:
    expected = set(ROOT_FILES)
    expected.update(f"antview/{name}" for name in ANTVIEW_FILES)
    for ant in range(1, 16):
        expected.update({
            f"result/Raw_images_ant{ant}_route1.mat",
            f"result/Record{ant}_route1.mat",
            f"result/ant{ant}route1.mat",
            f"result/ant{ant}route1.fig",
            f"result/ant{ant}route1.jpg",
        })
    return expected


def source_inventory() -> list[dict[str, str | int]]:
    actual = {p.relative_to(SOURCE).as_posix(): p for p in SOURCE.rglob("*")
              if p.is_file()}
    expected = expected_files()
    missing = sorted(expected - actual.keys())
    extra = sorted(actual.keys() - expected)
    if missing or extra:
        raise ValueError(f"source_data inventory mismatch; missing={missing}, extra={extra}")
    rows = []
    for relative in sorted(actual):
        digest = hashlib.sha256()
        with actual[relative].open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        rows.append({"relative_path": relative, "bytes": actual[relative].stat().st_size,
                     "sha256": digest.hexdigest()})
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def report_text(rows: list[dict], file_count: int) -> tuple[str, dict]:
    n_images = sum(int(r["route_images"]) for r in rows)
    n_scans = sum(int(r["recorded_scans"]) for r in rows)
    n_ties = sum(int(r["scan_tie_count"]) for r in rows)
    n_resets = sum(int(r["assisted_resets"]) for r in rows)
    n_zero = sum(int(r["en_post_zero_count"]) for r in rows)
    weighted_pre = sum(float(r["en_pre_mean"]) * int(r["route_images"]) for r in rows) / n_images
    weighted_post = sum(float(r["en_post_mean"]) * int(r["route_images"]) for r in rows) / n_images
    weighted_kc = sum(float(r["kc_active_fraction_mean"]) * int(r["route_images"])
                      for r in rows) / n_images
    summary = {
        "source_files": file_count,
        "ants": len(rows),
        "route_images": n_images,
        "angular_scans": n_scans,
        "sector_scores": n_scans * 61,
        "scan_ties": n_ties,
        "scan_tie_fraction": n_ties / n_scans,
        "assisted_resets": n_resets,
        "ants_without_reset": sum(int(r["assisted_resets"]) == 0 for r in rows),
        "en_pre_mean_spikes_per_image": weighted_pre,
        "en_post_mean_spikes_per_image": weighted_post,
        "en_post_zero_images": n_zero,
        "en_post_zero_fraction": n_zero / n_images,
        "kc_active_fraction_mean": weighted_kc,
        "all_last_positions_within_nest_radius": all(
            int(r["last_position_within_nest_radius"]) == 1 for r in rows),
        "kc_fired_total_invalid_shape_count": sum(
            r["kc_fired_total_shape"] != f"20000x{3*int(r['route_images'])}" for r in rows),
    }
    lines = [
        "# Baseline dữ liệu cũ MB_ant",
        "",
        "Báo cáo được tạo bằng `python finalize_legacy.py` từ bản sao nguyên trạng",
        "trong `source_data/`. `outputs/source_manifest.csv` lưu SHA-256 của từng tệp.",
        "",
        "## Kết quả đã kiểm tra",
        "",
        f"- {file_count} tệp nguồn; {len(rows)} tuyến kiến; {n_images} ảnh tuyến.",
        f"- EN trung bình theo ảnh: {weighted_pre:.3f} spike trước học, "
        f"{weighted_post:.3f} spike sau học. {n_zero}/{n_images} ảnh hậu học "
        f"({n_zero/n_images:.1%}) có EN bằng 0.",
        f"- KC hoạt động trung bình {weighted_kc:.2%} trên 20.000 KC.",
        f"- {n_ties}/{n_scans} lần quét ({n_ties/n_scans:.1%}) có ít nhất "
        "hai góc cùng đạt EN thấp nhất.",
        f"- {n_resets} lần đặt kiến trở lại tuyến; "
        f"{summary['ants_without_reset']}/{len(rows)} tuyến không cần reset.",
        "- Vị trí cuối của cả 15 tuyến nằm trong bán kính đích theo script gốc; "
        "điều này không xác lập tỷ lệ tự dẫn đường vì có reset.",
        f"- `KC_fired_total` có kích thước sai ở "
        f"{summary['kc_fired_total_invalid_shape_count']}/{len(rows)} bản ghi. "
        "Không dùng trường này để phân tích KC theo pha.",
        "",
        "| Kiến | Ảnh tuyến | EN trước | EN sau | EN=0 sau học | Quét đồng cực tiểu | Reset |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {int(row['ant'])} | {int(row['route_images'])} | "
            f"{float(row['en_pre_mean']):.2f} | {float(row['en_post_mean']):.2f} | "
            f"{float(row['en_post_zero_fraction']):.1%} | "
            f"{int(row['scan_tie_count'])}/{int(row['recorded_scans'])} | "
            f"{int(row['assisted_resets'])} |"
        )
    lines += [
        "",
        "## Cách hiểu và giới hạn",
        "",
        "- EN giảm trên ảnh đã học là chỉ dấu familiarity của mô hình, chưa đo "
        "độ chính xác heading hoặc hiệu quả sửa gyro drift.",
        "- Mỗi lần quét có 61 góc cách nhau 2°. `legacy_sector_scores.csv` "
        "dựng lại góc từ heading đã chọn và vị trí cực tiểu; nó là điểm của "
        "một EN quét nhiều góc, chưa phải các MBON sector độc lập.",
        "- Script gốc làm tròn heading rồi dựng lại tọa độ tuyến theo bước "
        "0,1 m. Dữ liệu này không có chuỗi gyro, timestamp, pose/heading truth "
        "theo thời gian hoặc nhãn nhầm cảnh để hiệu chuẩn reliability gate.",
        "- `error_rate` gốc bằng số reset chia số transition đã lưu. Khi lệch "
        "tuyến, kiến được đặt lại bằng vị trí tham chiếu; không báo cáo "
        "`error_rate` như xác suất thất bại tự điều hướng.",
        "- Kết quả được tính từ dữ liệu lưu sẵn. Chưa chạy lại script MATLAB/GPU; "
        "random seed và lý do dừng không được lưu trong bản ghi cũ.",
        "",
        "## Đầu ra cố định",
        "",
        "- `outputs/source_manifest.csv`: tên, kích thước, SHA-256 nguồn.",
        "- `outputs/dataset_audit.csv`: các phép đo và kiểm tra từng tuyến.",
        "- `outputs/legacy_sector_scores.csv`: 61 score/góc cho mỗi lần quét.",
        "- `outputs/legacy_summary.json`: tổng hợp máy đọc được.",
        "",
    ]
    return "\n".join(lines), summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-manifest", action="store_true",
                        help="Accept an intentional change to source_data")
    args = parser.parse_args()
    inventory = source_inventory()
    OUTPUT.mkdir(exist_ok=True)
    if MANIFEST.exists() and not args.refresh_manifest:
        previous = read_csv(MANIFEST)
        current = [{key: str(value) for key, value in row.items()} for row in inventory]
        if previous != current:
            raise RuntimeError("source_data differs from frozen manifest; inspect changes first")
    else:
        write_csv(MANIFEST, inventory)

    rows = [audit_one(SOURCE / "result", ant) for ant in range(1, 16)]
    write_csv(OUTPUT / "dataset_audit.csv", rows)
    with (OUTPUT / "legacy_sector_scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("ant", "step", "sector", "heading_deg", "en_spikes",
                         "selected_by_legacy_min"))
        scan_count = sum(export_one(SOURCE / "result" / f"ant{ant}route1.mat", ant, writer)
                         for ant in range(1, 16))
    if scan_count != sum(row["recorded_scans"] for row in rows):
        raise ValueError("exported scan count differs from audit")
    report, summary = report_text(rows, len(inventory))
    (ROOT / "LEGACY_BASELINE.md").write_text(report, encoding="utf-8")
    (OUTPUT / "legacy_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Frozen {len(inventory)} source files and audited 15 routes")
    print(f"Report: {ROOT / 'LEGACY_BASELINE.md'}")


if __name__ == "__main__":
    main()
