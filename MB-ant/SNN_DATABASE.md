# Database kết quả SNN Ardin

`snn_database.py` tạo SQLite ở `outputs/snn_results.sqlite`. Mã chỉ lưu số
liệu PN/KC/EN và thông tin nguồn; không lưu gyro, IMU hoặc dữ liệu robot.

## Dữ liệu hiện có

```powershell
cd MB_ant
python snn_database.py
```

Script nhập `outputs/ant1_learning/learning_per_image.csv`,
`learning_summary.csv`, hai ảnh PNG trong cùng thư mục, và
`outputs/legacy_sector_scores.csv`. Script cũng lập danh mục 15 ảnh JPG tuyến
và 15 tệp FIG MATLAB tương ứng trong `source_data/result/`. Người dùng cho biết
đã chạy MATLAB trên máy khác có GPU và lưu ảnh kết quả vào `MB_ant`; các tệp
ảnh hiện không có run ID hoặc metadata thực thi để nối chúng chắc chắn với
một lần chạy cụ thể. Dữ liệu CSV được gắn nhãn **archived** vì lấy từ các tệp
`source_data/result/` đã có sẵn. Ảnh được tham chiếu bằng đường dẫn và SHA-256;
SQLite không nhúng ảnh.

## Thêm một lần chạy SNN mới

Chép log JSONL do `experiment/run_snn_heading_scan.m` xuất ra vào máy này rồi
chạy:

```powershell
python snn_database.py --snn-jsonl "D:\duong-dan\snn_heading_seed73.jsonl" --run-label "pilot_seed73_20261027"
```

Mỗi `run-label` phải là duy nhất. Chạy lại với cùng dữ liệu sẽ bỏ qua bản đã
nhập. Nếu nội dung nguồn đổi mà nhãn vẫn cũ, script dừng để tránh âm thầm ghi
đè; dùng nhãn mới cho lần chạy mới. Log phải có `visual_source=snn_en_scan` và
`origin=simulator`. Script lưu vector điểm EN theo sector, góc tốt nhất theo
quy tắc cực tiểu đầu tiên của MATLAB, số góc đồng cực tiểu, và lỗi góc tròn
so với heading mô phỏng nếu có. Đây là kết quả **mô phỏng**, không phải phép đo
heading ở kiến thật.

Các tệp `_metadata.mat` chứa mô hình SNN lớn được giữ nguyên ở ngoài database;
hiện script chưa nhập ma trận kết nối và trọng số từ chúng.

## Bảng chính

| Bảng | Nội dung |
| --- | --- |
| `datasets` | Nhãn, loại nguồn, đường dẫn và SHA-256 |
| `learning_images`, `learning_summary` | EN và KC trước/trong/sau học của kiến 1 |
| `archived_sector_scores` | Điểm EN theo góc của dữ liệu MATLAB lưu sẵn |
| `snn_frames`, `snn_sector_scores` | Kết quả của log SNN mới khi được nhập |
| `figures` | Đường dẫn và checksum của ảnh biểu đồ |
| `archived_route_figures` | 15 cặp JPG/FIG theo số kiến |

Ví dụ truy vấn bằng Python:

```python
import sqlite3

with sqlite3.connect("outputs/snn_results.sqlite") as db:
    print(db.execute("SELECT label, kind FROM datasets").fetchall())
    print(db.execute("SELECT count(*) FROM snn_frames").fetchone())
```

Giữ tệp SQLite trong `outputs/` cùng bản gốc và script tạo lại. Đừng dùng số
liệu `archived` làm bằng chứng cho một lần chạy mới khi chưa có log mới.
