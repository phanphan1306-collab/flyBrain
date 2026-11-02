# Baseline dữ liệu cũ MB_ant

Báo cáo được tạo bằng `python finalize_legacy.py` từ bản sao nguyên trạng
trong `source_data/`. `outputs/source_manifest.csv` lưu SHA-256 của từng tệp.

## Kết quả đã kiểm tra

- 98 tệp nguồn; 15 tuyến kiến; 1223 ảnh tuyến.
- EN trung bình theo ảnh: 12.663 spike trước học, 0.752 spike sau học. 781/1223 ảnh hậu học (63.9%) có EN bằng 0.
- KC hoạt động trung bình 1.27% trên 20.000 KC.
- 225/1231 lần quét (18.3%) có ít nhất hai góc cùng đạt EN thấp nhất.
- 38 lần đặt kiến trở lại tuyến; 1/15 tuyến không cần reset.
- Vị trí cuối của cả 15 tuyến nằm trong bán kính đích theo script gốc; điều này không xác lập tỷ lệ tự dẫn đường vì có reset.
- `KC_fired_total` có kích thước sai ở 15/15 bản ghi. Không dùng trường này để phân tích KC theo pha.

| Kiến | Ảnh tuyến | EN trước | EN sau | EN=0 sau học | Quét đồng cực tiểu | Reset |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 80 | 12.59 | 0.84 | 60.0% | 5/78 | 1 |
| 2 | 82 | 12.70 | 0.67 | 67.1% | 20/84 | 3 |
| 3 | 82 | 12.56 | 0.79 | 62.2% | 37/83 | 3 |
| 4 | 84 | 12.58 | 0.70 | 66.7% | 13/89 | 4 |
| 5 | 84 | 12.74 | 0.81 | 56.0% | 2/84 | 2 |
| 6 | 82 | 12.63 | 0.73 | 64.6% | 10/83 | 3 |
| 7 | 82 | 12.51 | 0.70 | 67.1% | 29/83 | 3 |
| 8 | 80 | 12.61 | 0.90 | 57.5% | 18/79 | 3 |
| 9 | 83 | 12.57 | 0.60 | 75.9% | 6/83 | 2 |
| 10 | 80 | 12.86 | 0.68 | 67.5% | 6/78 | 1 |
| 11 | 77 | 12.68 | 0.86 | 57.1% | 2/75 | 0 |
| 12 | 79 | 12.58 | 0.97 | 54.4% | 25/85 | 4 |
| 13 | 87 | 12.79 | 0.57 | 70.1% | 26/86 | 3 |
| 14 | 82 | 12.91 | 0.74 | 64.6% | 11/81 | 3 |
| 15 | 79 | 12.62 | 0.75 | 65.8% | 15/80 | 3 |

## Cách hiểu và giới hạn

- EN giảm trên ảnh đã học là chỉ dấu familiarity của mô hình, chưa đo độ chính xác heading hoặc hiệu quả sửa gyro drift.
- Mỗi lần quét có 61 góc cách nhau 2°. `legacy_sector_scores.csv` dựng lại góc từ heading đã chọn và vị trí cực tiểu; nó là điểm của một EN quét nhiều góc, chưa phải các MBON sector độc lập.
- Script gốc làm tròn heading rồi dựng lại tọa độ tuyến theo bước 0,1 m. Dữ liệu này không có chuỗi gyro, timestamp, pose/heading truth theo thời gian hoặc nhãn nhầm cảnh để hiệu chuẩn reliability gate.
- `error_rate` gốc bằng số reset chia số transition đã lưu. Khi lệch tuyến, kiến được đặt lại bằng vị trí tham chiếu; không báo cáo `error_rate` như xác suất thất bại tự điều hướng.
- Kết quả được tính từ dữ liệu lưu sẵn. Chưa chạy lại script MATLAB/GPU; random seed và lý do dừng không được lưu trong bản ghi cũ.

## Đầu ra cố định

- `outputs/source_manifest.csv`: tên, kích thước, SHA-256 nguồn.
- `outputs/dataset_audit.csv`: các phép đo và kiểm tra từng tuyến.
- `outputs/legacy_sector_scores.csv`: 61 score/góc cho mỗi lần quét.
- `outputs/legacy_summary.json`: tổng hợp máy đọc được.
