# Giai đoạn 2 — log heading và benchmark CX–MB–gate

## Material Passport

- **Dữ liệu demo:** hoàn toàn mô phỏng, tạo bởi `phase2/synthetic.py` với seed cố định.
- **Dữ liệu kiến cũ:** được giữ trong `source_data/`; không dùng làm nhãn gyro hoặc heading truth.
- **Trạng thái xác minh:** pipeline chạy và có kiểm thử; chưa đánh giá trên robot hoặc mô phỏng cảnh ảnh mới.
- **Kết luận được phép:** mã ghi log, kiểm tra split, hiệu chuẩn và so sánh phương pháp hoạt động trên bài toán kiểm tra có kiểm soát.

Nếu đề tài của bạn dùng mạng SNN, đường chạy chính là
[`SNN_EXPERIMENT.md`](SNN_EXPERIMENT.md): MATLAB PN–KC–EN sinh điểm EN từ
ảnh mô phỏng ở pose mới, tạo gyro ảo có drift, rồi đưa log sang benchmark
Python. `phase2.run_demo` dưới đây chỉ là kiểm tra hạ tầng thuần Python.

## Chạy bản demo

Từ thư mục `MB_ant`:

```powershell
python -m phase2.run_demo --seed 20261027
python -m phase2.validate_log phase2/results/synthetic_heading.jsonl
python -m unittest discover -s tests
```

Đầu ra nằm trong `phase2/results/`:

- `synthetic_heading.jsonl`: 12 lượt, mỗi lượt 120 frame; validation/test tách theo `route_id`.
- `benchmark_config.json`: ngưỡng chọn trên validation, seed và quy tắc khởi tạo.
- `test_metrics.csv`: circular MAE, p95, tỷ lệ lỗi >45° và số hiệu chỉnh gây hại.
- `test_trace.csv`: từng frame, dự đoán, góc thật, confidence và lý do gate.

Kết quả kiểm tra với seed mặc định: gyro-only MAE 25,94°, visual-only 8,15°,
hợp nhất không gate 2,84°, confidence gate 2,41°, full gate 1,07°.
**Đây là kết quả trên dữ liệu mô phỏng được thiết kế để thử drift, mất ảnh và
nhầm cảnh; không phải bằng chứng thực nghiệm về kiến, CX hay robot.** Thay
seed và điều kiện sẽ cho số khác. Đọc cả p95 và số hiệu chỉnh gây hại trong
`test_metrics.csv`, không chỉ MAE. Bộ demo đã được dùng để sửa lỗi và điều
chỉnh mã trong quá trình phát triển, nên con số test demo cũng không phải
ước lượng hiệu năng khách quan cho một mô hình đã khóa trước thí nghiệm.

## Log thật cần thu

Mỗi dòng JSONL là một thời điểm đồng bộ. Các trường chính:

| Trường | Ý nghĩa |
|---|---|
| `run_id`, `route_id`, `condition`, `split`, `origin` | Nhận diện lượt, tuyến, điều kiện và nguồn `simulator`/`robot` |
| `frame`, `timestamp_s` | Chỉ số liên tiếp và thời gian tăng đơn điệu |
| `gyro_delta_deg` | Góc quay tích phân từ frame trước; frame đầu bằng 0 |
| `heading_truth_deg` | Góc thật từ pose mô phỏng hoặc hệ đo ngoài; bắt buộc ở validation/test |
| `sector_angles_deg`, `familiarity_scores` | Điểm quen thuộc cho từng góc; điểm nhỏ hơn là quen hơn; hai mảng rỗng khi mất ảnh |
| `visual_source` | `legacy_en_scan`, `sparse_kc_proxy`, hoặc tên decoder MBON cụ thể |
| `context_ok`, `context_source` | Kiểm tra độc lập về vị trí/tiến độ tuyến, không được tính từ `heading_truth_deg` |
| `odom_place_id`, `matched_place_id` | Bằng chứng thô cho cổng ngữ cảnh nếu dùng tiến độ tuyến |
| `image_path`, `event_type` | Đường dẫn ảnh và nhãn điều kiện để kiểm tra lỗi |

`phase2/capture.py` cung cấp `HeadingLogWriter` để nối vào vòng lặp cảm biến.
Nếu IMU có tần số cao hơn camera, dùng `phase2.sync.gyro_deltas_at_frames`
để tích phân vận tốc quay z (độ/giây) giữa hai mốc ảnh. Hai luồng phải dùng
cùng đồng hồ; đổi dấu/trục IMU cho đúng quy ước góc trước khi tích phân.
Ví dụ rút gọn:

```python
from pathlib import Path
from phase2.capture import HeadingLogWriter

with HeadingLogWriter(
    Path("phase2/data/route_a.jsonl"), run_id="route-a-run-1",
    route_id="route-a", split="validation", origin="simulator",
    condition="normal", visual_source="legacy_en_scan",
    context_source="odometry_place_progress",
) as log:
    log.append(
        timestamp_s=0.0, gyro_delta_deg=0.0, heading_truth_deg=70.0,
        sector_angles_deg=[68.0, 70.0, 72.0],
        familiarity_scores=[5.0, 1.0, 4.0], context_ok=True,
        odom_place_id=1.0, matched_place_id=1.1,
    )
```

Gắn `gyro_delta_deg`, ảnh và `heading_truth_deg` vào cùng mốc thời gian trước
khi ghi. Không lấy tiếp tuyến của tuyến kiến làm góc thật thân robot. Khi dùng
robot thật, `heading_truth_deg` cần một nguồn đánh giá độc lập; ở deployment
có thể để `null`, nhưng lượt đó không thể tính MAE. Mã hiện chưa gắn với một
camera, IMU, ROS topic hay hệ motion capture cụ thể.

## Tạo bằng chứng thị giác

Ưu tiên dùng mạng MB/EN MATLAB gốc để quét các góc tại **cùng một pose mới**,
ghi lại 61 điểm EN và góc quét. `outputs/legacy_sector_scores.csv` cho thấy
cách biểu diễn này trên log cũ, nhưng không có gyro/truth để chấm điểm fusion.

`phase2/visual_memory.py` có một encoder KC thưa và bộ nhớ familiarity đơn
giản để thử giao diện Python. Nó chọn khoảng 1,27% KC hoạt động, dùng 10 PN/KC
theo kết nối ngẫu nhiên và đo khoảng cách mã thưa đến ảnh đã học. Đây là
**proxy không spiking**, không tương đương mạch PN–KC–EN anti-Hebbian của mã
MATLAB và chưa phải MBON sinh học theo sector. `preprocess_proxy_panorama`
cũng không tái hiện `adapthisteq`; cùng một phép tiền xử lý phải được dùng ở
cả pha học và pha hỏi. Không đánh giá trên chính ảnh đã đưa vào bộ nhớ.

## So sánh có kiểm soát

```powershell
python -m phase2.validate_log phase2/data/my_experiment.jsonl
python -m phase2.benchmark phase2/data/my_experiment.jsonl --output phase2/results/my_experiment
```

Benchmark gồm `gyro_only`, `visual_only`, `ungated`, `confidence`, `full_gate`.
Hai phương pháp có confidence chọn ngưỡng trên validation bằng mục tiêu
`MAE + 0,25 × p95`; test chỉ được chấm sau khi chọn xong. Không để ảnh gần
nhau hoặc cùng `route_id` lọt qua cả hai split. Các phương pháp được khởi tạo
cùng một góc ban đầu lệch 5° so với truth, mô phỏng bước hiệu chuẩn ban đầu;
truth không được đưa vào các cập nhật về sau. Khi có dữ liệu thật, báo cáo
theo từng điều kiện drift, cảnh lặp, mất ảnh và lệch tuyến, kèm số lần sửa sai
lớn và tỷ lệ đi hết tuyến không reset.

## Việc còn phụ thuộc dữ liệu mới

1. Chạy pilot `run_snn_heading_scan` trên hai tuyến, xác nhận MATLAB/GPU,
   đơn vị và dấu góc, log JSONL và metadata mạng SNN.
2. Tạo ảnh quét tại pose mới, tách ảnh học với ảnh kiểm tra; ghi score của mạng
   MB/EN và ID của bộ nhớ KC được khớp.
3. Tính `context_ok` từ odometry hoặc tiến độ tuyến độc lập. Lưu cả bằng chứng
   thô; không dùng heading truth để quyết định gate.
4. Thu các tuyến validation và test riêng rồi mới báo cáo kết quả fusion thật.
