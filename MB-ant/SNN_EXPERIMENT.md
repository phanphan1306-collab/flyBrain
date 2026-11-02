# Thí nghiệm heading bằng mạng SNN PN–KC–EN

> **Tham khảo cũ:** thí nghiệm trong tệp này dùng dữ liệu tuyến kiến và gyro
> ảo. Baseline FlyBrain hiện tại không dùng camera hoặc robot và được mô tả
> trong [`FLY_SNN_BASELINE.md`](FLY_SNN_BASELINE.md).

## Mục đích

Thí nghiệm này dùng chính mã spiking PN/KC/EN MATLAB trong `source_data/`,
không dùng `phase2/visual_memory.py` làm kết quả chính. Mạng SNN tạo số spike
EN cho các góc camera quét. Một tác tử ảo được đặt tại pose **chưa dùng để
học** (trung điểm giữa hai ảnh tuyến và lệch ngang 3 cm theo mặc định).
Heading thật là hướng cơ thể được gán cho tác tử ảo trong mô phỏng; gyro có
bias và nhiễu được tạo từ biến thiên của heading đó. Đây là một thí nghiệm
**mô phỏng SNN**, không phải phép đo IMU hay góc đầu của kiến thật.

## Chạy thí điểm trong MATLAB

Yêu cầu: MATLAB có GPU/Parallel Computing Toolbox, Image Processing Toolbox,
`jsonencode`, cùng các hàm trong `source_data/`. Từ bất kỳ thư mục làm việc:

```matlab
addpath('U:\FlyBrain_MB_CX\FlyBrain01\FlyBrain\MB_ant\experiment')
logPath = run_snn_heading_scan([1 2]);
```

Đường dẫn trong `addpath` là theo ảnh VS Code bạn gửi; nếu đã chuyển dự án,
thay bằng đường dẫn đến `MB_ant\experiment` trên máy. Mặc định kiến 1 là
validation và kiến 2 là test. Đây chỉ là **pilot hai tuyến**; chưa đủ để báo
cáo kết luận tổng quát. Script dùng 20.000 KC, lấy tối đa 25 pose/tuyến và
13 góc/pose (bước 10°), nên có thể chạy lâu. Có thể giảm tải thí điểm:

```matlab
opts = struct('max_frames', 8, 'scan_step_deg', 20, 'route_stride', 8, 'seed', 73);
logPath = run_snn_heading_scan([1 2], ...
    'U:\FlyBrain_MB_CX\FlyBrain01\FlyBrain\MB_ant\phase2\results\snn_pilot', opts);
```

Sau khi pilot chạy đúng, dùng nhiều `antIds` hơn (ví dụ `[1 2 3 4 5 6]`),
khóa tùy chọn trước khi nhìn test, và mở rộng độ phân giải quét nếu thời gian
GPU cho phép. Thứ tự `antIds` quyết định split: nửa đầu validation, nửa sau
test. Mỗi kiến được huấn luyện một mạng mới với seed lưu trong metadata.

Script tạo `snn_heading_seed<seed>.jsonl` và một tệp
`snn_ant<id>_seed<seed>_metadata.mat` cho mỗi kiến trong thư mục đầu ra.
Metadata chứa vị trí/heading thật của pose kiểm tra, gyro, seed, kết nối
PN–KC dạng sparse, trọng số KC–EN và mã KC của ảnh học. Nếu đường dẫn log
đã tồn tại, script dừng để tránh ghi đè một lần chạy trước.

## Chấm điểm bằng Python trong VS Code

Từ terminal đang ở thư mục `MB_ant`, với đường dẫn đầu ra mặc định:

```bat
py -m phase2.validate_log phase2\results\snn\snn_heading_seed20261027.jsonl
py -m phase2.benchmark phase2\results\snn\snn_heading_seed20261027.jsonl --output phase2\results\snn_benchmark
```

Mở `phase2\results\snn_benchmark\test_metrics.csv` để xem circular MAE,
p95, tỷ lệ lỗi >45° và số hiệu chỉnh gây hại. `benchmark_config.json` ghi
ngưỡng được chọn **chỉ trên validation**; `test_trace.csv` ghi từng quyết
định gate. Báo cáo phải ghi rõ `origin=simulator` và
`visual_source=snn_en_scan`.

## Quan hệ với mô hình sinh học và dữ liệu cũ

- Mã SNN gốc có **một EN**. Một dãy điểm EN theo các góc là evidence theo
  heading sector ở tầng thuật toán; nó chưa tạo nhiều MBON chuyên biệt.
- Kết nối PN–KC ngẫu nhiên của 15 `Record*.mat` cũ không thể khôi phục đúng
  vì seed ban đầu không được lưu. Script này huấn luyện lại SNN với seed mới,
  lưu cả kết nối và trọng số. Kết quả mới không được trộn với baseline cũ.
- Cổng ngữ cảnh so sánh vị trí bộ nhớ KC được khớp với tiến độ tuyến từ
  odometry ảo có nhiễu. Nó không đọc `heading_truth_deg`, nhưng vẫn là một
  tín hiệu **mô phỏng** và cần kiểm tra lỗi khớp cảnh.
- Pose kiểm tra không trùng ảnh học, nhưng vẫn nằm gần cùng tuyến. Cần thêm
  phép thử cảnh lặp, thay đổi ánh sáng và vị trí lệch tuyến mạnh hơn trước
  khi kết luận tính tổng quát.
- MATLAB/GPU chưa có trong môi trường làm việc hiện tại của mình, vì vậy
  script MATLAB chưa được chạy xác minh. Hãy chạy pilot 8 frame trước, gửi
  log hoặc lỗi MATLAB nếu có để mình chỉnh tiếp.
