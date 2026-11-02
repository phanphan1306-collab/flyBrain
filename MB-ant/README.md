# MB_ant — baseline dữ liệu kiến

Hướng FlyBrain hiện tại là **mô hình SNN MB–CX/EB trên tín hiệu tổng hợp để
nghiên cứu**, không cần camera hoặc robot. Mã pilot chạy được, kết quả và giới
hạn ở [`FLY_SNN_BASELINE.md`](FLY_SNN_BASELINE.md). Phần còn lại của README
này là dữ liệu và thử nghiệm MB_ant cũ, được giữ để truy vết nguồn gốc.

`source_data/` là bản sao đầy đủ của [thư mục MB_ant trên Drive](https://drive.google.com/drive/folders/1iM_u-DmcdF918GHkFCHI1BtMYNA6xQQ7).
`LEGACY_BASELINE.md` là kết luận đã chốt từ 15 tuyến. Dữ liệu gốc và kết quả
phân tích được giữ riêng; không ghi đè tệp gốc khi chạy mã mới.

Giai đoạn tiếp theo đã có mã ghi log, mô phỏng và benchmark trong
[`PHASE2.md`](PHASE2.md). Chạy `python -m phase2.run_demo` để kiểm tra pipeline;
chỉ dùng kết quả từ log `origin=simulator` hoặc `origin=robot` cùng ground truth
độc lập khi đánh giá hiệu quả heading thực.

Với hướng **mạng SNN**, xem [`SNN_EXPERIMENT.md`](SNN_EXPERIMENT.md): script
MATLAB mới quét góc bằng PN–KC–EN ở pose mô phỏng chưa học, tạo gyro ảo có
drift, rồi xuất JSONL cho benchmark Python. Demo `phase2.run_demo` chỉ là
bài kiểm tra thuật toán thuần Python, không dùng output SNN.

## Chạy

```powershell
python -m pip install -r requirements.txt
python finalize_legacy.py
python -m unittest discover -s tests
```

Yêu cầu: Python, NumPy, SciPy; giai đoạn 2 còn dùng Pillow. Lệnh
`finalize_legacy.py` kiểm tra 98 tệp nguồn,
đối chiếu SHA-256 với manifest đã chốt, kiểm toán 15 tuyến và tạo lại:

- `LEGACY_BASELINE.md`: kết quả và giới hạn dữ liệu.
- `outputs/source_manifest.csv`: checksum của tệp gốc.
- `outputs/dataset_audit.csv`: phép đo theo từng kiến.
- `outputs/legacy_sector_scores.csv`: 61 score EN cho mỗi lần quét góc.
- `outputs/legacy_summary.json`: tổng hợp máy đọc được.

Nếu tệp nguồn thay đổi, lệnh dừng thay vì âm thầm thay baseline. Chỉ dùng
`--refresh-manifest` sau khi chủ động thay bộ dữ liệu và kiểm tra nguồn.

Góc sector trong CSV được dựng lại từ lựa chọn cực tiểu của script gốc; tâm
dải góc ở lần quét đầu của cả 15 kiến khớp heading lưu trong `Raw_images`.
File này là đầu vào thị giác cho bước fusion, không phải dữ liệu gyro hay
ground truth heading.

`experiment/ant_route_following_test_fixed.m` là bản vá tối thiểu của script
MATLAB/GPU gốc; cần MATLAB Parallel Computing Toolbox và GPU CUDA để chạy.
Khi chạy MATLAB, đặt thư mục hiện tại là `source_data`, thêm cả `source_data`
và `experiment` vào MATLAB path, rồi gọi
`ant_route_following_test_fixed`. Kết quả mới được ghi vào
`experiment_results/`, không ghi đè `source_data/result/`.

## Diễn giải đúng dữ liệu cũ

Kết quả và bảng 15 tuyến nằm trong `LEGACY_BASELINE.md`. Các số này mô tả
mô hình MB định hướng thị giác ở kiến, chưa đo hiệu quả sửa gyro drift.

- `Raw_images_ant*_route1.mat`: ảnh tuyến và heading của **đoạn đường đã tạo
  lại**, không phải chuỗi quan sát gyro của robot. Script gốc làm tròn hướng
  và tạo lại tọa độ bằng bước 0,1 m; `img_pos` vì vậy không còn là điểm tuyến
  nguyên thủy.
- `Record*_route1.mat`: số PN/KC hoạt động và số spike EN ở ba pha. EN thấp
  sau học là tín hiệu familiarity của mô hình. `KC_fired_total` của bản gốc
  có lỗi chỉ số; không dùng trường này để suy luận sparsity theo pha.
- `ant*route1.mat`: trajectory điều hướng, dãy EN khi quét góc, cùng số lần
  được đặt trở lại tuyến. `error_rate` chịu ảnh hưởng của việc đặt lại; điểm
  cuối gần tổ không chứng minh robot tự đi hết tuyến.
- Mô hình cũ quét nhiều góc qua **một EN**. Có thể xem điểm EN theo góc là
  bằng chứng theo heading sector ở tầng thuật toán. Chưa có nhiều MBON riêng
  cho các sector trong mã gốc.

## Nguyên mẫu cho giai đoạn sau baseline

`HeadingFusion.update(gyro_delta_deg, sector_deg, visual_scores)` tích phân
độ quay trên vòng tròn, đo độ rõ của cực tiểu familiarity, rồi sửa heading
có giới hạn. Khi thị giác mơ hồ, nó giữ dự đoán gyro. Một bất đồng góc lớn
cần xuất hiện lặp lại và phải được xác nhận bằng `context_ok=True` từ nguồn
độc lập như ước lượng vị trí/route progress. Đây là trạng thái CX lấy cảm
hứng ở mức tính toán; chưa mô phỏng vòng E-PG/P-EN sinh học. Các ngưỡng mặc
định chỉ là giá trị khởi đầu cho thử nghiệm, chưa được hiệu chuẩn trên dữ liệu.

## Bước thực nghiệm tiếp theo

1. **Tạo log đúng cho heading.** Ở mỗi thời điểm ghi `timestamp`, `gyro_delta`,
   `heading_truth`, ảnh toàn cảnh, vị trí truth/odometry, route ID, góc quét,
   vector EN/MBON scores, và lý do gate nhận/từ chối. Trong mô phỏng,
   `heading_truth` lấy từ pose thật; tiếp tuyến tuyến chỉ là hướng di chuyển
   tham chiếu, không thay thế heading thân robot.
2. **Xây bằng chứng thị giác theo góc.** Trước tiên dùng quét ảnh qua MB/EN
   gốc để có một score cho từng sector. Sau đó mới thử decoder MBON nhiều
   sector; tách rõ phiên bản nào là mô hình mới. Thêm ảnh ở điểm ngoài tuyến,
   cảnh lặp lại, thay đổi sáng và góc nhìn để đo nhầm cảnh.
3. **Hiệu chuẩn confidence trên tuyến validation riêng.** Dùng margin giữa
   hai cực tiểu, entropy/độ rộng đỉnh, mức score tuyệt đối và nhất quán theo
   thời gian. Dữ liệu train/validation/test phải tách theo tuyến hoặc kiến,
   không trộn các ảnh gần nhau của cùng tuyến vào hai tập.
4. **So sánh ablation.** Gyro only; visual only; gyro + visual không gate;
   gyro + confidence; gyro + confidence + consistency. Thử drift bias, nhiễu
   gyro, cảnh giống nhau, mất ảnh và lệch khỏi tuyến. Báo cáo circular MAE,
   p95 lỗi góc, số lần sửa sai lớn, tỷ lệ đến đích *không reset*, và quãng
   đường/ thời gian. Chọn ngưỡng trên validation, báo cáo test đúng một lần.

## Các sửa lỗi trong bản MATLAB sao chép

`KC_fired_total` được cấp phát `numKC × (3*numTrain)` và ghi ba pha vào ba
khối cột riêng. Chỉ số `current_pos` sau reset được dùng đúng điểm đã chọn.
`random_seed`, `numKC`, `stop_reason` và `assisted_resets` được lưu để lần
chạy sau truy vết được. Bản vá không xóa cơ chế reset, vì dữ liệu cũ cần
được so sánh với hành vi gốc. Chưa chạy bản MATLAB/GPU này trên máy.
