# Kế hoạch tiếp theo: FlyBrain chỉ dùng SNN

> **Lưu trữ hướng nghiên cứu trước:** kế hoạch dưới đây áp dụng cho tuyến
> kiến/ảnh toàn cảnh Ardin. Hướng FlyBrain hiện tại dùng tín hiệu tổng hợp,
> không camera hoặc robot; xem [`FLY_SNN_BASELINE.md`](FLY_SNN_BASELINE.md).

## Câu hỏi nghiên cứu đầu tiên

Sau khi học ảnh toàn cảnh dọc một tuyến, mạng PN–KC–EN kiểu Ardin (2016)
có chọn được hướng tham chiếu của tuyến ở **vị trí chưa dùng để học** không?
Đây là bài toán mô phỏng thị giác. Không có robot, gyro, IMU, odometry hay
path integration trong phép đo chính.

**Giả thuyết cần kiểm tra:** hướng có số spike EN thấp nhất thường nằm gần
hướng tiếp tuyến của tuyến hơn lựa chọn ngẫu nhiên trên cùng lưới góc.
Hướng tiếp tuyến chỉ là nhãn đánh giá trong mô phỏng, không được dùng để
chọn hoặc căn giữa các góc đem vào SNN.

## Bằng chứng hiện có và giới hạn

- `outputs/ant1_learning/learning_summary.csv` ghi EN trung bình của 80 ảnh
  trước/sau học là 12.59/0.84 spike. Đây là kiểm tra học độ quen thuộc trên
  ảnh đã có, chưa phải độ chính xác hướng ở ảnh chưa học.
- `outputs/legacy_summary.json` và `outputs/legacy_sector_scores.csv` mô tả
  bản chạy cũ, không thay cho log có seed và cấu hình của lần chạy mới.
- `experiment/run_snn_heading_scan.m` hiện căn lưới góc theo `gyroHeading`
  và tính `context_ok` bằng odometry ảo. Vì vậy **không dùng nguyên trạng làm
  thí nghiệm chính SNN-only**.
- `outputs/snn_results.sqlite` đã có cấu trúc để lưu kết quả, nhưng chưa có
  log SNN mới của pilot này.

## Việc cần tạo trước khi chạy

1. Tạo `experiment/run_snn_visual_scan.m` bằng cách tái sử dụng phần huấn
   luyện PN–KC–EN và `GPU_network_fun` từ script hiện có. Dùng một lưới góc
   cố định theo tọa độ thế giới, ví dụ `0:30:330` cho pilot, độc lập với
   hướng tham chiếu. Tại mỗi vị trí kiểm tra, dựng ảnh ở cả 12 góc và lưu
   toàn bộ số spike EN. Không đưa nhãn hướng vào hàm dự đoán.
2. Tách tệp cấu hình và log: lưu ant ID, seed, số ảnh học, danh sách chỉ số
   ảnh học/kiểm tra, bước góc, vị trí, góc tham chiếu, EN của từng góc,
   số KC hoạt động, thời gian chạy và phiên bản mã. Lưu `.mat` metadata/model
   ở ngoài SQLite. Ghi JSONL theo từng vị trí và chỉ công bố file hoàn chỉnh
   khi chạy xong.
3. Tạo `experiment/analyze_snn_visual_scan.py` để tính sai số góc tròn,
   tỷ lệ trong một bước góc, tỷ lệ đồng cực tiểu EN và chênh lệch EN giữa
   hai góc tốt nhất. Vẽ đường EN theo góc cho từng vị trí. Nếu đồng cực tiểu,
   báo cáo rõ quy tắc chọn góc và cả tỷ lệ tie; không che tie bằng `min`.
4. Chỉnh `snn_database.py` để nhập log SNN-only mới hoặc dùng schema hiện
   có với các trường bắt buộc đúng nghĩa. Không điền giá trị gyro/odometry
   giả để thỏa schema; nếu schema buộc các trường đó, tạo migration nullable.

## Thứ tự chạy trên máy có GPU

### Pilot 1 — kiểm tra pipeline

- 1 tuyến (ant 1), 1 seed, tối đa 8 vị trí ở trung điểm giữa hai ảnh đã học.
- 12 góc cố định `0:30:330` mỗi vị trí: 96 lượt suy luận SNN, cộng huấn luyện.
- Xuất log JSONL, `.mat` metadata và một đồ thị EN-vs-angle. Kiểm tra tất cả
  góc đã được quét; không thiếu frame/score; hai lần chạy cùng seed cho kết
  quả lặp lại trong cùng môi trường.
- Ở pilot này chỉ xác nhận mã chạy và dữ liệu đủ; chưa kết luận hiệu quả.

### Pilot 2 — đo độ chính xác

- Giữ cấu hình đã chốt, chạy các tuyến độc lập với ít nhất 3 seed mỗi tuyến.
- Mỗi tuyến học bằng ảnh của chính tuyến đó; đánh giá trên vị trí trung điểm
  chưa dùng để học. Chia tuyến thành validation/test để chọn mọi tham số và
  giữ test chưa động tới cho đến khi cố định phân tích.
- Báo cáo median/mean sai số góc tròn theo từng tuyến rồi tổng hợp theo tuyến,
  tỷ lệ chọn trong một bước góc, tỷ lệ tie và khoảng tin cậy qua tuyến/seed.
  So sánh với chọn ngẫu nhiên trên cùng lưới góc (mức tham chiếu của bài toán,
  không phải một mạng khác).

### Pilot 3 — thử độ bền và ý tưởng mới

- Chỉ làm sau khi Pilot 2 cho tín hiệu rõ: thay đổi vị trí ngang, thay đổi
  độ sáng hoặc độ phân giải ảnh. Đổi từng yếu tố một, giữ split/seed nhất quán.
- Nếu EN đồng cực tiểu nhiều, thử một readout dựa trên thời gian spike hoặc
  phân bố spike của chính mạng SNN; so với readout EN thấp nhất của Ardin trên
  cùng ảnh và seed. Đây là hướng đóng góp có thể kiểm chứng.
- LearnPI là mô hình path integration **rate-based**, nên không dùng làm
  baseline SNN cho bài toán chọn hướng bằng ảnh. Chỉ mở nhánh đó nếu đổi câu
  hỏi nghiên cứu sang path integration và chủ động xây phiên bản spiking.

## Điều kiện để chuyển bước

- Qua Pilot 1 khi log có đủ 8 x 12 score EN, seed/cấu hình/nguồn dữ liệu,
  và có thể tái tạo biểu đồ từ log mà không cần ảnh chụp màn hình.
- Qua Pilot 2 khi kết quả được tính trên vị trí chưa học, có báo cáo tie và
  sai số theo tuyến, không dùng nhãn hướng trong chọn góc.
- Chỉ nêu đóng góp mới sau khi so sánh SNN Ardin gốc và biến thể trên cùng
  tập kiểm tra và phân tích trường hợp thất bại.

## Đầu ra cần giữ

`config.json`, `snn_visual_scan_seed*.jsonl`, `*_metadata.mat`, tệp tổng
hợp CSV, đồ thị EN-vs-angle, nhật ký phiên bản mã và `outputs/snn_results.sqlite`.
Ảnh JPG/FIG cũ vẫn là bằng chứng trực quan của bản chạy cũ, không thay thế
log định lượng cho thí nghiệm mới.

## Nguồn chính

- Ardin et al. (2016), *PLOS Computational Biology*,
  https://doi.org/10.1371/journal.pcbi.1004683
- LearnPI, *eLife*, https://elifesciences.org/articles/69841
