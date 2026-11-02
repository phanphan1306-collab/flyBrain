# FlyBrain: baseline SNN mô phỏng MB–CX/EB

## Trạng thái và phạm vi

Đây là **baseline kỹ thuật tối giản do dự án tự viết**, chạy được trên CPU với
NumPy. Nó **không phải** bản tái lập định lượng của Ardin (2016), Stentiford
(2024) hoặc Goulard (2023), và kết quả dưới đây **chưa phải kết quả paper**.
Không sử dụng camera, robot, ảnh, dữ liệu ruồi thật hoặc dữ liệu kiến trong
benchmark mới. Mọi đầu vào là spike/cue do bộ sinh tổng hợp tạo ra. `MB_ant`
cũ được giữ nguyên làm tham chiếu độc lập của mô hình Ardin.

## Vì sao chọn ba nguồn này

1. **Stentiford et al. (2024)** cung cấp CX ring-attractor SNN để ước lượng
   hướng và có mã công khai. Đây là baseline tài liệu sát nhất cho EB/CX-SNN.
   Bài: https://doi.org/10.1371/journal.pcbi.1011913 ; mã:
   https://github.com/stenti/stentiford_cx_ra (HEAD đã kiểm tra:
   `eff923fbf5e8b5ad4b89257310753496536fbc99`). Bản gốc dùng PyGeNN và
   môi trường Python 3.10; trên máy này chưa có PyGeNN/GPU phù hợp, nên **chưa
   chạy lại mã tác giả**.
2. **Ardin et al. (2016)** là tham chiếu SNN cho mã hóa thưa KC và học MB.
   Bài: https://doi.org/10.1371/journal.pcbi.1004683 ; mã/dữ liệu:
   https://doi.org/10.5061/dryad.pf66v. Bản MATLAB trong `source_data/` đã
   có dữ liệu chạy trên máy GPU khác, xem `LEGACY_BASELINE.md`. Module MB mới
   trong `fly_snn/model.py` học cue→goal bằng nhiều neuron đầu ra, **khác**
   một EN và cơ chế synapse của Ardin; chỉ có cấu trúc mã thưa lấy cảm hứng.
3. **Goulard et al. (2023)** đã tích hợp MB với CX, vì vậy không thể tuyên bố
   bản thân việc ghép hai vùng là mới. Mỗi neuron của mô hình đó là
   **firing-rate**, không phải SNN. Bài:
   https://doi.org/10.1371/journal.pcbi.1011480 ; mã:
   https://github.com/RomanGoulard/CXModel_SpaceRep_Data (HEAD đã kiểm tra:
   `b45fe1eb8de0f0689c7d8fa584e169a41b439dfe`).

## Ba bước đã hiện thực

| Bước | Mã | Cách kiểm tra |
| --- | --- | --- |
| EB/CX-SNN | `HeadingRingSNN`: vòng neuron phát spike, ba phép chiếu kết nối chọn bởi tín hiệu quay `−1/0/+1`, cue mốc tùy chọn, ức chế cạnh tranh | Sai số hướng, tỷ lệ đồng cực đại, cập nhật qua biên 0/360° |
| MB-SNN | `ContextMemorySNN`: cue spike → KC thưa → quần thể goal-spike với Hebb có tín hiệu thưởng; `DirectMemorySNN` là đối chứng bỏ KC | Tỷ lệ nhớ đúng goal, KC hoạt động, tie |
| Ghép MB–CX | `SteeringSNN`: trọng số sin cố định biến spike goal và heading thành spike rẽ trái/phải | Độ chính xác quyết định rẽ; so cùng episode/seed |

`SteeringSNN` là giả thuyết mạch tính toán kiểu PFL; không nhận là bản sao
PFL3 trong ruồi. Quy tắc winner-take-all của EB và phép đọc spike population
là các giản lược kỹ thuật. Tín hiệu quay tổng hợp là đầu vào mô phỏng, không
phải gyro. Ground truth chỉ dùng tạo test cue và tính metric; hàm
`HeadingRingSNN.update()` không nhận ground-truth heading.

## Chạy được ngay

Từ thư mục `MB_ant`:

```powershell
python -m unittest discover -s tests
python -m fly_snn.benchmark --output outputs/fly_snn_pilot --seeds 11 23 37
python -m fly_snn.sweep --output outputs/fly_snn_sweep --n-seeds 20
python -m fly_snn.plot_results outputs/fly_snn_sweep
python -m fly_snn.paired_analysis outputs/fly_snn_sweep
python -m fly_snn.provenance outputs/fly_snn_sweep
```

Mỗi output path phải là thư mục **chưa tồn tại**, tránh ghi đè một lần chạy.
`trials.csv` chứa từng bước; `summary_by_seed.csv` là đơn vị tổng hợp độc lập;
`summary_aggregate.csv` có trung bình và SD giữa seed; `manifest.json` giữ
cấu hình; `paired_deltas.csv` có chênh lệch MB−direct theo seed và khoảng
bootstrap mô tả; `provenance.json` lưu SHA-256 của mã và kết quả. Mã chỉ cần
NumPy và Matplotlib cho hình.

## Quy trình đánh giá

- Context được tạo một lần cho mỗi seed; mẫu cue sạch là dữ liệu học. Lần
  kiểm tra dùng nhiễu cue mới trên **cùng context**, vì không thể yêu cầu mô
  hình nhớ goal của context chưa từng được dạy. Đây là kiểm tra độ bền trí nhớ,
  chưa phải tổng quát hóa sang context mới.
- Mỗi episode có true heading và true turn riêng để chấm điểm. Mạng chỉ nhận
  noisy turn, cue mốc có thể bị thiếu/nhiễu và cue context bị đảo bit. Mọi arm
  nhận đúng cùng episode, cue và EB activity. Bước đầu có mốc chuẩn để khởi
  tạo EB; bước này được giữ trong `trials.csv` nhưng **loại khỏi mọi metric**.
- Arm: `cx_fixed` (một goal cố định học từ tập train), `direct_cx` (cue→goal
  không KC), `mb_cx` (KC thưa), `mb_shuffled` (cùng cấu trúc MB nhưng nhãn
  goal bị hoán vị trước huấn luyện).
- Khi goal đối diện chính xác 180° với heading, hai hướng rẽ ngắn như nhau;
  bước đó không tính accuracy rẽ. Quy tắc hòa spike là chọn chỉ số đầu tiên
  và tỷ lệ tie được ghi riêng.
- Năm điều kiện định trước ở `fly_snn/sweep.py`: sạch, chuẩn, cue nhiễu,
  thiếu mốc, nhiễu kết hợp. Các seed 100–119 là pilot phát triển, chưa khóa
  như tập test cho paper.

## Kết quả pilot trên máy này (20 seed, chỉ để kiểm tra mã)

| Điều kiện | Sai số hướng EB (sector) | Goal accuracy trực tiếp | Goal accuracy MB | Turn accuracy trực tiếp | Turn accuracy MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sạch | 0.000 | 0.981 | 0.988 | 0.732 | 0.895 |
| Chuẩn | 0.120 | 0.967 | 0.992 | 0.746 | 0.895 |
| Cue nhiễu | 0.120 | 0.964 | 0.921 | 0.746 | 0.865 |
| Thiếu mốc | 0.353 | 0.967 | 0.992 | 0.744 | 0.874 |
| Kết hợp | 0.388 | 0.964 | 0.921 | 0.745 | 0.843 |

Đơn vị sector = 22.5°. Các số trên chỉ đúng cho bài toán tổng hợp và cấu hình
đã chọn; số KC/tham số của hai arm không bằng nhau. Không dùng bảng này để
kết luận vượt trội khoa học hoặc so trực tiếp với số liệu bài báo gốc.
Ở điều kiện cue nhiễu mạnh, **direct có goal accuracy cao hơn MB** (0.964 so
với 0.921), dù MB vẫn có turn accuracy cao hơn. Hai metric đo hai khía cạnh
khác nhau; không gộp thành một tuyên bố MB luôn tốt hơn.

## Còn thiếu để thành baseline đủ mạnh cho paper

So sánh bốn họ bộ nhớ trên cùng episode tổng hợp (direct, giãn nở dày,
quét ngưỡng MB thưa và direct khớp số synapse học) được mô tả trong
[`FOUR_ARM_COMPARISON.md`](FOUR_ARM_COMPARISON.md). Đây là pilot phát triển
riêng; bảng kết quả 20 seed phía trên vẫn là cấu hình cũ.

1. **Tái lập độc lập mã Stentiford** trên môi trường PyGeNN/GPU đúng phiên
   bản, và ghi cấu hình/commit/log của lần chạy; dùng các kích thích thanh
   mô phỏng của họ trước khi thay input. Chưa hoàn thành trên máy này.
2. **Chạy mới Ardin MATLAB** với seed được lưu và xuất spike theo từng ảnh;
   dữ liệu cũ thiếu seed mô hình, nên không thể gọi là tái lập có kiểm soát.
3. **So sánh khớp tài nguyên** (số neuron, synapse, spike, thời gian) giữa
   direct và KC-sparse, rồi khóa hyperparameter trên validation seeds và
   chạy test seeds mới đúng một lần.
4. **Tăng tính sinh học**: thay winner-take-all giản lược bằng interneuron
   spiking tường minh; khảo sát nhạy cảm tham số; kiểm tra cấu trúc/động học
   với dữ liệu ruồi. Hiện chưa có dữ liệu hành vi/điện sinh lý trong benchmark.
5. **Định nghĩa đóng góp hẹp**: MB→goal/CX có cải thiện gì trước cue mơ hồ
   hoặc thay đổi ngữ cảnh khi EB chỉ có noisy turn? Không gọi đây là mô hình
   não ruồi đã được xác thực.

