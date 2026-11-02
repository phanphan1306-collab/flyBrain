# So sánh bốn họ bộ nhớ trên cùng CX

Đây là thí nghiệm phát triển trên cue tổng hợp của FlyBrain. Mọi arm nhận cùng
context, chuỗi quay, landmark và cue nhiễu trong mỗi seed. Vòng CX và mạch
steering giống nhau. Bước 0 chỉ khởi tạo heading và không được chấm điểm.

| Họ | Tên trong CSV | Mã hóa cue | Synapse học |
| --- | --- | --- | ---: |
| Direct–CX | `direct_cx` | 48 bit cue nối thẳng tới 16 goal | 768 |
| Giãn nở dày | `dense_expansion_cx` | 512 neuron trung gian, 6 input/neuron, ngưỡng 1 | 8.192 |
| MB thưa | `mb_sparse_t3_cx`, `mb_sparse_t4_cx`, `mb_sparse_t5_cx` | Cùng 512 neuron và **cùng wiring theo seed**, lần lượt ngưỡng 3, 4, 5 | 8.192 mỗi arm |
| Direct khớp ngân sách | `direct_budget_cx` | Lặp 48 bit cue trên 512 kênh direct, không tạo đặc trưng phi tuyến | 8.192 |

Ngưỡng 4 là cấu hình MB trước đây. Ngưỡng 1 tạo mã hoạt động dày; ngưỡng 3–5
quét độ thưa. Số synapse học là số phần tử ma trận trọng số goal, không phải
số trọng số khác 0 sau học. `input_synapses` được ghi riêng: 3.072 với các arm
giãn nở và MB, 512 với direct khớp ngân sách, 0 với direct thường (cue nối
thẳng bằng 768 synapse học). CX và steering dùng chung nên không tính vào
bảng này. Bảng cũng báo tỷ lệ kênh đầu vào hoạt động thực tế; mức hoạt động
không được suy ra chỉ từ ngưỡng.

Direct khớp ngân sách chỉ kiểm soát **số synapse học được cấp phát**. Những
kênh cue lặp không mang thông tin mới, nên nó không khớp năng lực biểu diễn
hay số phép toán với MB. Không diễn giải nhánh này như đối chứng công bằng về
năng lượng hoặc SOP.

Điều kiện nhiễu được khóa trong `fly_snn/comparison.py`. `missing_landmarks`
chỉ tăng dropout từ 0,35 lên 0,80 so với `standard`; `turn_corruption` chỉ
tăng nhiễu quay từ 0,10 lên 0,20. `combined` tăng nhiều loại nhiễu cùng lúc.
Các arm được huấn luyện trên cùng prototype sạch và chấm trên cue nhiễu mới
của chính context đã học.

Chạy từ `MB_ant` với thư mục output chưa tồn tại:

```powershell
python -m fly_snn.comparison --output outputs/four_arm_dev --first-seed 100 --n-seeds 20
```

`trials.csv` chứa từng bước; `summary_by_seed.csv` chứa metric cho mỗi seed,
điều kiện và arm; `summary_aggregate.csv` chứa trung bình/SD qua seed cùng
đếm synapse và tỷ lệ hoạt động. `steering_accuracy` trong CSV là **turn
accuracy** đã định nghĩa trong giao thức, bỏ trường hợp goal đối diện 180°.
`heading_mae_sectors` dùng chung CX nên trùng giữa các arm. Tie của goal và
heading được báo riêng. Kết quả là pilot phát triển, chưa phải test độc lập.

## Pilot validation seeds 200–204

Chạy 5 seed mới với output `outputs/fly_snn_four_arm_validation_200_204/`.
`validation_sparsity_curves.png` vẽ goal/turn accuracy theo tỷ lệ neuron
giãn nở hoạt động, trung bình ± SD giữa seed, ở điều kiện chuẩn và cue nhiễu
mạnh. Những seed này đã được xem kết quả nên không dùng làm test cuối.

Ở điều kiện chuẩn, ngưỡng 3/4/5 tạo hoạt động 36,9%/11,9%/2,0%, goal
accuracy 99,1%/99,3%/93,6%, turn accuracy 76,9%/90,1%/91,7%. Khi cue nhiễu
mạnh, goal accuracy là 91,6%/90,4%/69,1% và turn accuracy là
75,8%/86,9%/75,2%. Ngưỡng 5 tăng turn accuracy so với ngưỡng 4 ở điều kiện
chuẩn chỉ 1,5 điểm phần trăm và chỉ 2/5 seed có chênh lệch dương; nó giảm
turn accuracy ở cue nhiễu mạnh. Ngưỡng 4 là ứng viên cân bằng cho vòng phát
triển tiếp theo, chưa phải tham số được xác nhận trên test độc lập.
