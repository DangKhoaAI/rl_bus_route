# Báo cáo thực nghiệm standard-v1

## Thiết lập thực nghiệm

Báo cáo tổng hợp standard-v1: so sánh ba cấu hình bộ mã hóa quan sát của MaskablePPO với ba bộ điều khiển Fixed, Proportional và Threshold trong cùng bộ mô phỏng xe buýt.

Kết quả chính: cả ba cấu hình RL có cost thấp hơn các baseline ở cả ba điều kiện. MLP10 có cost trung bình thấp nhất trong các cấu hình RL ở điều kiện bình thường và gián đoạn giao thông; Belief20-fade thấp nhất khi nhu cầu tăng đột biến. Chênh lệch giữa các cấu hình RL nhỏ và chỉ mang tính mô tả; báo cáo không khẳng định ưu thế thống kê giữa chúng.

| Hạng mục | Thiết lập chung |
| --- | --- |
| Mạng và đội xe | 3 tuyến hai chiều, 6 điểm dừng/tuyến; 12 xe, sức chứa 40 khách/xe |
| Thời gian mô phỏng | 240 phút/ngày; nhận khách trong 180 phút đầu; tick 30 giây; quyết định mỗi 120 giây |
| Dữ liệu | 500 ngày train, 100 ngày validation, 200 ngày cho mỗi điều kiện test |
| Ngân sách và seed | 245,760 transitions/seed; seeds 11, 22, 33 |
| PPO | learning rate 0.0003; gamma 1.0; GAE lambda 0.95; clip 0.2 |
| Rollout và tối ưu | 4 envs x 256 bước; batch 256; 4 epochs; entropy 0.01 |
| Runtime | Rust, CPU; 2 Torch threads; đánh giá theo batch 16 |
| Điều khiển | Giữ nguyên action masking, REASSIGN và SHORT_TURN; forecast và reward shaping tắt |
| Chọn checkpoint | Cost validation thấp nhất sau khi train; hòa thì chọn sớm nhất; đánh giá mỗi 12,288 transitions. Step 0 chỉ làm mốc tham chiếu. |

### Ba cấu hình

| Cấu hình | Encoder / cửa sổ | Điểm khác biệt |
| --- | --- | --- |
| MLP10 | MLP / 10 phút | Mạng feed-forward |
| GRU10 | GRU / 10 phút | Mã hóa cửa sổ quan sát bằng GRU |
| Belief20-fade | Belief / 20 phút | Đặc trưng rate Gamma-Poisson; prior shape/rate = 0.1/0.1; decay = 0.8 |

Belief20-fade là xấp xỉ cục bộ cường độ nhu cầu, không phải belief state đầy đủ của POMDP. Đây là snapshot tái lập trên bộ dữ liệu tham chiếu đã được dùng trong nghiên cứu trước; không phải một test set xác nhận hoàn toàn mới.

## Kết quả tổng hợp

Cost càng thấp càng tốt. Mỗi giá trị RL là trung bình trên các seed model và ngày kịch bản. Cột +/- là độ lệch chuẩn cost trung bình giữa ba seed train; baseline tham chiếu là baseline có cost thấp nhất trong từng điều kiện.

| Cấu hình | Điều kiện | Cost +/- seed SD | Baseline | Giảm cost | CI 95% (%) |
| --- | --- | --- | --- | --- | --- |
| MLP10 | Nhu cầu bình thường | 13,065.7 +/- 52.4 | Fixed | 17.6% | [17.5; 17.8] |
| GRU10 | Nhu cầu bình thường | 13,132.3 +/- 102.1 | Fixed | 17.2% | [17.0; 17.4] |
| Belief20-fade | Nhu cầu bình thường | 13,081.6 +/- 50.8 | Fixed | 17.5% | [17.4; 17.7] |
| MLP10 | Nhu cầu tăng đột biến | 25,197.2 +/- 67.2 | Threshold | 5.0% | [2.8; 7.3] |
| GRU10 | Nhu cầu tăng đột biến | 25,218.0 +/- 51.1 | Threshold | 5.0% | [2.7; 7.2] |
| Belief20-fade | Nhu cầu tăng đột biến | 25,178.5 +/- 20.8 | Threshold | 5.1% | [2.9; 7.4] |
| MLP10 | Gián đoạn giao thông | 13,412.4 +/- 65.3 | Fixed | 17.8% | [17.6; 17.9] |
| GRU10 | Gián đoạn giao thông | 13,460.0 +/- 89.2 | Fixed | 17.5% | [17.3; 17.6] |
| Belief20-fade | Gián đoạn giao thông | 13,419.3 +/- 61.5 | Fixed | 17.7% | [17.5; 17.9] |

Các khoảng tin cậy trên so sánh RL với baseline, không so sánh trực tiếp MLP10, GRU10 và Belief20-fade. Khoảng được bootstrap theo ngày sau khi lấy trung bình seed; không bao gồm bất định do seed train.

## Nhu cầu bình thường

test_id: Nhu cầu và giao thông theo cấu hình nền của bộ mô phỏng, với các seed kịch bản riêng cho test. Mỗi controller được chấm trên 200 ngày kịch bản giống nhau; RL có 3 seed train.

| Controller | Cost | Chờ TB (phút) | P95 chờ (phút) | Hoàn thành (%) | Bỏ cuộc (%) | Chưa xong (%) |
| --- | --- | --- | --- | --- | --- | --- |
| MLP10 | 13,065.7 | 5.57 | 11.28 | 100.00 | 0.00 | 0.00 |
| GRU10 | 13,132.3 | 5.63 | 11.69 | 100.00 | 0.00 | 0.00 |
| Belief20-fade | 13,081.6 | 5.57 | 11.28 | 100.00 | 0.00 | 0.00 |
| Fixed | 15,861.3 | 7.42 | 14.88 | 100.00 | 0.00 | 0.00 |
| Proportional | 16,056.6 | 7.07 | 14.73 | 99.64 | 0.00 | 0.36 |
| Threshold | 16,652.3 | 7.27 | 15.46 | 99.58 | 0.00 | 0.42 |

### Giảm cost so với từng baseline

| Cấu hình | Fixed | Proportional | Threshold |
| --- | --- | --- | --- |
| MLP10 | 17.6% [17.5; 17.8] | 18.6% [18.0; 19.2] | 21.5% [20.9; 22.2] |
| GRU10 | 17.2% [17.0; 17.4] | 18.2% [17.6; 18.8] | 21.1% [20.5; 21.8] |
| Belief20-fade | 17.5% [17.4; 17.7] | 18.5% [17.9; 19.1] | 21.4% [20.8; 22.1] |

Ô trong bảng: % giảm cost [CI 95%]. Giá trị dương có lợi cho RL.

Cả ba cấu hình RL giảm cost và P95 chờ so với các baseline; tỷ lệ hoàn thành là 100% trên điều kiện này.

## Nhu cầu tăng đột biến

test_ood_burst: Tại một điểm dừng ngẫu nhiên, thêm 8-20 hành khách mỗi tick trong 5-10 phút; thời điểm bắt đầu nằm trong phút 30-150. Tick mô phỏng là 30 giây. Mỗi controller được chấm trên 200 ngày kịch bản giống nhau; RL có 3 seed train.

| Controller | Cost | Chờ TB (phút) | P95 chờ (phút) | Hoàn thành (%) | Bỏ cuộc (%) | Chưa xong (%) |
| --- | --- | --- | --- | --- | --- | --- |
| MLP10 | 25,197.2 | 8.07 | 30.88 | 98.18 | 1.82 | 0.00 |
| GRU10 | 25,218.0 | 8.12 | 30.77 | 98.21 | 1.79 | 0.00 |
| Belief20-fade | 25,178.5 | 8.07 | 30.87 | 98.21 | 1.79 | 0.00 |
| Fixed | 33,747.4 | 10.49 | 39.13 | 95.35 | 4.65 | 0.00 |
| Proportional | 27,980.7 | 9.08 | 28.51 | 97.56 | 1.82 | 0.62 |
| Threshold | 26,531.9 | 9.11 | 26.64 | 98.59 | 0.90 | 0.52 |

### Giảm cost so với từng baseline

| Cấu hình | Fixed | Proportional | Threshold |
| --- | --- | --- | --- |
| MLP10 | 25.3% [24.7; 26.1] | 9.9% [7.7; 12.2] | 5.0% [2.8; 7.3] |
| GRU10 | 25.3% [24.6; 26.0] | 9.9% [7.7; 12.1] | 5.0% [2.7; 7.2] |
| Belief20-fade | 25.4% [24.7; 26.1] | 10.0% [7.8; 12.2] | 5.1% [2.9; 7.4] |

Ô trong bảng: % giảm cost [CI 95%]. Giá trị dương có lợi cho RL.

Đánh đổi dịch vụ: cả ba cấu hình RL có P95 chờ cao hơn Proportional và Threshold, dù cost tổng thấp hơn. Threshold hoàn thành 98,59% khách và có 0,90% bỏ cuộc, tốt hơn ba cấu hình RL theo hai chỉ số này.

## Gián đoạn giao thông

test_ood_traffic: Thời gian di chuyển trên hành lang của một tuyến ngẫu nhiên tăng 1,5-2,0 lần trong 30 phút. Mỗi controller được chấm trên 200 ngày kịch bản giống nhau; RL có 3 seed train.

| Controller | Cost | Chờ TB (phút) | P95 chờ (phút) | Hoàn thành (%) | Bỏ cuộc (%) | Chưa xong (%) |
| --- | --- | --- | --- | --- | --- | --- |
| MLP10 | 13,412.4 | 5.70 | 11.79 | 100.00 | 0.00 | 0.00 |
| GRU10 | 13,460.0 | 5.75 | 12.11 | 100.00 | 0.00 | 0.00 |
| Belief20-fade | 13,419.3 | 5.70 | 11.79 | 100.00 | 0.00 | 0.00 |
| Fixed | 16,307.8 | 7.57 | 15.20 | 100.00 | 0.00 | 0.00 |
| Proportional | 16,549.8 | 7.18 | 15.04 | 99.55 | 0.00 | 0.45 |
| Threshold | 16,924.6 | 7.34 | 15.82 | 99.63 | 0.00 | 0.37 |

### Giảm cost so với từng baseline

| Cấu hình | Fixed | Proportional | Threshold |
| --- | --- | --- | --- |
| MLP10 | 17.8% [17.6; 17.9] | 19.0% [18.3; 19.6] | 20.8% [20.2; 21.4] |
| GRU10 | 17.5% [17.3; 17.6] | 18.7% [18.0; 19.3] | 20.5% [19.9; 21.1] |
| Belief20-fade | 17.7% [17.5; 17.9] | 18.9% [18.3; 19.6] | 20.7% [20.1; 21.3] |

Ô trong bảng: % giảm cost [CI 95%]. Giá trị dương có lợi cho RL.

Cả ba cấu hình RL giảm cost và P95 chờ so với các baseline; tỷ lệ hoàn thành là 100% trên điều kiện này.

## Huấn luyện, cách tính và giới hạn

| Cấu hình | Seed | Val ban đầu | Val tốt nhất | Bước được chọn | Tổng bước |
| --- | --- | --- | --- | --- | --- |
| MLP10 | 11 | 16,320.4 | 13,199.1 | 159,744 | 245,760 |
| MLP10 | 22 | 14,598.7 | 13,114.0 | 98,304 | 245,760 |
| MLP10 | 33 | 16,485.2 | 13,213.5 | 233,472 | 245,760 |
| GRU10 | 11 | 18,075.8 | 13,189.8 | 172,032 | 245,760 |
| GRU10 | 22 | 16,574.6 | 13,157.2 | 147,456 | 245,760 |
| GRU10 | 33 | 15,482.5 | 13,311.8 | 208,896 | 245,760 |
| Belief20-fade | 11 | 15,084.0 | 13,218.6 | 208,896 | 245,760 |
| Belief20-fade | 22 | 16,155.9 | 13,125.6 | 86,016 | 245,760 |
| Belief20-fade | 33 | 15,310.6 | 13,215.4 | 208,896 | 245,760 |

### Cách tính

Cost là mục tiêu tổng hợp theo trọng số, đơn vị tương đương passenger-minute, không phải chi phí tiền tệ. Cost gồm thời gian chờ, thời gian trên xe, quá tải tiện nghi, vận hành/deadhead, từ chối lên xe lần đầu, bỏ cuộc, thay đổi nhiệm vụ và khách chưa xong cuối ca.

Đầu tiên lấy trung bình các seed model trên từng ngày kịch bản, sau đó lấy trung bình các ngày. P95 chờ trong bảng là trung bình P95 của từng ngày, không phải percentile gộp toàn bộ hành khách. Giảm cost = 100 x (baseline - RL) / baseline. CI 95% dùng 2.000 lần paired bootstrap theo ngày, seed 6001; giữ ghép cặp theo seed và hash kịch bản. Không hiệu chỉnh nhiều phép so sánh và không đưa biến thiên seed train vào CI.

### Kết luận và phạm vi

Trên snapshot này, ba cấu hình RL đều cải thiện cost so với baseline. GRU10 và Belief20-fade chưa cho thấy cải thiện nhất quán về cost trung bình so với MLP10. Khi nhu cầu tăng đột biến, cost thấp hơn không đồng nghĩa với mọi chỉ số dịch vụ tốt hơn.

Kết quả chỉ áp dụng cho bộ mô phỏng tổng hợp trên mạng cố định ba tuyến. Không suy rộng thành hiệu quả triển khai thực tế hay tính tối ưu toàn cục. Test dataset đã được dùng trước đây, và số seed train là ba. Báo cáo này không thay thế các kết luận lịch sử đã chốt trong thư mục reports/ của dự án.

### Nguồn và tái lập

Tất cả số trong báo cáo được đối chiếu lại với kết quả từng ngày. Dữ liệu gốc nằm trong evidence/: configs/ và protocol.json lưu thiết lập; data/manifest.json lưu định danh kịch bản; <policy>/<seed>/train/ lưu metadata và lịch sử; <policy>/<seed>/eval/ và baselines/shared/eval/ lưu kết quả. summary.csv và comparisons.csv lưu bảng tổng hợp. artifacts.json kiểm tra inventory, kích thước và SHA256 toàn bộ bundle.

Kiểm tra: uv run bus-rl artifact verify --path runs/standard-v1. Weights, optimizer state và log chẩn đoán vẫn ở experiments/; bundle không chứa chúng.
