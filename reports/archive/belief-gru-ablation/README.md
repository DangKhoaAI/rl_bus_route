# Ablation nhỏ: demand belief và GRU

**Kết quả:** GRU giảm mean cost 1,33% / 4,87% / 1,15% trên ID/burst/traffic so với MLP cùng ngân sách; belief riêng chưa hữu ích.

Branch: `experiment/belief-gru-ablation`. Đây là thử nghiệm encoder với ngân sách
nhỏ; không thay kết luận hay incumbent trong báo cáo cuối của nghiên cứu cũ.

## Phương pháp

Thiết kế 2×2, cùng MaskablePPO và observation gốc:

| Arm | Phần bổ sung vào observation đã flatten |
|---|---|
| `mlp` | Không; giữ nguyên encoder gốc |
| `belief` | Mean và standard deviation của posterior tốc độ khách đến |
| `gru` | GRU 32 hidden units trên 5 bin arrivals, cũ → mới |
| `belief_gru` | Cả hai phần trên |

**Belief:** mỗi trạm/hướng có tốc độ khách đến λ, đơn vị khách/control interval.
Giả sử λ không đổi trong cửa sổ tối đa 5 bin, số khách mỗi bin có phân phối
Poisson(λ), và prior λ ~ Gamma(shape=1, rate=1). Với tổng khách quan sát N và số
bin thực sự đã xảy ra n:

```text
λ | history ~ Gamma(1 + N, 1 + n)
mean = (1 + N) / (1 + n)
std  = sqrt(1 + N) / (1 + n)
```

Hai đặc trưng chia cho 40 để cùng thang với arrival counts. Bỏ qua các bin
padding đầu ca; đặc trưng ở các trạm padding được đặt bằng 0. n được tính từ
thời gian quan sát và horizon/control interval của config, không đọc future tape.
Posterior được tính lại trên cửa sổ trượt, không giữ state giữa các lần forward.
Đây là **belief xấp xỉ về local demand rate**, không phải posterior trên toàn bộ
trạng thái POMDP. Giả định Poisson độc lập và rate hằng cục bộ có thể sai khi burst.
Mean và std đều suy ra từ observation đang có; thử nghiệm kiểm tra lợi ích của
cách biểu diễn có cấu trúc, không bổ sung thông tin mà MLP không được phép thấy.

**GRU:** đọc vector 64 stop/direction counts ở mỗi bin, theo thứ tự cũ đến mới;
lấy hidden cuối 32 chiều. Hidden bắt đầu từ 0 ở mỗi forward. Đây là recurrent
encoder của cửa sổ 10 phút có sẵn, không phải recurrent policy nhớ xuyên episode.
Gradient đi qua cả 5 bước GRU trong PPO minibatch; không cần sequence rollout
buffer hay custom masked recurrent PPO. Không thêm LSTM thứ hai để giữ phạm vi nhỏ.

Giữ toàn bộ input gốc, actor/critic heads 128, action masks, reward, dữ liệu và
PPO hyperparameters. Giảm nhẹ chiều rộng MLP đầu ở các arm bổ sung để tổng số
tham số gần bằng nhau. Vì chiều rộng có thay đổi, contrast đánh giá encoder ở
cùng parameter budget, không cô lập thuần một phép thêm feature trên cùng MLP.

## Protocol

- 500 ngày train, 100 ngày validation từ `data/generated/base/manifest.json`.
- 98.304 transitions/arm/seed, seeds 11/22/33: tổng 1.179.648 transitions.
- Mỗi arm có cùng 40% ngân sách của core cũ (245.760); MLP control được train
  lại đúng ngân sách này, không so candidate 98k với incumbent 245k.
- Validation mỗi 12.288 bước; lưu checkpoint có cost nhỏ nhất, hòa giữ mốc sớm.
- Hoàn thành toàn bộ training trước khi đọc probe scores; không tuning sau probe.
- Probe mới: 50 ngày ID (seeds 61001–61050), 50 burst (62001–62050), 50 traffic
  (63001–63050), không trùng seed các split cũ. Tổng 1.800 episode probe.
- Metric chính: raw `total_cost_core`, thấp hơn tốt hơn. Báo thêm chờ trung bình,
  P95 chờ, tuyến chờ trung bình tệ nhất, unfinished và abandoned shares.
- CI 95%: paired day bootstrap sau khi lấy mean qua model seeds, 2.000 resamples,
  seed 6001. Seed std báo riêng; CI này không thể hiện uncertainty do model seed
  và chưa hiệu chỉnh nhiều phép so sánh. Đây là ablation thăm dò, không phải
  bằng chứng xác nhận thay incumbent.

Protocol, hash source/config/data, snapshot source và checkpoints nằm trong
`runs/belief-gru/20261022-r1/`. Các CSV nhỏ được sao chép vào thư mục báo cáo này;
raw per-day metrics và diagnostics giữ trong thư mục run được gitignore.

## Kết quả

Đổi cost so với MLP cùng ngân sách; **âm là tốt hơn**. Mean lấy qua 3 model seeds và 50 ngày/group.

| Encoder | ID | Burst | Traffic |
|---|---:|---:|---:|
| `belief` | +3.27% | -0.91% | +2.84% |
| `gru` | -1.33% | -4.87% | -1.15% |
| `belief_gru` | -1.18% | -4.62% | -1.10% |

Cost tuyệt đối: **mean ± std giữa model seeds** (std của mean cost từng seed, ddof=1).

| Encoder | ID | Burst | Traffic |
|---|---:|---:|---:|
| `mlp` | 13,441.5 ± 449.0 | 26,219.9 ± 1,383.8 | 13,826.3 ± 414.6 |
| `belief` | 13,880.6 ± 208.5 | 25,980.3 ± 424.4 | 14,219.6 ± 228.8 |
| `gru` | 13,262.3 ± 44.5 | 24,943.4 ± 121.5 | 13,667.9 ± 115.1 |
| `belief_gru` | 13,282.3 ± 101.4 | 25,009.3 ± 143.9 | 13,674.8 ± 113.1 |

GRU so với MLP:

| Probe | Δ cost [CI 95% theo ngày] | Seed tốt hơn | Δ P95 chờ (phút) | Δ abandoned share |
|---|---:|---:|---:|---:|
| ID | -179.2 [-224.5, -132.9] | 2/3 | -0.314 | +0.00000 |
| Burst | -1276.5 [-1670.5, -861.3] | 2/3 | -1.543 | -0.00549 |
| Traffic | -158.4 [-210.9, -105.4] | 2/3 | -0.149 | +0.00000 |

Unfinished share bằng 0 ở tất cả arm/probe. Với GRU, mean wait và worst-route mean wait cũng giảm trên cả ba probe. Belief riêng tăng P95 chờ khoảng 1,68 phút trên ID và 1,62 phút trên traffic; burst cost CI của belief còn cắt 0.

Thêm belief vào GRU làm mean cost tăng nhẹ **+0,15% / +0,26% / +0,05%** trên ID/burst/traffic; cả ba CI của contrast `belief_gru − gru` đều cắt 0. Chưa có bằng chứng lợi ích cộng thêm từ posterior này.

| Encoder | Tổng tham số | Mean train + validation (giây/run) | So với MLP |
|---|---:|---:|---:|
| `mlp` | 448,862 | 28.76 | +0.0% |
| `belief` | 448,830 | 31.68 | +10.2% |
| `gru` | 449,478 | 36.86 | +28.2% |
| `belief_gru` | 449,070 | 40.04 | +39.2% |

Tổng thời gian được ghi trong 12 run: **412,01 giây**, chưa gồm tạo probe và final probe evaluation. Wall time là quan sát trên một máy, có validation, không phải benchmark tốc độ riêng.

**Kết luận thăm dò:** GRU cửa sổ là hướng có tín hiệu tốt nhất để thử tiếp. Belief Gamma–Poisson đơn giản này chưa hữu ích trong contrast đang xét; không thêm độ phức tạp chỉ để giữ nó. MLP có seed spread lớn ở ngân sách ngắn, đặc biệt burst, và GRU chỉ thắng 2/3 seed: không suy ra lợi thế chắc chắn ở full budget. Giữ `algorithm.encoder="mlp"` mặc định; không thay incumbent, không tự tăng ngân sách hay tuning sau probe.

Dữ liệu: [summary.csv](summary.csv), [contrasts.csv](contrasts.csv), [runs.csv](runs.csv), [probe_days.csv](probe_days.csv), [protocol.json](protocol.json). `probe_days.csv` chứa đủ per-day metrics để tính lại các contrast và bootstrap CI.

Audit: đủ 12 run, 96 rollout và 96 PPO update/run, tổng 1.179.648 mask decisions; **16.128 finite checks** trong diagnostics đều qua. Metadata config/encoder/budget, checkpoint hashes, frozen source hashes và CI tính lại đều khớp. Probe có đủ 36 nhóm encoder×seed×split, mỗi nhóm 50 ngày, không trùng khóa.

## Tái lập và kiểm tra

```bash
.venv/bin/python scripts/run_belief_gru_ablation.py \
  --output runs/belief-gru/new-run
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
```

Chọn riêng encoder qua `algorithm.encoder = "mlp" | "belief" | "gru" |
"belief_gru"` trong config TOML. Loader dùng feature-extractor kwargs đã lưu
trong checkpoint; evaluation không cần đoán encoder từ config.

Kiểm tra mới bao phủ posterior có padding đầu ca, entity padding, thứ tự thời gian
GRU, gradient, độc lập thứ tự batch, cả bốn arm train trên native batches qua
auto-reset, lưu/nạp best checkpoint, tái lập cost callback và toàn bộ chuỗi hành
động giữa scalar/native-batch evaluator. Full pytest: **56 passed, 53 skipped**
(deprecated Python backend và các test không áp dụng được skip theo repo).

Ruff lint toàn repo qua. Ruff format trên các file thay đổi qua; full-repo format
check còn báo dòng `REPORT` quá dài có sẵn ở `scripts/build_native.py` (file không
thay đổi trong thử nghiệm này).

Checkpoint core cũ cũng nạp được và chạy đủ một episode Rust 120 bước, mọi hành
động đều hợp lệ. Với seed 11, toàn bộ 8 mốc validation của MLP control khớp chính
xác các mốc tương ứng trong 98.304 bước đầu của run core đã lưu trước đó.
