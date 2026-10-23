# Thử thêm: cửa sổ nhớ dài và cấu hình belief

**Kết quả:** tăng memory lên 20/30 phút chưa cải thiện GRU10. Belief có decay tốt hơn prior yếu không decay, nhưng vẫn kém GRU10.

Branch `experiment/belief-gru-ablation`; nối tiếp [ablation 10 phút](../belief-gru-ablation/README.md).

## Các cấu hình chốt trước khi chạy

| Arm | Encoder | Nhớ arrivals | Prior Gamma (shape, rate) | Decay/bin |
|---|---|---:|---|---:|
| `gru10` | GRU | 10 phút | — | — |
| `gru20` | GRU | 20 phút | — | — |
| `gru30` | GRU | 30 phút | — | — |
| `belief20` | Belief | 20 phút | (1, 1) | 1 |
| `belief20_weak` | Belief | 20 phút | (0.1, 0.1) | 1 |
| `belief20_fade` | Belief | 20 phút | (0.1, 0.1) | 0.8 |
| `gru20_belief_fade` | GRU + belief | 20 phút | (0.1, 0.1) | 0.8 |

Prior yếu giữ mean ban đầu bằng 1 khách/bin nhưng giảm số pseudo-observations.
Decay 0.8 giảm trọng số các bin cũ: half-life khoảng 6,21 phút; trong 20 phút,
tổng trọng số tối đa 4,46 bin, tương đương khoảng 8,93 phút exposure. Đây là
power-likelihood posterior xấp xỉ cho local demand rate, không phải bộ lọc chính
xác của demand process trong simulator.

Với y_l là số khách ở lag l, m_l đánh dấu bin thực sự đã xảy ra, w_l = decay^l:

```text
alpha = prior_shape + sum(m_l * w_l * y_l)
beta  = prior_rate  + sum(m_l * w_l)
mean  = alpha / beta
std   = sqrt(alpha) / beta
```

Mean/std chia cho 40, mask entity padding; không xem padding đầu ca là zero-arrival
evidence. Với decay=1, đây là conjugate Gamma–Poisson posterior theo giả định rate
hằng cục bộ. Shape/rate/decay đi vào checkpoint kwargs và metadata.

## Lịch sử dài hơn được thu thập như thế nào?

Native observation vẫn giữ nguyên 5 bin arrivals. Khi yêu cầu 20/30 phút, Python
gom arrivals vừa quan sát mỗi bước vào buffer 10/15 bin, từ mới đến cũ, qua key
`arrival_memory`. Không truy cập future tapes hoặc đích đến ẩn. Buffer reset
riêng từng slot, giữ snapshot terminal trước auto-reset, và dùng chung logic cho
scalar env, native training batch và native evaluation pool. Pool identity và
checkpoint schema hash bao gồm độ dài memory; CLI tự khôi phục từ metadata.

GRU đọc oldest → newest, hidden 32 và khởi tạo 0 cho mỗi cửa sổ. Chỉ kéo dài
**lịch sử arrivals**, còn queue/load/headway vẫn là snapshot hiện tại. Không có
hidden state chạy xuyên toàn episode. Raw 1.383 input và actor/critic heads giữ
như trước; key dài chỉ đi vào GRU/posterior, không flatten thêm vào raw MLP.
Do đó GRU10/20/30 có **cùng số tham số**, tránh nhầm lợi ích memory với tăng mạng.
Các arm belief vẫn giữ parameter budget gần control bằng cách điều chỉnh MLP width.

## Protocol

- Config: [memory-belief.toml](../../../configs/experiments/memory-belief.toml).
- 7 arm × 3 seed (11/22/33) × 98.304 transitions = **2.064.384 transitions**.
- Cùng 500 train days, 100 validation days, masks/reward/PPO/runtime như vòng trước.
- Validation mỗi 12.288 bước, checkpoint best cost, hòa giữ mốc sớm.
- Train lại GRU10 để có control ngay trong nghiên cứu này; không dùng model 245k
  làm control cho các model 98k. Không đổi learning rate hay GRU hidden size.
- Probe mới chốt trước training: 50 ID seeds 71001–71050, 50 burst 72001–72050,
  50 traffic 73001–73050. Không trùng train/validation/test cũ hoặc probe vòng 1.
- Chỉ mở probe scores sau khi đủ 21 run; báo tất cả arm, không tuning sau probe.
- Tổng **3.150 probe episode**. Cost thấp hơn tốt hơn; báo thêm wait, P95 wait,
  worst-route mean wait, unfinished và abandoned shares.
- CI paired bootstrap theo ngày, mean qua seeds, 2.000 resamples seed 6001;
  std giữa seed báo riêng. CI chưa bao gồm uncertainty do model seed và chưa
  hiệu chỉnh multiple comparisons. Kết quả chỉ là ablation thăm dò ở 98k bước.

## Kết quả

Δ cost so với **GRU10 cùng ngân sách**; âm tốt hơn. Đây là probe mới, không so trực tiếp với các giá trị trên probe vòng trước.

| Arm | ID | Burst | Traffic |
|---|---:|---:|---:|
| `gru20` | +0.99% | +1.04% | +1.03% |
| `gru30` | -0.07% | +0.45% | -0.14% |
| `belief20` | +3.06% | +2.10% | +3.04% |
| `belief20_weak` | +3.91% | +4.16% | +4.18% |
| `belief20_fade` | +1.15% | +3.33% | +0.97% |
| `gru20_belief_fade` | +1.51% | +2.41% | +1.55% |

**Tăng memory:** GRU20 tăng mean cost khoảng 1% trên cả ba probe; ID/traffic CI nằm trên 0, burst CI cắt 0. GRU30 giảm rất nhẹ trên ID/traffic nhưng tăng trên burst; cả ba CI so với GRU10 đều cắt 0, nên chưa có bằng chứng cải thiện. GRU30 còn tăng worst-route mean wait khoảng 0,47 phút trên ID và 0,46 phút trên traffic.

| Contrast | ID Δ cost [95% CI] | Burst Δ cost [95% CI] | Traffic Δ cost [95% CI] |
|---|---:|---:|---:|
| `gru20 − gru10` | +131.2 [+68.1, +199.4] | +275.6 [-32.4, +564.9] | +138.5 [+79.9, +201.7] |
| `gru30 − gru10` | -9.1 [-54.1, +37.0] | +119.9 [-302.3, +559.5] | -19.1 [-56.9, +18.1] |

**Belief:** prior yếu (0.1,0.1) không giúp, và còn tệ hơn (1,1) trên cả ba probe. Thêm decay 0.8 vào prior yếu giảm cost **2,66% trên ID và 3,08% trên traffic** so với chính prior yếu không decay (3/3 seed tốt hơn); burst giảm 0,80% nhưng CI cắt 0. Dù vậy, `belief20_fade` vẫn tệ hơn GRU10 trên cả ba probe, với CI nằm trên 0. Đây là cấu hình belief đáng giữ để nghiên cứu tiếp, chưa phải cải thiện của controller.

Kết hợp `gru20_belief_fade` cũng tệ hơn GRU20 khoảng **0,51% / 1,35% / 0,52%** trên ID/burst/traffic. Mặc dù contrast ghép có 2/3 seed tốt hơn, seed xấu còn lại kéo mean lên; không thể kết luận chỉ từ số seed thắng.

Cost tuyệt đối, **mean ± std giữa model seeds** (std của mean cost từng seed, ddof=1):

| Arm | ID | Burst | Traffic |
|---|---:|---:|---:|
| `gru10` | 13,259.7 ± 90.4 | 26,383.9 ± 164.4 | 13,477.8 ± 40.9 |
| `gru20` | 13,390.9 ± 493.2 | 26,659.5 ± 506.7 | 13,616.4 ± 535.8 |
| `gru30` | 13,250.6 ± 180.2 | 26,503.8 ± 460.3 | 13,458.7 ± 127.4 |
| `belief20` | 13,665.8 ± 88.3 | 26,936.9 ± 335.7 | 13,887.7 ± 49.6 |
| `belief20_weak` | 13,777.7 ± 150.2 | 27,481.0 ± 886.1 | 14,041.5 ± 172.4 |
| `belief20_fade` | 13,411.6 ± 392.9 | 27,261.9 ± 1,555.2 | 13,608.3 ± 379.6 |
| `gru20_belief_fade` | 13,459.7 ± 321.2 | 27,018.6 ± 744.9 | 13,686.8 ± 278.1 |

Unfinished share = 0 ở tất cả arm/probe. P95 chờ so với GRU10 và thời gian chạy:

| Arm | Δ P95 ID (phút) | Δ P95 Burst | Δ P95 Traffic | Mean train + validation (giây/run) |
|---|---:|---:|---:|---:|
| `gru10` | +0.000 | +0.000 | +0.000 | 35.19 |
| `gru20` | +0.501 | +0.005 | +0.439 | 39.80 |
| `gru30` | +0.174 | -0.190 | +0.104 | 43.96 |
| `belief20` | +1.424 | +0.082 | +1.392 | 32.81 |
| `belief20_weak` | +1.537 | +0.294 | +1.599 | 33.01 |
| `belief20_fade` | +0.350 | +0.718 | +0.192 | 33.05 |
| `gru20_belief_fade` | +0.611 | +0.450 | +0.599 | 43.42 |

GRU20/30 tốn thêm khoảng **13.1% / 24.9%** thời gian train + validation so với GRU10. Tổng thời gian được ghi trong 21 run là **783.71 giây**, chưa gồm tạo probe và final probe evaluation; đây không phải benchmark tốc độ tách riêng.

**Quyết định:** giữ GRU10 làm lựa chọn gọn nhất trong nhóm temporal encoder đã thử; chưa có lý do chọn memory dài hơn. Giữ các config belief để tái lập kết quả, không bật belief mặc định. `algorithm.encoder="mlp"`, history 10 phút vẫn là mặc định của repo; chưa thay incumbent nghiên cứu cũ hoặc tăng ngân sách sau khi xem probe.

Audit đã xác nhận đủ **21 run, 2.064.384 transitions, 3.150 episode probe**, 96 rollout và 96 đợt PPO update/run; **28.224 finite checks** đều qua. Metadata/config/budget/checkpoint hashes và frozen source hashes khớp; bootstrap CI tính lại khớp CSV. Cả 3 đường validation GRU10 (8 mốc/seed) tái lập chính xác vòng trước.

Dữ liệu: [summary.csv](summary.csv), [contrasts.csv](contrasts.csv), runs.csv (historical artifact: Git `782f3e4`), probe_days.csv (historical artifact: Git `782f3e4`), [protocol.json](protocol.json). Cột `encoder` trong CSV là tên arm; `policy_encoder` trong `runs.csv` là loại encoder thực sự. `probe_days.csv` có đủ dữ liệu để tính lại các contrast.

## Tái lập và kiểm tra

```bash
.venv/bin/python scripts/run_belief_gru_ablation.py \
  --config configs/experiments/memory-belief.toml \
  --output runs/belief-gru/new-memory-run
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
```

Chọn memory qua `algorithm.history_minutes`, belief qua `belief_prior_shape`,
`belief_prior_rate`, `belief_decay`. `study.arms` là cấu hình của runner sweep;
CLI `train` thông thường sử dụng trực tiếp `[algorithm]`.

Kiểm tra: **68 passed, 53 skipped**, lint qua. Test mới kiểm tra FIFO theo slot
không liên tiếp, reset độc lập, snapshot không bị mutate, đủ 120 bước có auto-reset,
memory dài khớp native 5 bin, posterior có decay/exposure, reject config sai,
train/reload và cùng action/cost giữa scalar và native evaluator (gồm batch cuối
không đầy). CLI smoke đã nạp checkpoint 30 phút bằng config evaluation mặc định
và tự khôi phục memory từ metadata.

Không thay simulator Rust, action masks, rewards hoặc các báo cáo benchmark đã
đóng băng. Format trên source thay đổi qua; full-repo format vẫn còn dòng REPORT
có sẵn trong `scripts/build_native.py` không thuộc thử nghiệm này.

Raw runs/checkpoints/diagnostics và snapshot source: `runs/belief-gru/20261022-memory-r1/`.
