# Cải thiện RL sau khi tăng tốc bằng Rust

Ngày: **2026-09-12**. Trạng thái: **đề xuất thực nghiệm; chưa triển khai theo tài liệu này**.
Implementation plan (English): [improve_RL.md](../plan/improve_RL.md).


## 1. Quyết định và mục tiêu

**Làm Rust trước để simulation/train nhanh, rồi mới thử các hướng cải thiện RL.** Rust là hạ tầng rút ngắn vòng lặp nghiên cứu, không phải thay đổi thuật toán học.

Thứ tự chung: **R0 khóa oracle → R1 kernel → R2 observation/mask → R3 tích hợp → R4 parity/speed acceptance → L0 baseline RL → L1 tuning → L2 mở rộng có điều kiện → L3 báo cáo**.

[rust_improve.md](rust_improve.md) là nguồn chuẩn của R0–R4, đặc biệt benchmark/gates §7. Tài liệu này là nguồn chuẩn của L0–L3. [spec_v1.0.md](spec_v1.0.md) và [plan_v1.0.md](../plan/plan_v1.0.md) vẫn xác định physics, metrics và nghiên cứu core/ablation. Không suy luận rằng T9–T11 hay full study đã xong chỉ từ code/spec tồn tại.

Ba câu hỏi phải báo cáo riêng:

1. **Throughput:** Rust tạo transitions và chạy full training nhanh hơn bao nhiêu?
2. **Sample efficiency:** trên cùng số transitions, policy nào đạt cost thấp hơn?
3. **Wall-clock efficiency:** trong cùng thời gian, phương án nào đạt chất lượng tốt hơn?

Không gọi cùng policy chạy nhanh hơn là policy thông minh hơn. Không gọi train nhiều transitions hơn là thắng trên cùng sample budget.

## 2. Điểm xuất phát có bằng chứng

- `reports/training-diagnosis-after.md`: learn 2048 transitions 7.69 s, khoảng 266 decisions/s; learn có cProfile. Khoảng 15 phút/full seed là ngoại suy, không là runtime production đã xác nhận.
- `reports/pilot.md`: PPO 12,288 transitions có mean core cost khoảng 18,207 so fixed 16,017 trên 100 validation days; wait thấp hơn nhưng unfinished cao hơn. Chưa phải kết quả full training và chưa chứng minh PPO thắng heuristics.
- Mean cost 15,471.25 trong trial diagnose chỉ là regression mốc cũ, không phải mục tiêu chất lượng RL. Không so trực tiếp với pilot vì khác checkpoint/budget/bộ ngày.
- `reports/forecast.md`: đã có diagnostic forecast, không đồng nghĩa đã chứng minh forecast cải thiện policy.

Trước L0, inventory artifacts full runs hiện có; chỉ đánh dấu hoàn thành việc có checkpoint/config/metadata/results tương ứng. Giữ các artifacts lịch sử; không ghi đè để tạo cảm giác cùng backend/revision.

## 3. Điều kiện vào L0: hoàn tất Rust R4

Hoàn tất các gates trong [Rust §7](rust_improve.md#7-acceptance-và-benchmark-protocol-nguồn-chuẩn-dùng-chung): parity physics/obs/mask/metrics, integration/checkpoint/forecast, benchmark không profiler, median simulation và learn **≥2×** Python after, full workflow thực đo nhanh hơn. **≥4× chỉ là stretch goal**.

R0 bao gồm đóng băng Python oracle và đo baseline không profiler; không yêu cầu tối ưu Python thêm trước Rust. Tối ưu observation bằng statistics/ring buffers và cache mask thực hiện trong Rust, bảo toàn arrival history và completed channel. Không được bỏ finished khỏi observation mà không có thống kê thay thế tương đương.

R4 chưa đạt thì tiếp tục sửa/tối ưu backend, chưa chạy sweep thuật toán. Smoke/full-seed kiểm định migration được phép trong R4 và không được trình bày như thử nghiệm cải thiện RL.

Sau acceptance, đóng băng Rust revision, manifests và dependency versions cho mỗi loạt nghiên cứu. Python là reference/fallback; mọi kết quả loạt mới phải ghi backend, build, hashes và compute thực tế.

## 4. L0 — Thiết lập baseline RL đáng tin

### 4.1 Chạy core trước, không đổi thuật toán

Dùng `configs/experiments/core.toml`, seeds **11, 22, 33**, mỗi seed **245,760 transitions**. Giữ MaskablePPO, feature extractor hiện tại, 221 actions, reward/physics/guards và tensors như oracle.

Cấu hình đối chứng hiện tại: CPU, n_envs=4, n_steps=256, batch_size=256, n_epochs=4, learning_rate=3e-4, ent_coef=0.01, gamma=1.0, gae_lambda=0.95, clip_range=0.2. Eval mỗi 12,288 transitions; chọn best checkpoint theo validation cost, tie chọn checkpoint sớm nhất như callback hiện tại.

Chốt validation ngày và limit dùng chung trước chạy; full comparison dùng 100 validation days theo split hiện có. Không dùng 10 ngày diagnose thay final validation. Seed benchmark R4 chỉ tái dùng nếu toàn bộ config, validation schedule, manifest và provenance đúng protocol L0; nếu không, chạy riêng.

Heuristics fixed/threshold/proportional được đánh giá cùng scenario tapes và metric. Tune threshold/proportional trên validation, ghi search budget, rồi freeze trước test. Không cấp future demand hay hidden state cho controller.

### 4.2 Chẩn đoán trước khi chọn thay đổi

Thu thập trên validation và train, chưa dùng held-out test để chọn hướng:

| Bằng chứng cần thu | Câu hỏi trả lời |
|---|---|
| Validation core cost theo transitions và wall clock, best/last checkpoint | Còn học, plateau hay tụt sau một mốc? |
| PPO entropy, approximate KL, clip fraction, value loss, explained variance | Exploration còn đủ? Update quá mạnh? Critic giải thích return được không? |
| Raw cost components, unfinished/abandoned, per-route wait | Cost cao do chờ, bỏ khách, deadhead, fairness hay cuối episode? |
| Action frequency và valid opportunity theo action family | Policy ít dùng vì mask hiếm cho phép hay vì không chọn? |
| Distribution obs/reward/returns, finite checks, trường hợp cực trị | Scaling hoặc outlier làm training khó? |
| Episode traces cho failure cases được chọn theo tiêu chí trước | Hành động nào gây hậu quả và reward trễ bao lâu? |

Các telemetry chưa có trong artifacts phải được bổ sung khi triển khai; không coi bảng này là số đo hiện có. Action family frequency phải kèm số cơ hội hợp lệ, không mặc định NOOP nhiều là policy lỗi.

**Gate L0:** đủ 3 core seeds, baseline validation, learning curves/diagnostics và manifest. Chấp nhận kết quả không thắng heuristics; không thay đổi mục tiêu để làm đẹp score.

## 5. L1 — Cải thiện cách train PPO, giữ nguyên bài toán

Ưu tiên theo bằng chứng L0. Mỗi vòng thay một nhóm yếu tố, luôn có core đối chứng. Các giá trị dưới đây là ứng viên đề xuất, không phải cấu hình đã tốt hơn.

| Ưu tiên | Hướng | Thiết kế nhỏ ban đầu | Bằng chứng để giữ |
|---|---|---|---|
| 1 | Train đủ lâu nếu curve còn giảm | Core budget B=245,760; thử 2B rồi 4B nếu validation còn cải thiện | Improvement theo curve; báo rõ tăng sample/compute, không gộp với fixed-B thắng |
| 2 | Learning rate và độ mạnh PPO update | LR {1e-4, 3e-4}; sau đó mới thử n_epochs {4, 8} hoặc clip_range {0.1, 0.2} tùy KL/clip fraction | Cost đa seed giảm và update ổn định |
| 3 | Exploration | ent_coef {0.003, 0.01, 0.03}; chỉ thử schedule sau fixed coefficients | Không collapse sớm; cost/unfinished tốt hơn, không chỉ entropy cao |
| 4 | Rollout và minibatch | n_steps {128, 256, 512}, giữ n_envs=4; batch_size chia hết rollout size | So ở cùng transitions, ghi số update/thời gian và advantage/value diagnostics |
| 5 | Value learning/scaling | Kiểm tra return scale trước; thử vf_coef hoặc normalization riêng khi có dấu hiệu | Critic và validation cải thiện, không chỉ value loss đổi thang đo |

Không chạy Cartesian grid toàn bảng. Screening tối đa **6 candidates/vòng**, seed 11, cùng B và cùng validation, dùng để loại cấu hình hỏng. Chọn tối đa 2 candidates để xác nhận thêm seeds 22,33 trước kết luận. Báo đầy đủ candidate/seed thất bại và tổng tuning compute. Không lấy seed 11 screening làm kết quả cuối cùng.

`gamma=1.0` giữ nguyên trong L1 vì mục tiêu hiện là tổng cost hữu hạn theo episode. Đổi gamma sẽ thay trọng số thời gian của mục tiêu, phải đưa sang L2. Reward normalization nếu thử chỉ fit/update trên train, freeze stats khi eval, lưu stats cùng checkpoint và báo raw core cost; không normalize metric báo cáo hoặc mask/valid flags như dữ liệu liên tục.

**Gate L1:** candidate được xác nhận đa seed, có control đúng B, configs/hashes và validation evidence; nếu không tốt hơn thì giữ core. Không bắt buộc tìm được cải thiện.

## 6. L2 — Mở rộng có điều kiện, tách từng giả thuyết

Chỉ chọn hướng có bằng chứng từ L0/L1; không triển khai tất cả cùng lúc.

### 6.1 Forecast causal

Ưu tiên kiểm tra module có sẵn: `configs/experiments/forecast.toml` so no-forecast core với cùng seeds/budget và backend. Fit historical forecaster trên train arrival logs rồi freeze. Giữ tensor shape và enabled flag; kiểm thử hai tapes giống quá khứ nhưng khác tương lai cho cùng prediction tại t.

MAE tốt không bảo đảm điều khiển tốt. Báo đồng thời forecast error và control cost/unfinished. Forecast không dùng scenario seed, future tape hoặc thông tin mà baseline không được phép truy cập.

### 6.2 Encoder hoặc bộ nhớ cho partial observability

Nếu flat MLP hiện tại khó biểu diễn cấu trúc route/stop/vehicle, thử encoder chia theo nhóm rồi pooling/fusion, giữ obs/action contract ở vòng đầu. Nếu history hiện tại thiếu thông tin dự báo, mới xem xét temporal encoder/recurrent policy.

Đây là thay đổi kiến trúc, cần implementation/compatibility spike riêng với MaskablePPO, masks, recurrent state/reset nếu có; không giả định bật recurrent là sẵn chạy. So cùng transitions, báo parameter count và inference/training wall. Đổi schema thì version/hash mới và không load checkpoint cũ như tương thích.

### 6.3 Reward hoặc credit assignment

Chỉ sau khi xác định cost cao đến từ component nào. Thử từng giả thuyết: scale penalty terminal unfinished, fairness trade-off hoặc shaping có giải thích. Giữ physics, demand, masks và guards; ghi rõ đây là thay đổi learning objective.

Luôn tính lại **total_cost_core với weights gốc** từ raw components để so chính. Không gọi reward lớn hơn là tốt hơn khi đổi weights. Báo trade-off completion, abandonment, waiting/P95 và chi phí vận hành. Nếu đổi gamma/shaping, ghi mục tiêu mới và không tuyên bố policy invariance khi chưa chứng minh điều kiện áp dụng.

### 6.4 Curriculum hoặc warm-start từ heuristic

Chỉ thử khi exploration/cold-start là vấn đề. Curriculum chỉ dùng train split, công bố lịch phân phối demand và bảo đảm giai đoạn cuối có train distribution mục tiêu. Warm-start chỉ dùng demonstrations từ train, rồi fine-tune bằng RL.

Báo riêng transitions/demonstrations và compute phụ; so PPO-from-scratch công bằng. Không dùng test/OOD held-out để tạo demonstration hoặc curriculum thích nghi theo test score.

## 7. Protocol so sánh và L3 acceptance

### 7.1 Hai loại ngân sách

- **Fixed transitions (chính):** mỗi candidate và control cùng B, ngày validation, seeds, checkpoint selection và backend revision. Tăng B là thí nghiệm budget riêng; inference/model compute khác phải báo.
- **Fixed wall clock (phụ):** chốt ngân sách trước, cùng phần cứng và quy tắc tính thời gian (gồm train + periodic validation + checkpoint, setup báo riêng); ghi transitions thực tế. Score test ngoài ngân sách training và thời gian đó báo riêng.

Đồ thị cost theo cả transitions và wall clock giúp tách hiệu quả thuật toán khỏi lợi ích Rust. Giữ benchmark backend của R4 riêng với benchmark học; không so Python core với Rust candidate rồi quy toàn bộ lợi ích cho thuật toán.

### 7.2 Chọn model và đánh giá cuối

Tune/chọn candidates chỉ bằng train/validation. Freeze cấu hình và kế hoạch phân tích trước khi evaluate trên **200 test_id + 200 burst + 200 traffic days**. Cùng ngày trên mọi phương pháp, paired theo scenario. Không dùng test score để quay lại chọn hyperparameters; nếu mở vòng khám phá sau test, phải ghi rõ exploratory và cần bộ confirmatory mới.

Theo plan nền: báo mean/std trên seeds 11,22,33; paired bootstrap **2,000 resamples, seed 6001**, lấy mean model seeds cho từng ngày trước khi resample ngày. Không coi 3×N rows là observations độc lập. CI này mô tả biến thiên theo ngày sau khi mean seeds; biến thiên training seeds báo riêng.

Với cost thấp là tốt, đặt `Δ = cost_candidate − cost_control`. Có bằng chứng cải thiện trên split khi CI 95% của paired mean Δ nằm dưới 0; báo thêm % cải thiện, từng seed và các metric dịch vụ. Nếu CI cắt 0, ghi chưa đủ bằng chứng. Không tuyên bố robust OOD nếu chỉ ID tốt; báo rõ split nào regression.

Ngưỡng regression có ý nghĩa thực tế cho unfinished/abandoned/worst-route và mức improvement mong muốn phải được chốt trong experiment manifest trước test. Không đặt ngưỡng sau khi xem kết quả, không hy sinh âm thầm service quality để giảm tổng cost.

### 7.3 Giữ nghiên cứu ablation có đối chứng

Bộ T9 gốc `core`, `no_reassign`, `no_short`, `fairness_zero` vẫn giữ cùng thuật toán/architecture/budget trong một bộ so sánh. Nếu muốn kiểm tra ablation với PPO đã tune, tạo bộ mới có tên/version riêng và chạy lại các arms tương ứng; không ghép core mới với ablations cũ.

Forecast so với no-forecast có cùng learning config. Tách hiệu ứng tuning, action subsets, reward và observation. Lưu raw components để fairness_zero vẫn được chấm theo core weights.

**Gate L3:** protocol, đủ artifacts đã cam kết, paired results/uncertainty, ID/OOD/service metrics, compute thực tế và failure analysis tái lập được. Gate không yêu cầu RL thắng baseline; kết luận âm hoặc chưa đủ bằng chứng là kết quả hợp lệ.

## 8. Artifacts và checklist thực thi

Paths sau là đề xuất đầu ra, chưa phải artifacts đã tồn tại:

| Giai đoạn | Đầu ra |
|---|---|
| R0–R4 | `reports/rust-migration.md`, `reports/rust-migration/`, raw `runs/rust-migration/` theo Rust spec |
| L0 | `reports/rl-improvement/baseline.md`, paired metrics/curves, run metadata |
| L1/L2 | `reports/rl-improvement/experiments.csv`, mỗi trial có config, hypothesis, seeds, B, elapsed, status và checkpoint links |
| L3 | `reports/rl-improvement.md`, paired ID/OOD tables, CIs, service trade-offs, learning curves và failure traces |
| Raw mới | `runs/rl-improvement/<experiment>/<seed>/`; không ghi đè pilot hoặc full runs cũ |

Manifest experiment ghi backend/build, data/config/obs/action/physical/reward hashes, algorithm settings, actual transitions/episodes, validation protocol, checkpoint hash, candidate search budget và service thresholds. Thêm normalization/forecast artifacts nếu dùng.

- [ ] R4 đạt toàn bộ gate; native backend frozen cho loạt mới.
- [ ] L0 core 3 seeds và heuristics có validation evidence.
- [ ] Chọn hướng L1 theo diagnostics, không sweep vô hạn.
- [ ] Xác nhận tối đa 2 candidates đa seed, giữ cả kết quả âm.
- [ ] Chọn L2 nếu có giả thuyết cần kiểm chứng; không bắt buộc.
- [ ] Freeze trước held-out test; paired statistics đúng đơn vị ngày.
- [ ] L3 báo throughput, sample efficiency và wall-clock efficiency riêng.

## 9. Nguồn và điểm bắt đầu triển khai

- [Rust migration spec](rust_improve.md), [spec nền](spec_v1.0.md), [plan T9–T11](../plan/plan_v1.0.md).
- [Pilot report](../../reports/pilot.md), [Python after](../../reports/training-diagnosis-after.md), [forecast diagnostic](../../reports/forecast.md).
- Code: `src/bus_rl/training/{train,callbacks,diagnose}.py`, `models/features.py`, `rewards/costs.py`, `evaluation/{runner,statistics}.py`, `configs/experiments/`.
- [SB3 RL Tips](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html): đánh giá môi trường riêng, nhiều runs và tuning có kiểm soát. Các candidates trong tài liệu này là đề xuất cho project, không phải kết quả đã xác nhận từ tài liệu SB3.
