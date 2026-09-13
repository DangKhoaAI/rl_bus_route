# Cải thiện RL sau khi tăng tốc bằng Rust

Ngày cập nhật: **2026-09-13**. Trạng thái: **đề xuất thực nghiệm; chưa triển khai theo tài liệu này**.
Implementation plan (English): [improve_RL.md](../plan/improve_RL.md).


## 0. Cách thực thi: danh mục thí nghiệm, không phải danh sách tính năng

**Không triển khai toàn bộ kỹ thuật trong tài liệu này.** L1/L2 là các giả thuyết để lựa chọn và thử riêng, không phải các thành phần bắt buộc phải thêm vào PPO. Mục tiêu là tìm cấu hình đơn giản có bằng chứng tốt hơn; giữ nguyên core nếu không hướng nào đủ bằng chứng. Không thêm code cho một hướng chỉ vì nó có tên hoặc checkbox trong plan.

Quy trình bắt buộc:

1. **L0 trước:** khóa runtime, chạy core 3 seeds và heuristic validation, thu diagnostics. Không code PopArt/BC/encoder/shaping trước kết luận L0.
2. **Chọn một giả thuyết:** dùng diagnostics để chọn một nhóm yếu tố, đăng ký control, candidate, budget, tiêu chí kiểm tra và tiêu chí giữ trước khi chạy. Nếu không có evidence thì DEFERRED/SKIPPED, không implement.
3. **Thử riêng:** candidate dùng config/model ID riêng, mặc định tắt; chỉ sửa phần cần cho giả thuyết. Chạy correctness/smoke trước, rồi screening seed 11. Không đồng thời đổi λ + LR + encoder + reward để tìm một score đẹp.
4. **Xác nhận:** chỉ ứng viên qua screening mới chạy seeds 22/33. So với control cùng settings/runtime và ngân sách thích hợp. Cải thiện trên seed 11 chưa đủ để KEEP.
5. **Quyết định:** KEEP, REJECT, INCONCLUSIVE hoặc DEFERRED theo evidence bên dưới; ghi cả kết quả âm. KEEP giữ thành phần/candidate để tích lũy vào thuật toán custom được chọn; không tự đổi default package hay viết lại core gốc.
6. **Tích lũy thành thuật toán custom:** giữ mọi thành phần đã KEEP và tương thích, không buộc chọn một hướng thắng duy nhất. Nếu A và B đều tốt độc lập, bước tiếp theo mặc định là xây candidate A+B và kiểm tra control, A, B, A+B cùng protocol. Nếu tổ hợp tốt hơn incumbent và đạt service limits thì KEEP A+B làm incumbent mới; nếu tương đương theo ngưỡng đã đăng ký, cân nhắc lợi ích riêng/compute/độ phức tạp. Nếu có tương tác xấu, giữ A/B riêng và ghi nguyên nhân, không xóa kết quả đã tốt. Không cần xin phép riêng để thử tổ hợp trong scope/budget đã đăng ký.
7. **L3 cuối:** khóa candidates và phân tích rồi mới mở held-out test. Nếu test không xác nhận, báo kết quả; không quay lại tune trên test đó.

### 0.1 Thẻ thí nghiệm bắt buộc trước khi code

Mỗi candidate trong ledger phải có: ID/task, hypothesis và link diagnostic L0; control ID; **một nhóm thay đổi**; config diff; implementation tối thiểu; correctness checks; budget/seeds/runtime; metric chính `total_cost_core`; service-regression limits; minimum improvement; screening/confirmation rules; điều kiện dừng và artifact paths. Trường bắt buộc thiếu thì chưa bắt đầu trial. Các giá trị hyperparameter trong tài liệu là gợi ý, không là lệnh chạy toàn grid.

### 0.2 Kiểm gì, khi nào giữ hoặc loại?

| Cổng | Kiểm tra | Quyết định |
|---|---|---|
| Correctness/smoke | Masks, shapes, finite, no-future, terminal/reset, checkpoint; thêm tests riêng của kỹ thuật | Fail: dừng full run, ghi INVALID; sửa bug rồi rerun cùng ID revision mới, không kết luận thuật toán kém từ bug |
| Screening seed 11 | Cost, service metrics, learning stability, compute so control | Chỉ shortlist tối đa 2 candidates/vòng; nếu kém rõ theo rule đăng ký thì REJECT, không chạy thêm seeds để cứu score |
| Confirmation 11/22/33 | Paired validation theo ngày, mean/std từng seed, effect size và service limits | KEEP chỉ khi đạt tiêu chí đã chốt; CI cắt 0 hoặc gain dưới ngưỡng: INCONCLUSIVE, giữ control |
| Held-out L3 | 200 ID + 200 burst + 200 traffic, protocol §7 | Báo xác nhận/không xác nhận và trade-off; không chọn lại hyperparameters từ test |

Rule mặc định cho confirmation: đăng ký minimum meaningful cost reduction **2%** so control trước trial (có thể thay bằng ngưỡng khác chỉ khi giải thích trước chạy); mean reduction đạt ngưỡng, paired day-bootstrap 95% CI của Δ nằm dưới 0, ít nhất 2/3 seeds có mean cost thấp hơn control và mọi service limits đã đăng ký đều đạt. CI validation là công cụ selection, chưa là xác nhận độc lập vì validation đã dùng tuning. Giữ nguyên quy tắc test-blind và uncertainty ở §7. Nếu ưu tiên compute thay vì quality, đăng ký mục tiêu riêng, không gọi nhanh hơn là sample-efficiency gain.

- **KEEP:** correctness + confirmation đạt. Lưu candidate/config/checkpoint, chưa bật default và chưa kết hợp tự động.
- **REJECT:** evidence không đạt rule hoặc regression vượt giới hạn. Giữ logs/results; code thử nghiệm để isolated nếu hữu ích, không gắn vào đường baseline.
- **INCONCLUSIVE:** evidence không đủ/noisy. Mặc định giữ control; chỉ chạy thêm nếu có budget/decision mới đăng ký, không lặp vô hạn tới khi CI đẹp.
- **DEFERRED/SKIPPED:** chưa có trigger, quá effort hoặc không được chọn. Không code, không xem như thất bại khoa học.

Giới hạn vòng đầu: **một vòng L1**, tối đa 6 candidates tính cả control; chọn tối đa 2 để xác nhận. Sau decision L1, mở tối đa 2 hướng L2 và thực hiện **lần lượt**, mỗi hướng một candidate đầu tiên cùng control. Được chủ động mở vòng tiếp theo và tổ hợp khi evidence vòng trước hỗ trợ; ghi hypothesis/budget trước chạy, không cần người dùng duyệt từng vòng. Giới hạn vòng đầu là giới hạn độ rộng tìm kiếm, không phải trần tổng số cải tiến. Không mở chỉ vì còn item chưa tick. Theo dõi incumbent là candidate tốt nhất hiện tại; thử thêm một thành phần trên incumbent và giữ core gốc làm mốc cùng các ablation phù hợp. Có thể dừng với core hoặc L1 đã chọn và SKIP toàn L2. Bắt buộc vẫn là L0 và ma trận nghiên cứu/L3 đã cam kết, không phải thử hết kỹ thuật.

## 1. Quyết định và mục tiêu

**Rust R4, memory M1/M3 và runtime O1/O2 đã có bằng chứng; bước tiếp theo là khóa cấu hình nghiên cứu rồi thực hiện L0.** Rust là hạ tầng rút ngắn vòng lặp nghiên cứu, không phải thay đổi thuật toán học.

Thứ tự chung: **R0 khóa oracle → R1 kernel → R2 observation/mask → R3 tích hợp → R4 parity/speed acceptance → L0 baseline RL → L1 tuning → L2 mở rộng có điều kiện → L3 báo cáo**.

[rust_improve.md](rust_improve.md) là nguồn chuẩn của R0–R4, đặc biệt benchmark/gates §7. Tài liệu này là nguồn chuẩn của L0–L3. [spec_v1.0.md](spec_v1.0.md) và [plan_v1.0.md](../plan/plan_v1.0.md) vẫn xác định physics, metrics và nghiên cứu core/ablation. Không suy luận rằng T9–T11 hay full study đã xong chỉ từ code/spec tồn tại.

Ba câu hỏi phải báo cáo riêng:

1. **Throughput:** Rust tạo transitions và chạy full training nhanh hơn bao nhiêu?
2. **Sample efficiency:** trên cùng số transitions, policy nào đạt cost thấp hơn?
3. **Wall-clock efficiency:** trong cùng thời gian, phương án nào đạt chất lượng tốt hơn?

Không gọi cùng policy chạy nhanh hơn là policy thông minh hơn. Không gọi train nhiều transitions hơn là thắng trên cùng sample budget.

## 2. Điểm xuất phát có bằng chứng

- `reports/training-diagnosis-v1.1-python-improve.md`: learn 2048 transitions 7.69 s, khoảng 266 decisions/s; learn có cProfile. Khoảng 15 phút/full seed là ngoại suy, không là runtime production đã xác nhận.
- `reports/pilot.md`: PPO 12,288 transitions có mean core cost khoảng 18,207 so fixed 16,017 trên 100 validation days; wait thấp hơn nhưng unfinished cao hơn. Chưa phải kết quả full training và chưa chứng minh PPO thắng heuristics.
- Mean cost 15,471.25 trong trial diagnose chỉ là regression mốc cũ, không phải mục tiêu chất lượng RL. Không so trực tiếp với pilot vì khác checkpoint/budget/bộ ngày.
- `reports/forecast.md`: đã có diagnostic forecast, không đồng nghĩa đã chứng minh forecast cải thiện policy.

Trước L0, inventory artifacts full runs hiện có; chỉ đánh dấu hoàn thành việc có checkpoint/config/metadata/results tương ứng. Giữ các artifacts lịch sử; không ghi đè để tạo cảm giác cùng backend/revision.

Bằng chứng hiện tại bổ sung:

- [Rust migration](../../reports/rust-migration.md): R0–R4 accepted, full-seed parity Python/Rust; [memory](memory_optimize.md): M1/M3 đã làm, sparse shared store/lazy kernels được giữ.
- [Runtime optimization](../../reports/runtime-optimization.md): O1/O2 và distribution-validation opt-in đã được kiểm chứng; 3 repetitions seed 11 cho cấu hình batch 16 khoảng 74–85 s/full run, paired speedup khoảng 1.96–2.16× so Rust scalar trong phiên đo. Đây không phải speedup thuật toán hoặc cam kết runtime cho mọi candidate.
- Full-seed 245,760 transitions seed 11 đạt best mean validation cost **13,199.1325** trên 100 ngày, curve/tensors giữ nguyên giữa các runtime được đối chiếu. Không so trực tiếp với pilot/diagnose khác budget/checkpoint; chưa chứng minh thắng held-out hoặc đủ 3 training seeds.
- Nhiều repetitions seed 11 là kiểm định runtime, không thay seeds 22/33. Run thiếu telemetry L0 không được coi là hoàn thành L0 chỉ vì có checkpoint; không thể tái tạo training diagnostics từ final weights.

## 3. Điều kiện vào L0: xác minh và khóa runtime đã đạt

R4 đã được nghiệm thu theo các gates trong [Rust §7](rust_improve.md#7-acceptance-và-benchmark-protocol-nguồn-chuẩn-dùng-chung): parity physics/obs/mask/metrics, integration/checkpoint/forecast, benchmark không profiler, median simulation và learn **≥2×** Python after, full workflow thực đo nhanh hơn. **≥4× chỉ là stretch goal**.

R0 bao gồm đóng băng Python oracle và đo baseline không profiler; không yêu cầu tối ưu Python thêm trước Rust. Tối ưu observation bằng statistics/ring buffers và cache mask thực hiện trong Rust, bảo toàn arrival history và completed channel. Không được bỏ finished khỏi observation mà không có thống kê thay thế tương đương.

Không mở lại migration hoặc chờ các tối ưu deferred để bắt đầu L0. Xác minh build đang dùng khớp evidence; regression mới phải giải quyết trước nghiên cứu. Full-seed kiểm định runtime không tự trở thành thực nghiệm cải thiện thuật toán.

Sau acceptance, đóng băng Rust revision, manifests và dependency versions cho mỗi loạt nghiên cứu. Python là reference/fallback; mọi kết quả loạt mới phải ghi backend, build, hashes và compute thực tế.

## 4. L0 — Thiết lập baseline RL đáng tin

### 4.1 Chạy core trước, không đổi thuật toán

Tạo **config nghiên cứu mới** dự kiến `configs/experiments/rl-improvement/core.toml`, kế thừa learning settings của core và cấu hình CPU 2 threads của `core-threads2.toml`; không sửa config lịch sử. Dùng seeds **11, 22, 33**, mỗi seed **245,760 transitions**. Giữ MaskablePPO, feature extractor hiện tại, 221 actions, reward/physics/guards và tensors như oracle.

Cấu hình đối chứng hiện tại: CPU, n_envs=4, n_steps=256, batch_size=256, n_epochs=4, learning_rate=3e-4, ent_coef=0.01, gamma=1.0, gae_lambda=0.95, clip_range=0.2. Eval mỗi 12,288 transitions; chọn best checkpoint theo validation cost, tie chọn checkpoint sớm nhất như callback hiện tại.

Runtime nghiên cứu được chọn (không đổi default toàn repo):

```toml
[algorithm]
torch_threads = 2

[runtime]
backend = "rust"
eval_batch_size = 16
reuse_eval_pool = true
native_batch = true
validate_distributions = false
```

Đây là phần override; config mới phải chứa toàn bộ learning/physical/control/reward settings cần thiết. Dùng cùng runtime và instrumentation cho các PPO arms; giữ CPU và 4 training envs. Heuristics dùng scalar evaluator với `eval_batch_size=1`, `native_batch=false` nếu batched path không hỗ trợ; ghi rõ đường chạy này, giữ cùng Rust physics/tapes/metrics và báo compute riêng. Forecast hoặc kiến trúc mới phải qua compatibility smoke; không silent fallback, không bỏ feature để giữ tốc độ.

`validate_distributions=false` không bảo đảm logits hữu hạn. L0 phải ghi kiểm tra finite của logits/value/returns/loss/gradients ở các điểm đã đăng ký, fail rõ khi sai; cấu hình debug bật lại validation. Ghi requested/effective flags, lịch kiểm tra và overhead; không âm thầm giảm diagnostics giữa các arms.

Chốt validation ngày và limit dùng chung trước chạy; full comparison dùng 100 validation days theo split hiện có. Không dùng 10 ngày diagnose thay final validation. Seed benchmark R4/runtime chỉ tái dùng nếu config, validation schedule, manifest, provenance và diagnostics cần thiết đúng protocol L0. Có thể tái-evaluate checkpoint để bổ sung per-day metrics nhưng không bịa training telemetry đã thiếu; chạy lại nếu thiếu evidence bắt buộc. Khi thêm instrumentation, không dùng wall time cũ như cùng điều kiện.

Heuristics fixed/threshold/proportional được đánh giá cùng scenario tapes và metric. Tune threshold/proportional trên validation, ghi search budget, rồi freeze trước test. Không cấp future demand hay hidden state cho controller.

### 4.2 Chẩn đoán trước khi chọn thay đổi

Thu thập trên validation và train, chưa dùng held-out test để chọn hướng:

| Bằng chứng cần thu | Câu hỏi trả lời |
|---|---|
| Validation core cost theo transitions và wall clock, best/last checkpoint | Còn học, plateau hay tụt sau một mốc? |
| PPO entropy, approximate KL, clip fraction, value loss, explained variance | Exploration còn đủ? Update quá mạnh? Critic giải thích return được không? |
| Raw cost components, unfinished/abandoned, per-route wait | Cost cao do chờ, bỏ khách, deadhead, fairness hay cuối episode? |
| Action frequency, số action hợp lệ K, valid opportunity và probability mass theo family | Policy ít dùng vì mask hiếm cho phép hay vì không chọn? |
| Entropy H và H/log(K) với K>1; báo riêng K=1 | Exploration giảm thật hay chỉ do support của mask nhỏ đi? |
| Distribution obs/reward/returns, finite checks, trường hợp cực trị | Scaling hoặc outlier làm training khó? |
| Episode traces cho failure cases được chọn theo tiêu chí trước | Hành động nào gây hậu quả và reward trễ bao lâu? |

Các telemetry chưa có trong artifacts phải được bổ sung khi triển khai; không coi bảng này là số đo hiện có. Action family frequency phải kèm số cơ hội hợp lệ, không mặc định NOOP nhiều là policy lỗi.

Định nghĩa diagnostics theo mask: một family có cơ hội khi ít nhất một slot của nó valid; ghi cả số valid slots, tổng xác suất các slot trong family và tỷ lệ được chọn có điều kiện trên cơ hội. Với K=1 ghi entropy=0 và normalized entropy không áp dụng; K=0 là lỗi. Chỉ tính entropy trên support valid, tránh 0×log(0). Đây là telemetry, chưa thay entropy term trong PPO loss. Thu trên cùng mốc/nguồn state đã đăng ký; không so raw action counts có denominator khác nhau.

**Gate L0:** đủ 3 core seeds, baseline validation, learning curves/diagnostics và manifest. Chấp nhận kết quả không thắng heuristics; không thay đổi mục tiêu để làm đẹp score.

## 5. L1 — Cải thiện cách train PPO, giữ nguyên bài toán

Ưu tiên theo bằng chứng L0. Mỗi vòng thay một nhóm yếu tố, luôn có core đối chứng. Các giá trị dưới đây là ứng viên đề xuất, không phải cấu hình đã tốt hơn.

| Ưu tiên | Hướng | Thiết kế nhỏ ban đầu | Bằng chứng để giữ |
|---|---|---|---|
| 1 | Train đủ lâu nếu curve còn giảm | Core budget B=245,760; thử 2B rồi 4B nếu validation còn cải thiện | Improvement theo curve; báo rõ tăng sample/compute, không gộp với fixed-B thắng |
| 2 | Learning rate và độ mạnh PPO update | LR {1e-4, 3e-4}; sau đó mới thử n_epochs {4, 8} hoặc clip_range {0.1, 0.2} tùy KL/clip fraction | Cost đa seed giảm và update ổn định |
| 3 | Exploration | ent_coef {0.003, 0.01, 0.03}; chỉ thử schedule sau fixed coefficients | Không collapse sớm; cost/unfinished tốt hơn, không chỉ entropy cao |
| 4 | Rollout và minibatch | n_steps {128, 256, 512}, giữ n_envs=4; batch_size chia hết rollout size | So ở cùng transitions, ghi số update/thời gian và advantage/value diagnostics |
| 5 | Credit assignment qua GAE | gae_lambda {0.95, 0.98, 1.0}, giữ gamma=1, n_steps và mọi yếu tố khác | Kiểm tra hậu quả trễ khi critic chưa tốt; theo dõi variance advantage, explained variance và cost đa seed |
| 6 | KL early stopping có điều kiện | Khi KL/clip fraction cho thấy update quá mạnh: target_kl {None, 0.01, 0.03}, thử riêng với LR/clip/epochs | Ít collapse và cost tốt hơn; ghi epochs/minibatches thực chạy, early-stop count và compute |
| 7 | Value learning/scaling | Kiểm tra return scale trước; thử vf_coef hoặc normalization riêng khi có dấu hiệu | Critic và validation cải thiện, không chỉ value loss đổi thang đo |

GAE λ cao hơn giảm mức suy giảm trọng số của TD residual xa nhưng có thể tăng variance; λ=0.95 không phải giới hạn tầm nhìn cứng vì critic vẫn bootstrap. Động cơ ở đây là deadhead/cooldown và reward trễ, chỉ mở sau diagnostics L0. `target_kl` là ứng viên **chưa được wire trong config/model factory tại lúc viết**; thêm field nullable, truyền tới MaskablePPO và kiểm tra semantics phiên bản đang khóa trước thử nghiệm. Không nhầm early stopping các epochs PPO với dừng toàn training budget B.

Không chạy Cartesian grid toàn bảng. Screening tối đa **6 cấu hình/vòng tính cả control**, seed 11, cùng B và cùng validation, dùng để loại cấu hình hỏng. Chọn tối đa 2 candidates để xác nhận thêm seeds 22,33 trước kết luận. Báo đầy đủ candidate/seed thất bại và tổng tuning compute. Không lấy seed 11 screening làm kết quả cuối cùng.

`gamma=1.0` giữ nguyên trong L1 vì mục tiêu hiện là tổng cost hữu hạn theo episode. Đổi gamma sẽ thay trọng số thời gian của mục tiêu, phải đưa sang L2. Reward normalization nếu thử chỉ fit/update trên train, freeze stats khi eval, lưu stats cùng checkpoint và báo raw core cost; không normalize metric báo cáo hoặc mask/valid flags như dữ liệu liên tục.

**Gate L1:** quyết định KEEP/REJECT/INCONCLUSIVE theo §0 được ghi với evidence đa seed cho shortlisted candidates, có control đúng B, configs/hashes và validation evidence; nếu không tốt hơn thì giữ core. Không bắt buộc tìm được cải thiện.

## 6. L2 — Mở rộng có điều kiện, tách từng giả thuyết

Chỉ chọn hướng có bằng chứng từ L0/L1; không triển khai tất cả cùng lúc.

### 6.1 Forecast causal

Ưu tiên kiểm tra module có sẵn: `configs/experiments/forecast.toml` so no-forecast core với cùng seeds/budget và backend. Fit historical forecaster trên train arrival logs rồi freeze. Giữ tensor shape và enabled flag; kiểm thử hai tapes giống quá khứ nhưng khác tương lai cho cùng prediction tại t.

MAE tốt không bảo đảm điều khiển tốt. Báo đồng thời forecast error và control cost/unfinished. Forecast không dùng scenario seed, future tape hoặc thông tin mà baseline không được phép truy cập.

### 6.2 Encoder hoặc bộ nhớ cho partial observability

Nếu flat MLP hiện tại khó biểu diễn cấu trúc route/stop/vehicle, thử shared encoder theo từng stop/vehicle rồi masked pooling/fusion, giữ obs/action contract ở vòng đầu. Giữ route/direction/position features cần cho điều phối, không pooling mất vị trí; padding không được đóng góp. So với MLP có parameter budget gần tương đương, ghi chênh lệch tham số để tách lợi ích cấu trúc khỏi tăng capacity. Nếu history hiện tại thiếu thông tin dự báo, mới xem xét temporal encoder/recurrent policy.

Đây là thay đổi kiến trúc, cần implementation/compatibility spike riêng với MaskablePPO, masks, recurrent state/reset nếu có; không giả định bật recurrent là sẵn chạy. So cùng transitions, báo parameter count và inference/training wall. Đổi schema thì version/hash mới và không load checkpoint cũ như tương thích.

### 6.3 Reward hoặc credit assignment

Chỉ sau khi xác định cost cao đến từ component nào. Thử từng giả thuyết: scale penalty terminal unfinished, fairness trade-off hoặc shaping có giải thích. Giữ physics, demand, masks và guards; ghi rõ đây là thay đổi learning objective.

Nếu thử shaping, ưu tiên giả thuyết potential-based riêng: `r'_t = r_t + gamma*Phi_{t+1}(o_{t+1}) - Phi_t(o_t)`, potential causal, cố định trong một experiment, bằng 0 ở mọi terminal thật. Với gamma=1 và cùng phân phối initial state, phần thêm telescope thành hằng số theo initial state; kiểm tra bằng fixture trên nhiều action traces. Truncation/bootstrap phải xử lý riêng, không giả terminal=0 khi chưa kết thúc nhiệm vụ. Không double-count terminal settlement, không dùng destination ẩn/future tape. Điều kiện này không bảo đảm PPO xấp xỉ sẽ học giống nhau hoặc tốt hơn; version shaping, lưu cả reward gốc và shaped, so bằng objective gốc. Đây là hướng L2 có điều kiện, không sửa reward baseline.

Luôn tính lại **total_cost_core với weights gốc** từ raw components để so chính. Không gọi reward lớn hơn là tốt hơn khi đổi weights. Báo trade-off completion, abandonment, waiting/P95 và chi phí vận hành. Nếu đổi gamma/shaping, ghi mục tiêu mới và không tuyên bố policy invariance khi chưa chứng minh điều kiện áp dụng.

### 6.4 Curriculum hoặc warm-start từ heuristic

Chỉ thử khi exploration/cold-start là vấn đề. Curriculum chỉ dùng train split, công bố lịch phân phối demand và bảo đảm giai đoạn cuối có train distribution mục tiêu. Warm-start chỉ dùng demonstrations từ train, rồi fine-tune bằng RL.

Báo riêng transitions/demonstrations và compute phụ; so PPO-from-scratch công bằng. Không dùng test/OOD held-out để tạo demonstration hoặc curriculum thích nghi theo test score.

### 6.5 Critic target normalization bằng PopArt — ứng viên mới

Nguồn: [van Hasselt et al., Learning values across many orders of magnitude](https://arxiv.org/abs/1602.07714). Paper đề xuất normalization target thích nghi kèm bảo toàn output chưa normalize khi statistics đổi. Áp dụng ở đây là giả thuyết của project, chưa có số đo cải thiện.

Chỉ mở nếu L0 thấy return/terminal penalties lệch thang đo giữa ngày và explained variance kém. Dùng critic dự đoán target chuẩn hóa với running mean/std học từ train returns; khi stats đổi, rescale output head để giá trị dự đoán ở đơn vị reward gốc không nhảy chỉ do đổi normalization. Actor loss vẫn dùng advantage ở contract đã định, reward gốc và total_cost_core không đổi.

Thiết kế: control là PPO được chọn sau L1; candidate thêm PopArt cho value head, cùng B/seeds/runtime. Không đồng thời bật VecNormalize reward, đổi vf_coef hoặc đổi architecture. GAE/bootstrap/old values phải cùng đơn vị reward gốc; value loss dùng target/value chuẩn hóa nhất quán. Chốt stats update tại rollout boundary, ổn định epsilon/std floor và optimizer-state treatment trước code; lưu mean/variance/count cùng checkpoint, freeze tại eval. Nếu value clipping được thêm sau này, xác định lại đơn vị clipping.

Acceptance: test đổi stats giữ unnormalized predictions trong tolerance khai báo; constant-return/zero-variance finite; GAE hand fixture đúng đơn vị; save/load và frozen eval stats đúng; so critic error ở đơn vị gốc, raw cost/service metrics đa seed. PopArt không phải drop-in config của model hiện tại; cần custom head/loss integration spike. Khi diagnostics không chứng minh vấn đề scale, SKIPPED.

### 6.6 Research shortlist và protocol cụ thể

Các nguồn dưới đây là nguồn phương pháp gốc; mức ưu tiên và cách chuyển sang bus control là suy luận thiết kế, không phải kết quả paper trên bài toán này.

| ID / task | Kỹ thuật | Trigger từ L0 | Thử nghiệm và đối chứng |
|---|---|---|---|
| BC-PPO / L2.4 | Masked behavior cloning rồi PPO fine-tune | Policy cold-start kém hơn heuristic, action family hữu ích ít được khám phá | Cùng PPO architecture, BC trên demonstrations train-only, sau đó PPO; đối chứng PPO scratch và heuristic teacher |
| SET-PPO / L2.2 | Shared entity encoder + masked pooling, theo Deep Sets | Flat MLP cần nhiều samples hoặc khó chia sẻ quy tắc giữa xe/trạm | So flat MLP với structured encoder có parameter count gần nhau, giữ 221 slots và reward |
| POPART / L2.5 | Value target normalization bảo toàn output | Return scale khác mạnh và critic học kém | Chỉ đổi value normalization; không gộp reward clipping/normalization |
| PBRS / L2.3 | Potential shaping causal | Hậu quả dispatch/short-turn trễ, GAE tuning chưa giải quyết | Potential cố định dựa trên observation hợp lệ; so cùng PPO không shaping và chấm raw core cost |

**BC-PPO cụ thể.** Teacher là threshold hoặc proportional được chọn bằng validation và khóa trước thu demonstrations; không gọi teacher là expert tối ưu. Ghi `(obs, mask, action)` trên train scenarios, tối thiểu cả episode để giữ trạng thái tự nhiên. Pretrain actor bằng masked negative log-likelihood `-log pi(a_teacher|obs,mask)`; không huấn luyện critic bằng nhãn action. Nếu shared encoder thay đổi, critic sẽ cần học lại giá trị trong PPO. Manifest ban đầu thử một budget 12,288 demo transitions và tối đa 5 epochs BC; đây là budget đề xuất, không phải thông số tối ưu. Early selection BC bằng split nội bộ theo train-day, không random chia các timestep cùng ngày qua hai tập.

Báo hai đối chứng: scratch B vs BC(D)+PPO(B) để đo warm-start với extra data, và scratch(B+D) để đối chiếu tổng env-transition budget; vẫn báo riêng demo labeling/BC compute. Theo dõi imitation accuracy cùng return, không dùng accuracy làm quality gate. Dữ liệu teacher có NOOP imbalance: báo family counts, chưa reweight loss ở lần thử đầu. Nếu thử DAgger sau BC, thu nhãn teacher trên states do learner thăm và aggregate train-only data; không reset teacher state giữa các action tùy tiện. [Ross et al., DAgger](https://arxiv.org/abs/1011.0686) giải quyết distribution shift trong imitation tuần tự; không suy ra heuristic labels sẽ tốt hơn PPO. DAgger là follow-up riêng, không bắt buộc và không nằm trong budget BC ban đầu.

**SET-PPO cụ thể.** [Zaheer et al., Deep Sets](https://arxiv.org/abs/1703.06114) cung cấp kiến trúc cho hàm bất biến theo permutation của tập. Bus có topology và action IDs, nên không coi mọi trạm/tuyến hoán vị tự do. Encode từng vehicle/stop với route/direction/position và ID mapping cần thiết; pooling bỏ padding bằng validity mask rồi fuse context/history. Giữ đường thông tin entity-to-action, không chỉ global average làm mất xe nào ứng với slot nào. Test padding invariance; chỉ test permutation khi đồng thời remap observation IDs/action slots hợp lệ. Không tuyên bố generalization topology từ cùng mạng cố định.

**PBRS cụ thể.** [Ng et al., policy invariance under reward transformations](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf) là nền tảng potential-based shaping. Candidate đầu đơn giản: potential âm từ queue và excessive-wait quan sát được, scale cố định theo train stats, time-dependent về 0 ở terminal theo §6.3. Không dùng state ẩn hoặc thông tin tương lai. Tránh thêm bonus dispatch/boarding tùy ý; chúng có thể đổi mục tiêu. Nếu shaping chỉ lặp thông tin reward đã dense và không cải thiện quality ở fixed B thì loại.

Không mở toàn shortlist cùng lúc. Sau L0/L1 chọn tối đa **hai hướng L2** trong đợt đầu theo evidence; mỗi hướng có control riêng và xác nhận 3 seeds trước kết hợp. Forecast sẵn có vẫn có thể được chọn thay một hướng. PopArt, DAgger, recurrent/Transformer không phải prerequisites. Asymmetric privileged critic chưa chọn vì mở rộng training-information contract; không đưa hidden destination/future tape vào critic trong core hiện tại. Giữ invalid-action masking trong cả rollout và PPO update, theo [Huang & Ontañón](https://arxiv.org/abs/2006.14171); không đánh đổi guards lấy exploration.

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

Runtime parity và algorithm acceptance là hai việc khác nhau: thay backend/packing/runtime phải giữ hành vi theo spec tối ưu; thay λ, LR, entropy hoặc architecture được phép tạo weights/actions khác core. Candidate thuật toán phải qua correctness, masks, finite/checkpoint tests rồi so chất lượng đa seed; không yêu cầu bit-identical với thuật toán đối chứng. Giữ runtime cố định trong một contrast để không quy lợi ích batching cho algorithm.

## 8. Artifacts và checklist thực thi

Paths sau là đề xuất đầu ra, chưa phải artifacts đã tồn tại:

| Giai đoạn | Đầu ra |
|---|---|
| R0–R4 | `reports/rust-migration.md`, `reports/rust-migration/`, raw `runs/rust-migration/` theo Rust spec |
| L0 | `reports/rl-improvement/baseline.md`, paired metrics/curves, run metadata |
| L1/L2 | `reports/rl-improvement/experiments.csv`, mỗi trial có config, hypothesis, seeds, B, elapsed, status và checkpoint links |
| L3 | `reports/rl-improvement.md`, paired ID/OOD tables, CIs, service trade-offs, learning curves và failure traces |
| Raw mới | `runs/rl-improvement/<experiment>/<seed>/`; không ghi đè pilot hoặc full runs cũ |

Manifest experiment ghi backend/build và actual library hash, torch_threads requested/effective, eval_batch_size, native_batch, reuse_eval_pool, validate_distributions requested/effective, instrumentation/finite-check settings, data/config/obs/action/physical/reward hashes, algorithm settings, actual transitions/episodes, validation protocol, checkpoint hash, candidate search budget và service thresholds. Thêm normalization/forecast artifacts nếu dùng.

- [ ] Xác minh evidence R4 + memory/runtime hiện có và khóa runtime/config/build cho loạt L0 (không chạy lại migration).
- [ ] L0 core 3 seeds và heuristics có validation evidence.
- [ ] Chọn hướng L1 theo diagnostics, không sweep vô hạn.
- [ ] Xác nhận tối đa 2 candidates đa seed, giữ cả kết quả âm.
- [ ] Chọn L2 nếu có giả thuyết cần kiểm chứng; không bắt buộc.
- [ ] Freeze trước held-out test; paired statistics đúng đơn vị ngày.
- [ ] L3 báo throughput, sample efficiency và wall-clock efficiency riêng.

## 9. Nguồn và điểm bắt đầu triển khai

- [Runtime optimization spec](runtime_optimize.md), [runtime results](../../reports/runtime-optimization.md), [memory spec](memory_optimize.md).
- [Rust migration spec](rust_improve.md), [spec nền](spec_v1.0.md), [plan T9–T11](../plan/plan_v1.0.md).
- [Pilot report](../../reports/pilot.md), [Python after](../../reports/training-diagnosis-v1.1-python-improve.md), [forecast diagnostic](../../reports/forecast.md).
- Code: `src/bus_rl/training/{train,callbacks,diagnose}.py`, `models/features.py`, `rewards/costs.py`, `evaluation/{runner,statistics}.py`, `configs/experiments/`.
- [SB3 RL Tips](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html): đánh giá môi trường riêng, nhiều runs và tuning có kiểm soát. Các candidates trong tài liệu này là đề xuất cho project, không phải kết quả đã xác nhận từ tài liệu SB3.
