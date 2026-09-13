# Tối ưu runtime sau Rust: batching, caching và parallelism

Ngày: **2026-09-12**. Trạng thái: **Đề xuất; O0–O2 đã triển khai/nghiệm thu (opt-in);
O3 hoãn theo số đo; O5 đã có protocol nhiều repetition nhưng chưa đổi default; O4 chưa.**
Kết quả: [reports/runtime-optimization.md](../../reports/runtime-optimization.md).**

Tài liệu kế thừa [rust_improve.md](rust_improve.md),
[memory_optimize.md](memory_optimize.md) và
[báo cáo migration](../../reports/rust-migration.md).
Spec này xác định phạm vi, contract và acceptance của đợt tối ưu tiếp theo;
không phải báo cáo kết quả hay implementation plan đã hoàn thành.

## 1. Mục tiêu và quan hệ với thực nghiệm RL

Giảm wall time của train + validation và tăng số thực nghiệm hoàn thành trên
cùng tài nguyên, bằng cách giảm lời gọi nhỏ, allocation và công việc lặp ở
ranh giới Python/Rust/PyTorch. Mốc so sánh chính là **Rust sau R4 và các tối ưu
bộ nhớ đã đạt**, không phải Python trước port.

Không mở lại R0–R4, không tự thay đổi trạng thái L0–L3 trong
[improve_RL.md](improve_RL.md). Đợt này là nhánh runtime độc lập: chỉ thay
runtime dùng cho nghiên cứu tại ranh giới một đợt so sánh, sau O5. Các run đã
bắt đầu tiếp tục dùng revision đã khóa. Tối ưu không đạt vẫn được ghi nhận
là rejected/deferred; runtime hiện tại tiếp tục hợp lệ.

Giữ nguyên physics, thứ tự tick/boarding, reward, 221 action slots, guards,
observation, scenario hashes, thuật toán PPO và protocol validation.
Tăng throughput không chứng minh policy tốt hơn trên cùng số transitions.

## 2. Baseline đã có: không triển khai lại

| Hạng mục | Trạng thái theo tài liệu/code hiện tại | Cách kế thừa |
|---|---|---|
| Native kernel, observation/history, mask cache, Gym/evaluator, checkpoint provenance | R0–R4 accepted | Giữ làm reference và đường scalar để đối chiếu |
| Shared sparse native store, `Arc<Scenario>`, lazy kernel | Đã có | Chia sẻ store; chỉ tạo state cho env đang hoạt động |
| Tape int8, digest chuẩn hóa int32 | Memory M1 đã làm | Giữ kiểm tra miền giá trị và hash |
| Metadata-only Python scenarios trong Rust run | Memory M3 đã làm | Giữ đường này; hiện CLI train/diagnose/baseline không forecast dùng được, evaluate/forecast còn nạp đầy đủ |
| Canonical sparse Python tape, bỏ store, lazy/mmap | M2/M4/M5 còn deferred hoặc một phần | Không đưa lại thành công việc bắt buộc của spec này |
| Torch threads = 2 | Đã chọn trong `core-threads2.toml` | Baseline dùng config này; default trong `AlgorithmConfig` vẫn là 16 |
| Observation validation switch | Đã có, mặc định bật | Hai phía benchmark cùng setting; không tính tắt validation là batching speedup |

Snapshot báo cáo hiện tại: full workflow 245,760 transitions + 20 lần
validation × 100 ngày, Python/Rust total wall **597.99/156.37 s**, RSS
**479/445 MB**. Đây là số lịch sử để định hướng; O0 phải đo lại, không dùng
trực tiếp làm mẫu số cho candidate mới. Native revision được tài liệu ghi là
`aedc4faa4c43`; O0 xác minh artifact/build thực tế thay vì giả định HEAD giống nó.

Profile 16-thread cũ cho thấy eval inference chiếm ~82%, PPO update ~57%
learn. Các tỷ trọng này có profiler và khác cấu hình 2-thread; chỉ là giả
thuyết ưu tiên, không phải profile hiện tại hay cam kết speedup.

## 3. Điểm còn có thể tối ưu

- `evaluation/runner.py::evaluate_scenarios` dùng một env, duyệt scenario
  tuần tự; `PPOController.act` gọi predict một observation mỗi lần.
- `training/train.py` dùng `DummyVecEnv`: inference đã nhận batch các env,
  nhưng stepping còn lần lượt qua từng wrapper/native call.
- `Kernel.step_contract` tạo dict/arrays cho từng env; wrapper chuyển đổi
  observation, costs và mask trước khi VecEnv gom kết quả.
- `BestValidationCallback` gọi `mean_cost` đồng bộ; mỗi đợt dựng evaluator
  env mới. Scenario store đã cache nên cần đo riêng phần setup còn lại.
- `_STORE_CACHE` hiện có giới hạn 4 entry theo danh sách hash/flags. Cache
  mới phải kiểm soát lifetime, tránh giữ thêm nhiều store/pool cho cùng dữ liệu.

## 4. Contract bất biến

### 4.1 Simulation và dữ liệu

Replay cùng scenario và action trace phải giữ exact counters, IDs, lineage,
status, mask và event ordering. Observation dùng `rtol=atol=1e-6`;
cost/reward dùng `rtol=atol=1e-9`, kế thừa Rust spec. Không nới tolerance
sau khi thấy lỗi. Không dùng future tape cho policy/forecast.

Batch output có trục đầu là env, thứ tự cố định; mỗi env có RNG và state
riêng. Giữ đúng seed, scenario selection, episode start, terminal observation,
terminated/truncated, auto-reset và `StepCosts`. Output cũ không bị lần
step/reset kế tiếp hoặc caller khác sửa. Internal reusable buffers chỉ được
dùng với ownership/copy boundary chứng minh bằng test.

### 4.2 Training và evaluation

Giữ 4 env, n_steps=256, batch_size=256, n_epochs=4 và mọi hyperparameter
khác như baseline. Không tăng n_envs rồi giảm n_steps để gọi đó là cùng
thuật toán; độ dài trajectory và bootstrap đã thay đổi.

Giữ 100 validation days, thứ tự aggregation và lịch 12,288 transitions;
không dùng eval-limit nhỏ hoặc validation thưa hơn để đạt speed gate.
Giữ best-checkpoint selection, tie rule và save/load behavior.

Fixed checkpoint scalar/batch phải có cùng deterministic actions và
per-scenario metrics trong tolerances, ngoại trừ wall-time fields được đo
riêng. Batch inference có thể đổi rounding và argmax gần hòa: ghi first
divergence, logits/mask và reproducer; candidate chưa đạt compatibility nếu
action đổi, dù mean cost gần nhau. Không silently đổi tie-break.

Với train, yêu cầu cùng rollout ordering và RNG consumption. Full seed phải
đối chiếu validation curve, lựa chọn best, policy/optimizer/tensor states như
baseline bit-identical đã đạt; so nội dung tensor, không hash toàn ZIP có
timestamp. Nếu không đạt, candidate ở trạng thái experimental, chưa thay
runtime của đợt RL. Không giảm yêu cầu chỉ vì simulation golden đã qua.

## 5. O0 — Khóa Rust baseline và đo lại bottleneck

Ghi revision/dirty diff, native build, dependency lock, config/manifests,
checkpoint, seeds, máy, Torch/BLAS/native threads, logging/trace/validation
flags. Giữ baseline có thể chạy lại cạnh candidate.

Đo riêng load/pack, env reset, rollout inference, native step, FFI/array
packing, wrapper/VecEnv, buffer, PPO update, validation inference/step/summary,
checkpoint I/O và total wall. Timer lồng nhau không cộng chồng. Profile riêng
để giải thích; speed acceptance tắt profiler và TIMERS.

Đầu ra: baseline manifest, raw timings/RSS và thứ tự ưu tiên theo profile
2-thread. Nếu bottleneck khác giả thuyết, cập nhật ưu tiên với bằng chứng
trước khi triển khai stage phụ thuộc.

## 6. O1 — Batched evaluation và tái sử dụng evaluator

Ưu tiên đầu tiên: chạy đồng thời một nhóm scenario và gọi model một batch
observation/mask mỗi control step. Thử batch 1/4/8/16/32, giới hạn bằng số
ngày còn lại và RAM; không thay training batch_size hoặc training n_envs.
O1 vẫn được step env tuần tự, chưa cần native threads.

- Gán scenario theo thứ tự input, ghi scenario ID cho từng slot; xử lý batch
  cuối không đầy, horizon khác nhau và slot đã done. Mỗi ngày hoàn thành đúng
  một lần; không step env đã kết thúc. Kết quả trả về theo input order.
- Lấy terminal summary/trace trước reset/reuse slot; không trộn reward,
  costs, lineage hoặc forecast cache giữa các ngày.
- Forecast có state phải có state riêng mỗi slot và reset mỗi episode.
  Nếu chưa hỗ trợ, cấu hình batch + forecast báo lỗi rõ; không bỏ forecast
  hoặc âm thầm chuyển chế độ. Đường scalar vẫn dùng được bằng lựa chọn rõ ràng.
- O1 tập trung deterministic PPO; random/stateful heuristic giữ scalar
  cho tới khi có contract RNG phù hợp, tránh đổi thứ tự lấy số ngẫu nhiên.
- Pool sống trong evaluator/callback, chia sẻ store bất biến; reset sạch
  trước mỗi lượt. Không cache action/logits/value/metrics xuyên checkpoint.
  Invalidate pool khi scenario order/hash, flags, config, forecast hoặc
  contract thay đổi; giải phóng khi kết thúc, không global cache vô hạn.
- Validation vẫn đồng bộ và giữ nguyên model mode/RNG sau evaluation.

Acceptance: batch=1 tương đương scalar; mọi batch được bật đạt §4 và không
leak state sau ít nhất 20 lượt evaluation. Báo time và RSS theo batch size.
Tách benchmark batching khỏi pool reuse bằng ablation, tránh gộp lợi ích.

## 7. O2 — Native batch step và giảm allocation

Đề xuất `BatchKernel` sở hữu N episode states, tham chiếu shared store và
cung cấp reset theo slot, mask batch, step batch và terminal summary theo
slot. Tên API là đề xuất; chốt signature trước triển khai.

Một call nhận N actions, thực hiện đúng một control interval trên mỗi env
và trả dict các arrays `(N, *single_env_shape)`, rewards/dones/masks/costs.
Không batch nhiều bước thời gian khi policy chưa chọn action kế tiếp.

- Làm bản tuần tự trước để tách lợi ích giảm FFI/allocation khỏi parallelism.
- Prevalidate shape, slot IDs và toàn bộ actions/masks trước mutation; lỗi
  đầu vào phải giữ nguyên tất cả state. Lỗi nội bộ giữa batch phải fail rõ
  và đánh dấu batch không tiếp tục được tới khi reset; không trả partial
  success như một transition hợp lệ.
- Adapter SB3 giữ Gym/VecEnv terminal-info, auto-reset, seeding và interfaces
  collector/MaskablePPO thực sự sử dụng. Test với collector thật và checkpoint.
- Đóng gói array theo batch, tránh Python object graph từng tick; không dựng
  Python WorldState trong train. Giữ copy cần thiết bảo vệ rollout buffer.
- Reset slot dùng store đã pack; không giữ kernel cho mọi scenario.

Acceptance: scalar-vs-batch replay, interleaved envs, retained outputs,
invalid batch, partial reset, VecEnv auto-reset và full training parity đạt.
Benchmark với cùng 4 env; không lấy tăng số env làm lợi ích của O2.

## 8. O3 — Parallel native envs, có điều kiện

Chỉ thử khi O2 profile còn chi phí native đáng kể. Worker pool cố định,
thử 1/2/4 workers trong tài nguyên máy; chia việc theo env, giữ tuần tự toàn
bộ tick/xe/khách bên trong mỗi env. Output luôn theo slot, không theo thứ tự
worker hoàn thành. Không shared mutable RNG hoặc float reduction liên env.

Tách compute Rust khỏi Python object construction; quản lý GIL theo phiên
bản PyO3 đang khóa. Worker không gọi Python/forecast trong vùng native
parallel. Đếm tổng Torch/BLAS/Rayon threads và giới hạn oversubscription.

`SubprocVecEnv` chỉ là ứng viên benchmark riêng nếu native pool không phù
hợp: tính cả IPC, startup và bản sao store/process; không mặc định nhanh hơn.

Acceptance: O2 contract vẫn đạt; đo thêm thread scaling, pool teardown,
worker error và tổng RSS. Không có lợi vượt noise thì giữ worker=1 và ghi
O3 deferred/rejected, không bắt buộc parallel để hoàn thành đợt tối ưu.

## 9. O4 — Caching, prefetch và chạy nhiều seed có điều kiện

### 9.1 Cache bổ sung

Chỉ cache feature/template thực sự bất biến nếu O0/O2 chứng minh bị dựng lại
đáng kể. Khóa bao gồm mọi input ảnh hưởng; có invalidation và giới hạn bytes.
Không cache toàn bộ observation theo hash state khi chưa đo chi phí khóa,
hit rate và bộ nhớ. Không lặp lại mask/store cache đã có.

Mở metadata-only cho evaluate hoặc forecast là thay đổi riêng, phải kiểm
tra mọi consumer cần tape và forecast causality; không giả định M3 đã hỗ trợ.

### 9.2 Prefetch

Mặc định tắt: working set hiện được pack sẵn trong RAM. Chỉ mở khi chứng
minh I/O/load/packing hoặc cold reset là bottleneck. Queue có giới hạn bytes,
giữ scenario ordering, kiểm tra hash, truyền lỗi và có cancel/cleanup.
Ghi cold/warm separately và tính startup vào total wall.

Không suy đoán action tương lai; không dùng tape tương lai làm observation.
Không overlap PPO update với thu rollout bằng weights cũ trong spec này.
Lazy/mmap diện rộng vẫn thuộc memory M5, không tự mở lại ở đây.

### 9.3 Nhiều seed độc lập

Có thể thử launcher giới hạn 1/2/4 jobs, mỗi job riêng seed/output/metadata
và thread budget. Báo tổng seed/giờ, makespan, latency từng seed và tổng RSS
cây tiến trình. Không coi throughput nhiều jobs là speedup một seed, không
chạy benchmark single-job trong lúc launcher đang chiếm tài nguyên.
Đây là nhánh độc lập, không thay nội dung/thứ tự dữ liệu của từng run.

## 10. O5 — Benchmark và quyết định sử dụng

### 10.1 Protocol

1. Rust release, cùng máy/device/dependencies, initial model, scenarios và
   workload. Torch threads=2 cho baseline chính. Mọi thay đổi threads được
   báo như thí nghiệm riêng; candidate/baseline có cùng resource budget.
2. Warm-up riêng; tối thiểu 5 repetitions xen kẽ cho microbench, fixed
   checkpoint eval và isolated learn 12,288 transitions. Báo raw, median,
   min/max, actual transitions; giải thích/rerun khi noise che lợi ích.
3. Evaluation dùng cùng 100 ngày; có zero/normal/peak/burst/traffic và cases
   episode/reset/trace/forecast ở correctness suite. Báo riêng từng workload.
4. Candidate cuối và baseline chạy full 245,760 transitions với 20 × 100
   validation, ít nhất 3 paired repetitions cùng seed 11, đảo thứ tự chạy.
   Phân biệt repetition runtime với nhiều training seeds của nghiên cứu RL.
5. Tính setup, train, validation, checkpoint, total wall và peak RSS của mỗi
   process; multiprocessing đo thêm peak tổng cây tiến trình. So cả live
   retained bytes sau nhiều resets/evals để phân biệt leak với allocator cache.
6. Ablation: baseline → eval batch → pool reuse → native batch sequential
   → native workers. Cache/prefetch và concurrent seeds có bảng riêng.

### 10.2 Gates đề xuất cho đợt mới

Các ngưỡng dưới đây là tiêu chí lựa chọn mới, **chưa phải kết quả đã đạt**;
không tái sử dụng gate 2× Python của R4 để tuyên bố tối ưu hậu Rust thành công.

| Gate | Điều kiện |
|---|---|
| Correctness | §4 và tests của mọi feature được bật đạt; không unresolved divergence |
| Component | Candidate giữ lại giảm median wall phần nhắm tới ít nhất 10% so với baseline tương ứng; lợi ích phải lớn hơn dao động đo |
| Full workflow | Cấu hình cuối giảm median total wall ít nhất 10% so với Rust O0; không workload bắt buộc nào chậm hơn 5% chưa giải quyết |
| Memory | Peak RSS cấu hình single-job cuối không quá 1.20× baseline đo lại; không tăng retained state theo số episode/eval sau warm-up |
| Training | 2048-transition smoke finite/save-load; full-seed parity §4.2 đạt |
| Reproducibility | Commands, raw results, config/build hashes và lựa chọn fallback đầy đủ |

Chọn batch/worker nhỏ nhất trong nhóm có hiệu năng tương đương khi noise
không phân biệt được. Nếu chỉ component đạt nhưng full workflow không đạt,
giữ opt-in/experimental, không đổi default. Tối ưu không lợi có thể đóng
bằng rejected/deferred có evidence; không hạ ngưỡng sau benchmark.

## 11. Cấu hình, artifacts và bàn giao

Các field sau **đề xuất, chưa tồn tại**: `runtime.eval_batch_size` (1),
`runtime.reuse_eval_pool` (false), `runtime.native_batch` (false),
`runtime.validate_distributions` (true), `runtime.native_workers` (1). Prefetch/launcher chỉ thêm cấu hình nếu stage
đó được mở. Validate tổ hợp không hỗ trợ và ghi requested/effective values
vào metadata; không silent fallback. Các giá trị mặc định ban đầu giữ đường
hiện tại, chỉ đổi sau O5 accepted.

Artifacts mới dự kiến:
- `reports/runtime-optimization.md`: baseline, parity matrix, ablations,
  benchmark, accepted/rejected/deferred và giới hạn.
- `reports/runtime-optimization/`: manifests, profile, raw timing/RSS,
  per-day paired metrics, tensor comparison và commands/exit status.
- `runs/runtime-optimization/`: logs/checkpoints/traces lớn, được link/hash
  từ report; giữ nguyên evidence Rust/memory cũ.

Checklist ban đầu:
- [x] O0: baseline sau R4 + M1/M3 được khóa và đo lại.
- [x] O1: eval batching/pool reuse được kiểm chứng riêng (opt-in).
- [x] O2: native batch sequential và ownership/VecEnv đạt (opt-in).
- [ ] O3: native parallel — hoãn; số đo O2 cho thấy native step không còn trội.
- [ ] O4: cache/prefetch/concurrent seeds có quyết định riêng; được phép deferred.
- [ ] O5: full workflow gates đạt với protocol nhiều repetition (O2 1.96–2.16× paired,
  batch 16 vs 32 là nhiễu) nhưng chưa chạy nhiều seed/host và chưa đổi default.

## 12. Nguồn triển khai cần đối chiếu

- `src/bus_rl/evaluation/runner.py`, `evaluation/summary.py`: scalar rollout, metrics, trace.
- `src/bus_rl/learning/training/train.py`, `training/callbacks.py`: VecEnv, validation, checkpoint selection.
- `src/bus_rl/execution/environments/rust/environment.py`, `environments/factory.py`: reset/step, observation, mask ownership.
- `src/bus_rl/backend/native.py`: sparse store packing, metadata tapes và shared cache.
- `crates/bus-sim-python/src/lib.rs`: native boundary, contract arrays và kernel lifetime.
- `crates/bus-sim-core/src/engine.rs`, `observation.rs`, `guards.rs`: compute giữ nguyên semantics.
- `src/bus_rl/config.py`, `configs/experiments/core-threads2.toml`: defaults và baseline thực nghiệm.
- `scripts/benchmark_backends.py`, `profile_native_training.py`, `tune_runtime.py`: tận dụng harness hiện có.
- `tests/parity/`: mở rộng parity/ownership/integration suite khi triển khai.
