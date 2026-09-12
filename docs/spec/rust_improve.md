# Rust kernel: tăng tốc simulation trước thực nghiệm RL

Ngày cập nhật: **2026-09-12**. Trạng thái: **spec triển khai; chưa implement Rust**.
Implementation plan (English): [rust_improve.md](../plan/rust_improve.md).


## 1. Quyết định và quan hệ với kế hoạch RL

**Làm Rust trước**, để simulation và training nhanh hơn, rồi mới thực hiện các vòng thử nghiệm cải thiện thuật toán trong [improve_RL.md](improve_RL.md). Không yêu cầu một đợt tối ưu Python mới trước Rust. Python after được giữ làm oracle và backend fallback.

Thứ tự chung: **R0 khóa oracle → R1 kernel → R2 observation/mask → R3 tích hợp → R4 parity/speed acceptance → L0 baseline RL → L1 tuning → L2 mở rộng có điều kiện → L3 báo cáo**. R0–R4 được định nghĩa ở tài liệu này; L0–L3 ở tài liệu RL. Chỉ bắt đầu thực nghiệm thuật toán khi R4 đạt.

Spec nền: [spec_v1.0.md](spec_v1.0.md). Plan nghiên cứu: [plan_v1.0.md](../plan/plan_v1.0.md). Hai tài liệu improve bổ sung thứ tự triển khai mới; không tự đánh dấu T9–T11 hay các thực nghiệm đã hoàn thành. Tình trạng kết quả phải dựa vào artifacts thực tế.

Mục tiêu port là giữ nguyên bài toán và thuật toán, giảm thời gian thực nghiệm. Tăng throughput không đồng nghĩa tăng chất lượng policy trên cùng số transitions.

## 2. Bằng chứng hiệu năng và giới hạn dự báo

Nguồn gốc giữ nguyên:

| Vai trò | Artifacts |
|---|---|
| Python trước tối ưu | `reports/training-diagnosis.md`, `reports/training-diagnose/` |
| Python after, oracle | `reports/training-diagnosis-after.md`, `reports/training-diagnose-after/` |

Năm tối ưu Python đã được báo cáo: counters, cached load, conservation debug-only, waiting-only hot list, IntEnum.

| Đo 2048 transitions, 4 env, CPU | Trước | After |
|---|---:|---:|
| learn | 21.71 s | 7.69 s |
| rollout collect | 21.42 s | 7.35 s |
| PPO update | 0.28 s | 0.34 s |
| decisions/s | 94 | 266 |
| eval 10 ngày | 8.63 s | 3.46 s |

**Learn được đo khi cProfile và nested timers đang bật** trong `training/diagnose.py`; eval diễn ra sau khi tắt cProfile, nhưng timers vẫn bật. Khoảng 15 phút/245,760 transitions là ngoại suy learn có profiler, không phải full-seed runtime đã đo. Không dùng 266 decisions/s làm mẫu số cho acceptance production.

Trong after, `env.step` tổng 6.04 s, observe khoảng 2.57 s, engine ticks 2.48 s; các timer có quan hệ cha/con, không cộng chồng. `action_masks` của env được gọi từ collector và step guard. `valid_action_mask` còn có caller khác; kiểm tra toàn bộ callers khi bỏ tính lặp.

Không được coi 95.6% rollout là Rust-accelerable: rollout còn policy inference, distribution, buffer, VecEnv và callback. Ước tính từ step + mask ngoài step cho phần có thể port khoảng 6.49/7.69 ≈ 84%; phần còn lại khoảng 1.20 s. Với giả định phần còn lại không đổi:

`S = 7.69 / (1.20 + 6.49 / K)` với K là speedup phần được port.

K=10 → S≈4.2; K=20 → S≈5.0; K→∞ → S≈6.4. Đây là mô hình trên profile, không phải cam kết benchmark. Bỏ dự báo cũ trần 23× và cam kết 4–8×. **Mục tiêu stretch là ≥4× end-to-end; gate bắt buộc ở §7 là ≥2×**, đo lại không profiler.

## 3. Phạm vi và kiến trúc

### 3.1 Rust và Python

Rust sở hữu domain state, engine/ticks, passengers, vehicles, dispatcher, travel, cost components/reward, observation và mask. Python giữ CLI, configs, JSON/NPZ, SB3 MaskablePPO, feature extractor, forecast, baselines, thống kê và plots.

Một lần gọi native `step` thực hiện action, toàn bộ ticks của control interval, observation, reward và mask mới. Số ticks lấy từ config, không hardcode 4. Không port PPO/GAE/PyTorch; không đổi Discrete(221), tensor schema, reward, physics hoặc guards trong migration.

Layout đề xuất:

```text
crates/bus-sim/       # domain, engine, passengers, vehicles, dispatcher,
                     # travel, costs, observe, masks; rlib
crates/bus-sim-py/    # PyO3 cdylib, Python import bus_sim
```

Có thể gộp rlib + cdylib nếu đơn giản hơn. Build release cho mọi benchmark. Backend config đề xuất: `runtime.backend = "python" | "rust"`; chưa phải option CLI hiện có. Truyền lựa chọn này xuyên suốt train, validation callback, diagnose, evaluate và baselines; không silent fallback nếu người dùng chọn Rust nhưng extension thiếu.

### 3.2 State và dữ liệu

- Vehicle lưu Vec/array theo ID; cohort arena và bucket waiting theo `(route, direction, origin)`; trên xe lưu CohortId. Giữ thứ tự duyệt/boarding/tie-break giống oracle, không để thứ tự hash map quyết định physics.
- Load và passenger totals là fields cập nhật ở mọi transition/split/alight/abandon. Chọn kiểu số theo giới hạn config, kiểm tra overflow; không mặc định mọi tổng đều vừa u16.
- Giữ lineage/cohort identity và thông tin finished cần cho summary/trace. Không loại bỏ dữ liệu chỉ vì không còn trong hot list.
- Pack network, tapes, traffic, flags và weights một lần; reset chỉ chọn scenario và tạo episode state. Native phải sở hữu dữ liệu hoặc giữ owner Python hợp lệ suốt lifetime; không giữ borrowed NumPy slice hết hạn. Không đọc file mỗi reset, không convert từng scalar qua FFI.
- SoA/arena là lựa chọn triển khai nhằm giảm scan/alloc; ưu tiên parity trước tối ưu layout phức tạp.

### 3.3 Tick semantics

Theo `sim/engine.py`, action áp dụng trước vòng ticks. Mỗi tick: complete phase (có thể board) → arrivals → abandon → board terminal-idle → dispatch-ready → integrate costs → giảm timers → tăng clock → conservation khi bật → event log → terminal unfinished settlement đúng một lần. Bảo toàn rounding travel, ngưỡng thời gian, mission changes và thứ tự xe/cohort.

Nếu code oracle và spec nền mâu thuẫn, ghi thành discrepancy có reproducer; không âm thầm sửa physics trong port. Chốt phạm vi oracle trước acceptance và tách mọi sửa semantics sang revision riêng.

## 4. Observation và mask: tối ưu nhưng không đổi thông tin

### 4.1 Không bỏ finished trực tiếp khỏi observe

`env/observation.py` đang dùng `iter_cohorts()` gồm finished để tính arrival history 5 khoảng, arrivals khoảng vừa qua và completions khoảng vừa qua. Chỉ duyệt waiting + onboard sẽ làm sai observation dù shape không đổi.

Rust dùng thống kê incremental/ring buffer để thay scan, theo đúng contract oracle:

| Feature | Nguồn và semantics cần giữ |
|---|---|
| Queue, mean/max age, excessive wait | Waiting hiện tại; thresholds và normalization giữ nguyên |
| Arrival history 5 khoảng | Tất cả arrivals đã xuất hiện trước observation time, kể cả khách đã finished; split không nhân đôi mass |
| Arrivals khoảng vừa qua | Cửa sổ `[t-control_interval, t)`, mỗi người tính một lần |
| Boarded channel | Theo code hiện tại: chỉ cohort **đang ONBOARD** có boarding time trong cửa sổ; không thay bằng tổng boarding events |
| Completed channel | Cohort COMPLETED có completion time trong cửa sổ, đúng count sau split |

Update bucket khi transition làm thay đổi feature; không chỉ đếm event rồi giả định tương đương. Không đọc future arrival tape để tạo observation/forecast. Kiểm tra ranh giới t=0, đúng biên cửa sổ, hoàn thành ngay sau board, split nhiều lần, abandon và cuối episode.

Giữ đầy đủ keys, float32 output, shapes, scaling, valid tensors, context và forecast flag. Tích lũy nội bộ với precision phù hợp rồi cast; chốt tolerance trước đối chiếu, không nới tolerance để che lỗi. Nếu muốn đổi nghĩa boarded channel, đó là thí nghiệm observation riêng ở L2, không thuộc Rust port.

### 4.2 Mask cache

Reset tạo mask cho state đầu tiên. Step kiểm tra action bằng mask của **state hiện tại**, áp dụng action/ticks rồi tạo mask của **state mới**. `action_masks()` đọc cache không tính lại. Mọi đường mutation state trong debug phải invalidate/recompute cache; không cho caller sửa cache native qua writable view.

Giữ NOOP valid, mapping 221 slots, flags M1/M2/M3 và tất cả guards. Action ngoài range hoặc masked-out phải bị reject trước mutation. Rà các lần validation bên dispatcher để bỏ tính lặp có bằng chứng, giữ guard cho API độc lập nếu vẫn public.

## 5. FFI, wrapper và evaluator

Contract đề xuất, chưa phải API đã implement:

```text
reset(scenario_index) -> (obs, mask)
step(action_index) -> (obs, reward, terminated, truncated, mask, cost_tuple)
episode_summary_inputs() -> terminal snapshot đủ để tính mọi metric hiện có
trace_snapshot() -> state/event snapshot khi bật trace
debug_snapshot() -> state/counters phục vụ golden tests
```

Gym wrapper giữ `super().reset(seed=seed)`, RNG chọn scenario và `options.scenario_index` theo oracle; gọi reset native, cập nhật cache mask rồi trả `(obs, {})`. Step trả tuple Gym 5 phần, chuyển `cost_tuple` thành `StepCosts` đúng tên/thứ tự/đơn vị cho evaluator. Hết horizon hiện là terminated=True, truncated=False; không tự đổi thành time-limit truncation. Terminal observation và auto-reset của VecEnv phải đúng.

Không dựng Python WorldState mỗi step train. Dict NumPy ở boundary được phép; tránh graph Python và allocation không cần thiết bên trong ticks. Output obs/mask không bị ghi đè khi step/reset tiếp theo chạy: trả owned arrays/copies hoặc cơ chế ownership có kiểm thử tương đương. Không tiết kiệm copy bằng cách làm hỏng rollout buffer/terminal observation.

Forecast vẫn Python: predict từ obs hiện tại và time hiện tại, thay forecast tensor và context flag giống wrapper hiện tại; reset cache forecast đúng episode. Không dùng state đặc quyền/future tape.

`evaluation/runner.py` hiện đọc `env.state`, cohorts, vehicles và logs để tính wait/P95/censoring, per-route metrics, action counts/headways và traces. **Tích hợp evaluator là bắt buộc**, không giả định wrapper mỏng tự tương thích:

- Refactor qua summary/trace interface chung; Python backend có adapter cùng contract.
- Export terminal cohort/lineage data, departures/action logs và totals cần thiết một lần cuối episode; thống kê/plots vẫn ở Python. Đối chiếu toàn bộ metrics, không chỉ cost.
- Trace snapshot chỉ bật khi yêu cầu; benchmark hai backend cùng chế độ trace.
- Baselines và PPO dùng cùng env, sensors, mask, metrics. Validation callback phải dùng backend đã chọn.

## 6. R0–R4: thứ tự triển khai

| Stage | Công việc và đầu ra |
|---|---|
| R0 — Oracle | Khóa Python after revision, dependency lock, manifests/scenario order/config/seeds; tạo golden tick/step, fixed checkpoint eval và benchmark Python không profiler. Không cần tối ưu Python thêm. |
| R1 — Kernel | Domain, lifecycle/split, action families, travel, cost từng component, terminal settlement; đối chiếu từng tick, conservation bật trong tests. |
| R2 — Obs/mask | Ring buffers/statistics §4, full obs, masks, reset semantics; golden và tests chống alias/stale cache. |
| R3 — Integration | PyO3 release, backend selection, wrapper, StepCosts, evaluator snapshots/trace, forecast, provenance và checkpoint save/load. |
| R4 — Acceptance | Chạy toàn bộ gates §7, ghi artifacts; đạt rồi đóng băng Rust backend revision để bắt đầu L0. |

Golden suite phải có zero/normal/peak/burst/traffic, nhiều scenario seeds, M1/M2/M3, mọi action family, valid/invalid action, donor/cooldown, partial boarding/capacity, abandonment, short-turn và terminal settlement. Replay cùng action sequence trên hai backend để tách physics khỏi policy randomness. Test split kiểm tra conservation/lineage và mọi channel observation.

So counters, status, IDs và masks exact; float obs mặc định `rtol=1e-6, atol=1e-6`, cost/reward `rtol=1e-9, atol=1e-9`. Mọi sai biệt ngoài tolerance phải điều tra trước gate. Với aggregate metrics, dùng tolerance cost/float tương ứng và đối chiếu categorical/None exact.

Fixed checkpoint eval dùng cùng checkpoint hash, ngày và deterministic actions trên hai backend. Mean cost cũ **15471.25** là mốc regression của trial cũ, chỉ bắt buộc khi phục hồi đúng checkpoint/config/days đó. Nếu checkpoint không còn, tạo oracle checkpoint mới trên Python và ghi provenance; không giả lập mốc cũ. Mean bằng nhau không thay thế per-step/per-day parity.

Train smoke 2048 transitions xác minh vận hành/save-load/finiteness; không ép train mới phải ra đúng 15471.25 hoặc weights bitwise identical. Sai lệch policy lớn dù golden đạt vẫn phải điều tra seed/order/numerics trước R4.

## 7. Acceptance và benchmark protocol (nguồn chuẩn dùng chung)

### 7.1 Protocol

1. Cùng máy, CPU device, Torch threads, dependency versions, manifests, scenario order/seeds, trace/logging, conservation tắt trên cả hai. Rust release; không compile trong thời gian đo.
2. Tắt cProfile và TIMERS cho speed acceptance. Diagnose có profiler chạy riêng để giải thích hotspots.
3. Warm-up riêng, sau đó ít nhất 5 lần đo mỗi backend, chạy xen kẽ để giảm ảnh hưởng nhiệt/tải. Ghi từng lần, median và min/max; nếu biến động lớn thì điều tra/rerun trước kết luận.
4. Benchmark simulation-only trên cùng legal action traces, reset/episode count, normal/peak/OOD. Báo decisions/s, ticks/s và wall time; không gộp setup/pack với steady-state.
5. Benchmark learn ít nhất **12,288 transitions** mỗi lần với core config: 4 env, n_steps=256, batch_size=256, n_epochs=4, cùng seed pair, cùng model ban đầu. Tắt periodic eval cho phép đo learn riêng; giữ các setting thuật toán khác. Báo transitions thực tế, learn wall và throughput.
6. Benchmark fixed-checkpoint eval cùng bộ ngày; sau đó một full seed 245,760 transitions có validation schedule giống nhau trên mỗi backend để đo wall thực tế (gồm validation/checkpoint). Đây là kiểm định backend, không tuning thuật toán.
7. Ghi setup/packing, steady-state, eval, total train, peak RSS, compiler/build flags và revision. Không so Rust không profiler với Python có profiler; không lấy native-only speedup làm learn speedup.

### 7.2 Gates bắt buộc

| Gate | Tiêu chí |
|---|---|
| Physics | Golden suite R0–R2 đạt, conservation/lineage/terminal settlement đúng |
| Obs/mask | Exact schema/mask, numerical tolerance §6, history/completion parity, reset và output ownership đúng |
| Integration | Gym/VecEnv, forecast, every metric/trace, baseline, validation backend, checkpoint save/load đạt |
| Learn smoke | 2048 transitions chạy được, obs/reward/loss finite, không masked invalid action |
| Simulation speed | Median simulation wall tổng bộ traces ≤50% Python after; báo riêng từng workload để lộ regressions |
| Learn speed | Median learn wall ≤50% Python after theo §7.1, tức **≥2×**; stretch **≥4×**, không bắt buộc |
| Full workflow | Full seed + validation thực đo nhanh hơn Python; giải thích mọi eval/RSS regression trước rollout |
| Reproducibility | Artifacts đủ chạy lại; metadata backend/build và oracle hashes đầy đủ |

Nếu chưa đạt, tiếp tục profile/tối ưu Rust trong R4; giữ Python làm oracle/fallback và chưa mở L0 tuning. Không âm thầm giảm gate hay đánh dấu xong. Gate này là quyết định kỹ thuật cho kế hoạch Rust-first, không phải tốc độ đã đạt.

## 8. Artifacts, lineage và chuyển sang RL

Artifacts mới đề xuất:

- `reports/rust-migration.md`: parity matrix, discrepancies, benchmark protocol/results, R0–R4 checklist, decision.
- `reports/rust-migration/`: oracle manifest/hashes, compact golden summaries, per-run benchmark JSON/CSV, fixed-checkpoint paired metrics. Traces/checkpoints lớn để `runs/rust-migration/`, link/hash trong report.
- Giữ nguyên toàn bộ baseline và after reports cũ. Không tạo số liệu đo trước khi chạy.

Ghi backend, native version/build/revision, Python revision/dirty, dependency lock, seed/scenario order và physical/control/reward/obs/action hashes. Backend khác không tự đồng nghĩa physics khác: chỉ load checkpoint xuyên backend khi contract hashes phù hợp và backend parity đã được xác nhận; không disable kiểm tra provenance.

Sau R4, **Rust là backend của loạt thực nghiệm mới** ở [improve_RL.md](improve_RL.md); Python tiếp tục là reference/fallback. Đóng băng revision trong một đợt so sánh. Chưa parallelize native envs/SubprocVecEnv trong migration; nếu làm sau, đó là benchmark riêng và phải kiểm soát rollout size `n_envs × n_steps`.

## 9. Nguồn code cần đối chiếu khi triển khai

- `src/bus_rl/domain.py`, `sim/engine.py`, `sim/passengers.py`, `sim/vehicles.py`, `sim/dispatcher.py`, `sim/travel.py`.
- `src/bus_rl/env/observation.py`, `env/bus_dispatch.py`, `control/actions.py`, `control/guards.py`, `rewards/costs.py`.
- `src/bus_rl/training/diagnose.py`, `train.py`, `callbacks.py`, `checkpoint.py`, `evaluation/runner.py`.
- `tests/test_engine.py`, `test_passengers.py`, `test_vehicles.py`, `test_control.py`, `test_env.py`, `test_rewards.py`, `test_forecast.py`, `test_pipeline.py`.
