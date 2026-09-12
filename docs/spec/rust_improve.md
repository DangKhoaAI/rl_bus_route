# Rust kernel: hybrid speedup after Python hot-path opts

Ngày: **2026-09-12**. Trạng thái: đề xuất kỹ thuật; **chưa implement**.

Python T1–T11 đã xong. Năm tối ưu hot-path (counters, cache `load`, tắt conservation khi train, waiting-only list, `IntEnum`) đã đo. Tài liệu này ghi **khi nào** và **cách nào** chuyển kernel sang Rust — không thay PPO, không rewrite toàn bộ `bus_rl`.

Spec vật lý / action / observation: [spec.md](../spec.md). Plan Python: [plan.md](../plan.md).

Baseline đo:

| Vai trò | Path |
|---|---|
| Trước optimize Python | `reports/training-diagnosis.md`, `reports/training-diagnose/` |
| Sau optimize Python | `reports/training-diagnosis-after.md`, `reports/training-diagnose-after/` |

Eval mean cost hai lần diagnose **không đổi**: `15471.25`. Mọi port Rust phải giữ số này trên cùng 10 ngày / cùng seed.

---

## 1. Kết luận đo được

### 1.1 Python opts đã ăn phần dễ

Cùng lệnh `bus-rl diagnose` (2048 steps, 10-day eval, `n_envs=4`, CPU):

| | Trước | Sau | |
|---|---:|---:|---|
| `learn()` | 21.71 s | **7.69 s** | **×2.82** |
| decisions/s | 94 | **266** | |
| seed 245,760 steps | ~43 phút | **~15 phút** | |
| eval 10 ngày | 8.63 s | 3.46 s | ×2.49 |
| `ppo_update` | 0.28 s | 0.34 s | không đổi |
| `engine.ticks` (2 rollout) | 18.0 s | 2.48 s | ×7.3 |
| conservation trên train | 1.72 s | ~0 | |

Sim vật lý không còn là #1. `board` / `tick_costs` / `_count` / `StrEnum` đã hết nóng.

### 1.2 Bottleneck *sau* Python opts

`env.step` ~3.0 ms (trước ~10 ms):

```text
env.step ~3.0 ms
  observe          1.25 ms   #1  (cProfile tottime 2.17 s / 7.64 s)
  advance_interval 1.40 ms
    ticks          1.21 ms   (arrivals, phase, dispatch, board)
  action_masks     0.22 ms × 2 lần / step
ppo_update                   0.34 s / 7.69 s = 4.4%
```

`observe()` còn **chậm hơn trước** (0.72 → 1.25 ms) vì vẫn `iter_cohorts()` gồm `finished`. MaskablePPO + `BusDispatchEnv.step` tính mask hai lần.

PPO/torch không phải mục tiêu port (`torch.nn.linear` 0.16 s). GPU không cứu rollout.

### 1.3 Rust còn nhanh hơn không

Có. Amdahl trên bản *after*: sim+obs+mask ≈ 95.6%, PPO ≈ 4.4% → trần ~×23 nếu env Python = 0.

Kỳ vọng hybrid PyO3 tử tế: **×4–8 nữa** trên bản after (seed ~15 phút → ~2–4 phút). Cộng ×2.82 Python: **~×10–20 vs baseline gốc**.

Điều kiện: port **cả vòng `step`** (tick + observe + mask) trong **một lần FFI**. Port mỗi `sim/` rồi để Python `observe` thì `observe` vẫn 1.25 ms — lãi gần như chết.

Chưa làm Rust nếu 15 phút/seed chấp nhận được, hoặc chỉ cần thêm ×1.2–1.5: sửa `observe` không walk `finished`, cache mask trong `step`.

---

## 2. Phạm vi

### 2.1 Trong Rust (`bus_sim`)

- `domain`: `Phase` / `Pattern` / `PassengerStatus` dạng `u8`, `Vehicle`, `Cohort`, `World`
- `sim`: `advance_tick`, `advance_interval`, passengers, vehicles, dispatcher, travel
- `rewards/costs`: `integrate_tick_costs`, `interval_cost` (cùng trọng số Python)
- `env/observation`: điền ndarray cố định, không alloc dict Python trên hot path
- `control/guards`: `valid_action_mask` 221 slot
- PyO3: class env, **một** `step(action) -> (obs, reward, done, mask, costs)`

Thứ tự sự kiện tick **bắt buộc giống Python** (spec §4.1): complete phase → arrivals → abandon → board → dispatch → costs / timers.

### 2.2 Ở lại Python

CLI, provenance, SB3 / MaskablePPO, forecast, pandas / plot, `data/io` (JSON+NPZ), baselines, configs TOML, report T11.

Baseline và PPO gọi cùng env wrapper. Không viết trainer Rust.

### 2.3 Ngoài phạm vi

- Full rewrite `bus_rl`
- Gymnasium / PPO / GAE bằng Rust
- Đổi schema obs/action (vẫn Discrete 221, shape cố định spec)
- Dual physics không có flag so sánh
- `SubprocVecEnv` *cộng* Rust trước khi crate khớp golden

---

## 3. Kiến trúc

```text
Python                         Rust
──────                         ────
data/io, TOML, NPZ
  pack scenario 1 lần  ----->  Scenario tapes (&[i32], traffic)
SB3 MaskablePPO
  BusDispatchEnv  ---------->  bus_sim::Env
    reset(i)                     reset
    step(a)                      step: 4 ticks + observe + mask + reward
    action_masks()               mask cache từ step vừa rồi
forecast, eval, plots          (không biết)
```

Feature flag: `BUS_RL_BACKEND=python|rust` (hoặc `config.backend`). Diagnose / eval phải chạy được cả hai trên cùng manifest.

### 3.1 Layout crate

```text
crates/bus-sim/
  Cargo.toml
  src/lib.rs
  src/domain.rs
  src/engine.rs
  src/passengers.rs
  src/vehicles.rs
  src/dispatcher.rs
  src/travel.rs
  src/costs.rs
  src/observe.rs
  src/masks.rs
crates/bus-sim-py/          # PyO3 cdylib, import Python: bus_sim
  Cargo.toml
  src/lib.rs
```

Có thể gộp một crate `cdylib` + `rlib` nếu muốn đơn giản. Không bắt buộc workspace lớn.

### 3.2 Data layout (quan trọng hơn “viết Rust”)

Python after vẫn là object graph. Rust phải SoA / arena:

| Python | Rust |
|---|---|
| `vehicles: dict[int, Vehicle]` | `[Vehicle; F_max]` hoặc `Vec<Vehicle>` index = id |
| `cohorts` waiting list | bucket `(route, dir, origin)` + arena |
| `bus.passengers: list` | `Vec<CohortId>` trên xe |
| `state.finished` | vec riêng; **không** vào `observe` |
| `Vehicle.load` property | field `u16`, cập nhật khi board/alight |
| counters waiting/onboard/… | field, ± khi đổi status |
| `arrival_tape` NumPy | `&[i32]` shape `(T, R, 2, S, S)` |
| `StrEnum` / `IntEnum` | `#[repr(u8)] enum` |

Không `HashMap` trên hot tick. Không alloc `PassengerCohort` Python-style mỗi split nếu tái sử dụng slot arena.

### 3.3 FFI — một hàm / decision

Không lộ `WorldState` ra Python khi train. Không `for _ in range(4): rust.tick()`.

```text
reset(scenario_id) -> obs
step(action: u16) -> (obs, reward, done, mask, cost_tuple)
```

`obs` = các `numpy.ndarray` sẵn, dtype `float32`, shape spec (stops `(4,2,8,7)`, vehicles `(16,27)`, …). Mask = `bool[221]` hoặc `uint8[221]`.

`action_masks()` của Gym trả **cache** từ `step`/`reset` vừa rồi. Không tính mask lần hai trong `step` (bug đo được trên Python after).

`CONSERVATION_CHECKS`: bật trong `cargo test` / pytest backend rust; **tắt** khi `diagnose`/`train` giống Python after.

Pack scenario **một lần** lúc `Env.__init__`: network, tapes, traffic, flags M1/M2/M3, reward weights. Rust không đọc JSON mỗi `reset`.

---

## 4. Python wrapper

`BusDispatchEnv` mỏng; không nhánh logic vật lý:

```python
class BusDispatchEnv(gym.Env):
    def __init__(self, scenarios, config, ...):
        # pack tapes/network/reward/control -> bus_sim.Env
        self._core = bus_sim.Env(...)
    def reset(self, *, seed=None, options=None):
        index = ...
        return self._core.reset(index), {}
    def step(self, action_index):
        obs, reward, done, mask, costs = self._core.step(int(action_index))
        self._mask = mask
        return obs, reward, done, False, {"costs": costs}
    def action_masks(self):
        return self._mask
```

Observation space / action space giữ nguyên Gymnasium. Forecast: nếu bật, Python gắn tensor `forecast` **sau** `obs` Rust (hiếm trên hot train core; không nhét vào crate trừ khi profile bắt buộc).

---

## 5. Thứ tự implement

Không port cả đống rồi so 1 scalar reward.

1. **Oracle (bắt buộc trước code Rust)**  
   Python after, cùng seed/scenario: dump mỗi control step `{action, obs, mask, reward, costs, counters}`.  
   Gate: eval 10 ngày = `15471.25`.

2. **`domain` + `advance_tick` NOOP** (không action) vs golden từng tick.

3. **passengers / vehicles / dispatcher** + action M3 (`DISPATCH`/`RECALL`/`REASSIGN`/`SHORT_TURN`/`SET_HEADWAY`/`NOOP`).

4. **costs** — từng component (`waiting_pm`, `abandoned_count`, …), không chỉ tổng.

5. **`observe` + `mask`** — `np.testing.assert_allclose` / so bit mask.

6. **PyO3 `reset`/`step`** + wrapper Gym.

7. **`bus-rl diagnose`** 2048 steps, backend rust: cost `15471.25`, ghi fps vào report mới (không đè `training-diagnose-after/`).

8. Train lại seed T9 **chỉ khi** cần số liệu trên backend mới. Artifact lineage (schema hash, physical hash) không được im lặng đổi physics.

Mỗi bước có test fail-then-implement. Lệch golden = bug; không “xấp xỉ”.

---

## 6. Gate chấp nhận

| Gate | Tiêu chí |
|---|---|
| Physics | Cùng scenario/seed: conservation (khi bật), event counts, cost components |
| Obs/mask | Shape + dtype; mask 221 khớp Python; obs `allclose` rtol/atol chặt |
| Eval | 10 ngày diagnose: mean cost **15471.25** |
| Train smoke | 2048 steps MaskablePPO CPU, save/load checkpoint như T5 |
| Speed | Diagnose fps **≥ 4×** bản after (~266 decisions/s → ≥ ~1000) trên cùng máy; nếu không đạt, không gọi port “xong” |
| Lineage | `BUS_RL_BACKEND` ghi vào run metadata; load checkpoint không trộn nhầm physics |

Không đạt speed gate thì giữ Python after làm runtime; crate có thể ở `dev` nhưng không thay default train.

---

## 7. Việc không làm / bẫy

- Viết lại MaskablePPO, GAE, feature extractor bằng Rust.
- Port pandas / matplotlib / CLI / IO.
- Giữ Python `observe` “tạm” sau khi port engine.
- Hai implementation không golden — ablation T9 chết.
- Gọi conservation mỗi tick trên train.
- `int(np.scalar)` từng ô tape (đúng bệnh `_add_arrivals` Python).
- Parallel `n_envs` trong Rust **trước** khi 1-env khớp oracle.

Rẻ hơn, vẫn Python after: `observe` chỉ waiting + onboard; một lần mask / step. Làm trước crate nếu chưa chắc cần ×4–8.

---

## 8. Quyết định mặc định

- Default sản phẩm REL / T9 đã chạy: **Python after**.
- Rust là **opt-in backend** khi 15 phút/seed vẫn chậm cho vòng lặp tiếp theo.
- Khi implement: kernel + observe + mask + một FFI `step`. Không “sim Rust, observation Python”.
