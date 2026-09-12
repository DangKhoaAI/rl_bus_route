# Tối ưu bộ nhớ hậu R4

Ngày cập nhật: **2026-09-12**. Trạng thái: **M1 và M3 đã làm; M2/M4/M5 là kế hoạch**.
Liên quan: [rust_improve.md](rust_improve.md), [plan/rust_improve.md](../plan/rust_improve.md),
[reports/rust-migration.md](../reports/rust-migration.md).

Tài liệu này chỉ bàn **bộ nhớ**. Tốc độ đã đạt gate R4 (Rust nhanh ~3.7×) và
không nằm trong phạm vi ở đây.

## 1. Bối cảnh và số đo hiện tại

Full seed 245,760 transitions (`core-threads2.toml`, seed 11, 100 validation
days), peak RSS:

| Thành phần | Dung lượng | Ghi chú |
|---|---:|---|
| `Scenario` phía Python × 600 | ~244 MB | **arrival tape 198 MB** + network/fleet/traffic/digest ~46 MB |
| torch + sb3 + policy + rollout buffer | ~390 MB | giống nhau ở cả hai backend |
| native `ScenarioStore` (Rust) | 28.8 MB | sparse, chỉ Rust |
| **Tổng Python / Rust** | **662 / 690 MB** | tỉ lệ 1.04× |

Hai lỗi lưu trữ đã được sửa trước tài liệu này (xem `reports/rust-migration.md`
mục R4.3): wrapper tạo sẵn một `Kernel` cho mỗi scenario (~243 KB scratch mỗi
kernel) và store giữ arrival tape dạng dày. Kết quả: Rust từ 1236 MB xuống
690 MB, training vẫn bit-identical.

Điểm mấu chốt: **~244 MB scenario là dữ liệu dùng chung cho cả hai backend**
(`load_split` nạp sẵn mọi scenario). Vì vậy tối ưu chỉ phía Rust không thể đưa
Rust xuống dưới Python; muốn vậy phải giảm phần dùng chung, hoặc thôi giữ nó
trong run Rust.

Arrival tape dày `(480, 3, 2, 6, 6) int32` = **405 KB/scenario** nhưng chỉ
**1.46% ô khác 0** (1,509 / 103,680), giá trị lớn nhất đo được là **5**.

## 2. Mục tiêu và ràng buộc

- **Bất biến hành vi:** physics, reward, observation, mask, `scenario_hash`,
  golden fixtures và oracle manifest phải giữ nguyên. Mọi thay đổi biểu diễn
  dữ liệu phải chứng minh output giống hệt (`export_oracle.py verify` + full
  pytest + full seed bit-identical).
- **Giữ `scenario_hash`:** `scenario_digest` chuẩn hoá tape về `int32` trước khi
  băm, nên đổi dtype/biểu diễn không làm đổi hash và không phá fixture đã commit.
- **Ưu tiên Rust:** chỉ nhận tối ưu chung nếu Rust **không mất tốc độ**.
- Không đổi `docs/spec/spec_v1.0.md` và không mở lại R0–R4.

## 3. Các hạng mục

### M1 — Tape dày dtype hẹp (int8) — ĐÃ LÀM

Vì giá trị lớn nhất là 5 và biến thể OOD lớn nhất là `flood` (`×20` → 100) và
`capacity` (45), `int8` (≤127) là đủ. Việc này giảm tape 198 MB → ~49 MB ở **cả
hai** backend, Rust không mất gì.

Thay đổi:
- `data/scenario.py::generate_scenario` sinh tape `int8`; kiểm tra biên sau
  `_apply_burst`.
- `domain.py::scenario_digest` băm `ascontiguousarray(arrivals, dtype=int32)`
  để giữ nguyên `scenario_hash`.
- `data/io.py::load_scenario` ép `int8` khi nạp (dữ liệu cũ int32 vẫn đọc được).
- `parity/scenarios.py::_with_arrivals` ép `int8` + kiểm tra dải giá trị.
- `crates/bus-sim-py` nhận `PyReadonlyArray5<i8>` và nâng lên `i32` khi pack
  (store vẫn sparse như cũ), nên lõi kernel không đổi.

**Acceptance:** `scenario_hash` không đổi; `verify` tái tạo 11 fixture;
`pytest` xanh; full seed bit-identical; peak RSS giảm ~145 MB ở cả hai backend.

### M2 — Biểu diễn arrival thưa (canonical) — KẾ HOẠCH

Đưa cấu trúc thưa thành dạng chuẩn ở phía Python
(`SparseTape`: offsets theo tick + mảng `(route, dir, origin, dest, count)`),
để tape 198 MB → ~12 MB. Đây là bản "đúng" của M1 và còn làm `engine._add_arrivals`
của Python nhanh hơn (hiện vẫn duyệt cả lưới 216 ô/tick như Rust trước đây).

Ảnh hưởng: `generate_scenario`, `data/io.py`, `sim/engine.py`,
`parity/scenarios.py` (các biến thể `zeros/×3/×20/set-cell` phải thành thưa),
`forecasting` (nếu đọc tape), payload native. `scenario_digest` vẫn chuẩn hoá
`int32` nên hash không đổi.

**Ước tính:** cùng hướng với M1 nhưng chạm nhiều call site hơn; trung bình.
Làm sau M1 và chỉ khi cần thêm ~37 MB.

### M3 — Run Rust không giữ `Scenario` Python — ĐÃ LÀM (tóm tắt ở mục 5)

Đây là hướng duy nhất đưa **Rust thấp hơn Python**, và đã đạt: xem mục 5.

### M4 — Bỏ bản sao store native — MỘT PHẦN

Store đã sparse (28.8 MB). Phần còn lại: `Kernel.from_store` giữ `Arc<Scenario>`
trên tape đã pack; muốn bỏ hẳn cần build kernel trực tiếp từ `Scenario` mỗi
`reset` (mất tốc độ vì pack lại) hoặc zero-copy mượn mảng Python. Hiện 28.8 MB
là đánh đổi hợp lý; **không làm** trừ khi M3 yêu cầu.

### M5 — Nạp scenario lười / mmap — KẾ HOẠCH

`load_split` nạp sẵn cả 600 scenario. Có thể nạp lười theo `(split, index)` với
LRU, hoặc đổi `tapes.npz` sang `.npy` để `mmap_mode='r'` (page cache thay vì
anonymous RSS). Full seed chạm gần hết 600 scenario nên LRU ít lợi; mmap có thể
giảm áp lực bộ nhớ nhưng `/usr/bin/time maxrss` vẫn tính trang đã chạm. Xếp sau
M2/M3.

## 4. Acceptance và verification

Mọi hạng mục phải thoả:
- `python scripts/export_oracle.py verify` OK (11 fixture, hashes, reference
  15471.25).
- `python -m pytest -q` xanh (kể cả differential và full-seed parity).
- Full seed Rust bit-identical với Python (curve 20 điểm, weights, optimizer).
- `scenario_hash` của mọi scenario không đổi sau thay đổi biểu diễn.
- Đo lại peak RSS và ghi vào `reports/rust-migration/full-workflow-memory.json`
  và mục R4.3 của report.

## 5. Kết quả

### M1 — đã làm (2026-09-12)

- Tape `int8`: `(480,3,2,6,6)` = **101 KB/scenario** (trước 405 KB).
- 600 scenario phía Python: **249 MB → 67 MB**.
- Native store (sparse): 30.1 MB (~50 KB/scenario).
- Peak RSS full seed: Python **479 MB**, Rust **508 MB** (tỉ lệ 1.06×; trước khi
  tối ưu bộ nhớ là 662 / 1236 MB). Rust 1236 → 508 MB.
- `scenario_hash` không đổi (digest chuẩn hoá int32), `verify` OK, 149 test xanh,
  full seed bit-identical (curve 20 điểm, weights, optimizer).
- Tốc độ không đổi: Python 595.35 s vs Rust 161.61 s (3.68×).

Chi tiết số đo: `reports/rust-migration/full-workflow.json` và
`full-workflow-memory.json`; mục R4.3 của `reports/rust-migration.md`.

### M3 — Run Rust không giữ `Scenario` Python — ĐÃ LÀM

Rust run trước đây nạp cả 600 `Scenario` (kèm tape) vào Python chỉ để đưa cho
store native. Đã thêm đường nạp metadata-only:
- `load_split(..., with_tapes=False)` trả `Scenario` rỗng tape + `path` (chỉ đọc
  `scenario.json`); `Scenario` có thêm field `path`.
- `NativeScenarioStore` đọc `tapes.npz` từng scenario một lần, kiểm tra
  `scenario_digest`, rồi chỉ giữ bản sparse; `_scenario_tapes` dùng chung cho
  `build_kernel` và store.
- CLI (`train`/`diagnose`/`baseline`) dùng đường metadata cho run Rust khi không
  bật forecast; `evaluate` và forecast vẫn nạp đầy đủ.

Kết quả full seed: Rust **445 MB** < Python **479 MB** (0.93×). Rust 508 → 445 MB;
bit-identical (curve, weights, optimizer). Revision native không đổi
(`aedc4faa4c43`, chỉ đổi phía Python).

### M2, M4, M5 — kế hoạch

- **M2** (thưa canonical) trùng effort M1 và chỉ còn ~50 MB phía Python (Rust
  không được thêm) → để sau.
- **M4** (bỏ store dedup 30 MB) mất tốc độ → không làm.
- **M5** (lazy/mmap) ít lợi vì full seed chạm gần hết scenario → để sau.
