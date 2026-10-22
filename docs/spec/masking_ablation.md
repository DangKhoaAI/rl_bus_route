# A/B action masking: apply và free

Ngày: **2026-10-22**. Trạng thái: **thiết kế đã chốt, triển khai sau**.

Phụ thuộc [masked_algorithm_comparison.md](masked_algorithm_comparison.md).
Spec trước phải hoàn thành pipeline REINFORCE/A2C/PPO có mask trước khi
bắt đầu phần này. Tài liệu này bổ sung trục mask mode cho các trainer đó.

## 1. Câu hỏi thí nghiệm và quyết định thiết kế

Với cùng một thuật toán, cùng dữ liệu, mạng, reward và ngân sách tương tác,
việc loại action không hợp lệ khỏi phân phối có giúp học và điều phối
tốt hơn không?

Thêm một flag:

```text
--mask-mode {apply,free}
```

| Mode | Phân phối actor | Action sai trong phạm vi 221 slot |
|---|---|---|
| apply | Xác suất chỉ trên action hợp lệ | Collector kiểm tra không được chọn sai |
| free | Xác suất trên toàn bộ 221 action | Thực thi thành NOOP, vẫn tiến đủ một interval |

**Quy tắc đã chọn: không thêm penalty cho invalid action.** Action sai làm
lỡ một cơ hội can thiệp; hệ thống vẫn vận hành 2 phút, sinh chi phí như
một NOOP trong cùng state. Agent học từ hậu quả vận hành đó.

Tắt mask chỉ tắt ràng buộc phân phối policy. Giới hạn sức chứa, donor guard,
cooldown và tính khả thi của simulator vẫn được thực thi. Xe đang có khách
không được chuyển tuyến bởi một invalid request.

## 2. Hợp đồng thực thi action

Gọi `a_requested` là action actor chọn, `m_env` là mask khả thi của state
trước bước và `a_executed` là action thực sự gửi xuống kernel.

Trong execution adapter của study A/B:

```text
nếu a_requested không phải mã action hợp lệ thuộc [0, 220]:
    báo lỗi đầu vào, không mutate environment
nếu m_env[a_requested] == True:
    a_executed = a_requested
ngược lại:
    a_executed = NOOP (index 0)

chạy kernel.step(a_executed) đúng một control interval
reward = reward vận hành gốc của bước đó
```

Action trong slot padding của xe/tuyến không tồn tại cũng là invalid và
được map NOOP. Mã ngoài action space, kiểu dữ liệu sai hoặc batch shape
sai là lỗi lập trình; không map chúng thành NOOP.

Một lần request luôn tiêu thụ một bước RL và cùng thời lượng mô phỏng.
Không resample đến khi hợp lệ, không bỏ qua bước, không trả reward 0 mà
không chạy simulator, không terminate sớm vì action sai.

Không cộng mission-change cost cho invalid request vì thực tế không có
mission change. Chi phí waiting/operating/abandonment và settlement cuối
ca vẫn được tính đầy đủ qua bước NOOP.

Nếu chỉ còn NOOP hợp lệ, apply chọn NOOP với xác suất 1. Free vẫn lấy mẫu
trong toàn bộ action space; request sai và NOOP thật cùng dẫn đến vận hành
NOOP, nhưng được log thành hai loại quyết định khác nhau.

## 3. Cùng execution adapter cho hai arm

Giữ Rust kernel và FFI strict như hiện tại: trực tiếp gửi action sai xuống
kernel vẫn phải bị từ chối. Mapping thuộc adapter Python, trước khi gọi
kernel. Không làm yếu guard hoặc sửa semantics action table.

Study A/B dùng cùng execution config cho cả apply và free:

```toml
[action_execution]
invalid_action = "noop"
```

Trong apply, collector kiểm tra sampled action hợp lệ trước adapter.
Nếu apply chọn sai, đó là lỗi và phải fail; adapter không được che một lỗi
masking bằng cách âm thầm sửa giúp collector.

Ngoài study A/B, default execution vẫn là `invalid_action="error"` như
trước. Checkpoint của spec thứ nhất và checkpoint lịch sử tiếp tục dùng
strict execution khi thiếu section action_execution.

Resolved config free + error bị từ chối trước khi train. Các preset A/B
đặt noop sẵn cho cả hai arm, nên người dùng chỉ cần đổi một flag mask-mode;
không cần thêm flag CLI riêng cho invalid handling.

Hai arm vì vậy có cùng môi trường sau khi mở rộng action space bằng
các request quy về NOOP. Khác biệt chủ động duy nhất là policy có loại
các request sai khỏi phân phối hay không.

## 4. Thu rollout và cập nhật policy

### 4.1 Phân biệt policy mask và feasibility mask

- `m_env` luôn là tính khả thi thật của state, dùng trong adapter/metrics.
- Apply dùng `m_policy=m_env` cho phân phối actor.
- Free dùng không mask hoặc một mask toàn True cho phân phối actor.

Trong free, feasibility mask không được dùng để sửa logits, chọn action,
chuẩn hóa xác suất hay tính entropy/loss. Observation vẫn giữ nguyên như
apply, bao gồm entity-valid fields hiện tại. Không thay đầu vào để tạo thêm
một yếu tố khác trong phép A/B.

Mask khả thi có thể được tính nội bộ để đảm bảo action execution và đo
invalid rate; việc đó không được làm thay đổi phân phối free.

### 4.2 Lưu action đã yêu cầu để học

Rollout giữ `a_requested` và log probability của action đó, kể cả khi
adapter thực thi NOOP. Reward/observation/terminal là kết quả thực tế
của bước đã thực thi.

```text
actor chọn REASSIGN không hợp lệ
adapter thực thi NOOP
buffer.actions = REASSIGN
buffer.log_prob = log pi_free(REASSIGN | observation)
buffer.reward = reward thực tế của bước NOOP
```

Nếu thay action trong buffer bằng NOOP, gradient sẽ chấm nhầm lựa chọn
của policy. Cấm mutate array actions gốc trong collector/callback khi
adapter tạo array action thực thi.

### 4.3 Quy tắc loss

REINFORCE dùng return đầy đủ từ các bước đã thực thi; tính log probability
của requested action trên đúng support theo mode.

A2C dùng cùng GAE/value target và loss của spec trước; chỉ thay support
của actor/entropy theo mode.

PPO free tính cả old/new log probability trên full support. Tử và mẫu của
ratio không được dùng hai mask mode khác nhau. Buffer có thể lưu mask toàn
True trong trường policy-mask, nhưng phải tách nó khỏi m_env dùng đo
tính khả thi.

MaskablePPO 2.9.0 đã có `learn(use_masking=False)` và
`predict(action_masks=None)`. Có thể dùng cùng class cho hai mode để giữ
PPO implementation chung. Phải kiểm cả train, validation và evaluate;
không chỉ tắt mask khi predict mà vẫn train có mask.

## 5. CLI, cấu hình và checkpoint

### 5.1 Giao diện

Thêm `algorithm.mask_mode="apply"` làm default; CLI `--mask-mode` cho
model train và test diagnose có default None ở parser để giữ ưu tiên
CLI truyền rõ → TOML → default.

Thêm ActionExecutionConfig vào RunConfig, default invalid_action=error.
Chỉ hỗ trợ error/noop trong spec này; penalty không phải tham số tunable.

Các lệnh đích dùng chung preset và chỉ đổi mode/output:

```bash
uv run bus-rl model train \
  --config configs/studies/masking/ppo.toml \
  --manifest data/generated/base/manifest.json \
  --algo ppo --mask-mode apply --seed 11 \
  --output experiments/masking/ppo/apply/11/train

uv run bus-rl model train \
  --config configs/studies/masking/ppo.toml \
  --manifest data/generated/base/manifest.json \
  --algo ppo --mask-mode free --seed 11 \
  --output experiments/masking/ppo/free/11/train
```

Chạy tương tự cho reinforce và a2c. Mỗi arm train từ đầu; không tái dùng
weights của arm còn lại.

### 5.2 Checkpoint là nguồn quyết định hành vi

Lưu mask mode lúc train và action_execution trong metadata/checkpoint.
Evaluate tự khôi phục cả hai. CLI không được mặc định apply làm ghi đè
checkpoint free.

Nếu evaluate có flag mask-mode, chỉ chấp nhận khi khớp checkpoint; mode
mâu thuẫn bị từ chối rõ. Tắt/bật mask chỉ ở inference của cùng checkpoint
là một câu hỏi khác, chưa thuộc phép A/B này.

Khôi phục checkpoint cũ: algorithm ppo, mask apply, invalid_action error.
Checkpoint spec thứ nhất: giữ đúng thuật toán, mask apply, execution error.
Checkpoint A/B mới: đọc đầy đủ mode và noop adapter đã ghi.

Metadata mới dùng learning_schema_version 2, bổ sung execution contract:

```text
algorithm.name, mask_mode
learning_schema_version = 2
action_execution.invalid_action = noop
action_execution.invalid_penalty = 0
action_execution.advance_interval = true
action_execution.version
action_execution_hash
```

Các giá trị penalty/advance_interval là khai báo quy tắc cố định, không
phải knob điều chỉnh trong spec này. Physical/reward/obs/action schema
hashes giữ nguyên. Execution hash riêng mô tả adapter và phải giống nhau
trong cặp apply/free; không đưa mask mode của policy vào execution hash.

## 6. Scalar, native batch và telemetry

Mapping phải hiện diện ở scalar training/evaluation, NativeBatchVecEnv
và NativeBatchEvalPool. Dùng một helper chung cho requested→executed,
không tạo ba định nghĩa khác nhau.

Factory và eval-pool cache key phải bao gồm execution contract. Không tái
sử dụng pool strict cho một run yêu cầu noop adapter vì dataset/runtime
trùng nhau.

Với native batch, lấy m_env của từng slot trước step, tạo một bản sao
executed_actions, rồi gọi kernel một lần. Kiểm shape/range của toàn batch
trước khi mutate bất kỳ slot nào. Native validation vẫn xử lý executed
actions như trước.

Mỗi bước lưu được:

```text
requested_action_index
executed_action_index
requested_action_valid
valid_action_count
invalid_action_fallback = noop | none
```

Giữ field trace action cũ mang nghĩa action thực thi để đọc trace vận hành
không sai; thêm field requested riêng. Nếu tên cũ hiện chứa action yêu cầu,
phải version hóa và sửa reader, không đổi nghĩa ngầm.

Episode/results thêm invalid_request_count, invalid_request_rate,
requested_noop_count và executed_noop_count. Invalid request không được
đếm thành một can thiệp vận hành đã thực hiện.

Diagnostics free không dùng assertion probability mass trên invalid phải
bằng 0 của mode apply. Thay bằng kiểm tổng probability mass bằng 1, finite,
và đo invalid_probability_mass = sum(pi(a), m_env[a]==False).

Entropy lấy trên phân phối policy thực tế: valid support trong apply,
full support trong free. Khi so normalized entropy giữa mode, dùng cùng
mẫu số log(221); nếu giữ metric normalize theo số action hợp lệ cũ thì
đặt tên riêng để tránh trộn hai định nghĩa.

Counters phải đúng qua auto-reset và terminal observation. Diagnostics
không resample action hoặc tiêu thụ RNG của policy.

## 7. Ma trận A/B và đọc kết quả

Tạo preset mới dưới configs/studies/masking cho reinforce/a2c/ppo và một
study matrix masking-ablation.toml. Mỗi preset đặt execution noop.
Hai mode trong một cặp có cùng mọi hyperparameter khác.

| Thuật toán | Arm A | Arm B |
|---|---|---|
| REINFORCE | apply + noop adapter | free + noop adapter |
| A2C | apply + noop adapter | free + noop adapter |
| PPO | apply + noop adapter | free + noop adapter |

Dùng cùng encoder GRU10, dataset/tapes, seeds 11/22/33, reward, rollout,
optimizer, budget và validation protocol của spec trước. Full matrix có
18 lần train. Có thể screening PPO trước, nhưng full comparison phải ghi
rõ arm/seed nào thực sự đã chạy.

Đối chiếu chính là apply so với free trong từng thuật toán. Báo paired cost,
waiting/service metrics, invalid rate, executed NOOP rate và wall time.
Giữ đủ trace để thấy arm free có lãng phí quyết định vì chọn sai không.

Không thêm penalty để làm free học nhanh hơn trong cùng study. Nếu nghiên
cứu penalty về sau, đó là một trục thí nghiệm mới với reward contract riêng.

Report/study identity phải bao gồm algorithm và mask mode; không gộp kết
quả ppo/apply và ppo/free vào cùng method khi aggregate/bootstrap.
Resume checks thêm execution hash và mode; không tái sử dụng kết quả
apply cho free vì cùng algorithm name.

## 8. Acceptance

1. Cùng state, một invalid request qua adapter cho đúng observation, reward,
   terminal và cost của NOOP thực hiện trực tiếp; thời gian tăng đúng Δ.
2. Invalid request ở bước cuối vẫn tính settlement và kết thúc ca đúng.
3. Out-of-range/type/batch-shape sai vẫn bị reject trước mọi mutation.
4. Apply không chọn invalid dù invalid logit cao nhất. Free có thể chọn
   invalid với logits đó; không bị resample hoặc sửa logit.
5. Buffer lưu requested action và raw-policy log probability; executed
   NOOP chỉ thuộc execution trace.
6. PPO ratio dùng cùng support ở old/new; REINFORCE/A2C cũng cập nhật
   requested action với phân phối đúng mode.
7. Scalar/native batch cho cùng requested/executed trace và cost với
   policy xác định; reset/counters không lẫn giữa episode.
8. Checkpoint round-trip khôi phục đúng mode/adapter. Old checkpoint giữ
   apply/error; explicit evaluation override mâu thuẫn bị từ chối.
9. Trong study A/B, execution hash, physical/reward hashes khớp giữa hai
   mode. Chỉ algorithm config/mask mode phân biệt policy arms.
10. Khi policy luôn request action hợp lệ, adapter không đổi trace/cost
    so với strict execution.
11. Một episode toàn invalid requests vẫn có đủ 120 transitions; metrics
    thể hiện invalid rate 1 và executed NOOP rate 1.
12. Direct native APIs vẫn reject invalid action như tests hiện có.

Hoàn thành khi hai mode chạy xuyên suốt ba trainer, CLI/checkpoint/report
không trộn nhãn và các kiểm tra đạt. Chạy Ruff/full pytest; nếu thay Rust
hoặc FFI ngoài thiết kế adapter này thì phải thêm gates theo AGENT.md.

## 9. Nguồn đối chiếu

- [Spec nền bắt buộc](masked_algorithm_comparison.md).
- [Scalar env](../../src/bus_rl/execution/environments/rust/environment.py).
- [Native batch](../../src/bus_rl/execution/environments/rust/batch.py).
- [Pool](../../src/bus_rl/evaluation/pool.py),
  [runner](../../src/bus_rl/evaluation/runner.py).
- [Native invalid-action validation](../../crates/bus-sim-python/src/lib.rs).
- [Batch contract tests](../../tests/integration/bus_rl/execution/environments/rust/test_batch_contract.py).
- [Diagnostics](../../src/bus_rl/learning/training/diagnostics.py).
- [MaskablePPO 2.9.0: learn/predict và use_masking](https://sb3-contrib.readthedocs.io/en/v2.9.0/modules/ppo_mask.html).
