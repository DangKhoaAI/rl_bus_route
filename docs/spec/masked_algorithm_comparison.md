# Bổ sung REINFORCE có mask và A2C có mask

Ngày: **2026-10-23**. Trạng thái: **đã triển khai và chạy xác nhận standard-v1**.

Spec này thêm hai thuật toán để so với MaskablePPO hiện tại.
[masking_ablation.md](masking_ablation.md) là phần mở rộng được triển khai sau phần nền này.
Không cần triển khai spec thứ hai để nghiệm thu spec này.

## 1. Mục tiêu và phạm vi

| Tên cấu hình | Thuật toán thực hiện | Mask |
|---|---|---|
| `reinforce` | REINFORCE Monte Carlo, không learned baseline | Apply |
| `a2c` | Advantage Actor–Critic đồng bộ, GAE, Adam | Apply |
| `ppo` | PPO-Clip qua sb3-contrib MaskablePPO hiện tại | Apply |

Actor–Critic được hiện thực bằng A2C; không thêm một thuật toán thứ tư mang
tên Actor–Critic. Tên thuật toán và mask mode được ghi riêng trong metadata.

Giữ nguyên tuyến, fleet, scenarios, dynamics, reward, horizon, 221 action,
guards và observation contract. Rust kernel vẫn từ chối action sai. Mọi
thuật toán áp mask khi thu dữ liệu, cập nhật và đánh giá.

Phạm vi nền này không gồm mask-free, đổi action sai thành NOOP, penalty mới,
REINFORCE có learned baseline hoặc persistent recurrent policy. Phần mask-free
thuộc spec mở rộng. Evidence lịch sử trong `reports/` giữ nguyên; kết quả mới
được bổ sung vào `runs/standard-v1` theo yêu cầu chạy xác nhận, giữ nguyên mọi
file evidence gốc.

## 2. Hiện trạng và thiết kế tích hợp

Env Rust, native batch, loader, encoder, metrics và artifact pipeline đều
có thể tái sử dụng. GRU hiện tại đọc lại cửa sổ arrivals mỗi lần forward,
nên thêm thuật toán không cần quản lý hidden state xuyên qua rollout.

| Thành phần | Hiện trạng | Việc cần làm |
|---|---|---|
| `config.py` | Chưa có selector thuật toán | Thêm `algorithm.name` |
| `training/train.py` | Luôn tạo MaskablePPO | Factory chọn đúng trainer |
| `training/checkpoint.py` | Luôn MaskablePPO.load | Load theo metadata |
| `training/callbacks.py` | Validation gắn method ppo | Dùng danh tính thực |
| `evaluation/runner.py` | PPO_METHODS, PPOController | Controller learned policy chung |
| `cli.py` | Evaluate checkpoint ép về ppo | Khôi phục algorithm từ checkpoint |
| Report/study runner | Nhận diện và mô tả PPO | Giữ tên và cấu hình từng arm |
| Diagnostics | Một số trường chỉ dành cho PPO | Tách kiểm tra chung và riêng |
| `training/diagnose.py` | Event/update và mean-cost gắn PPO | Timer và đánh giá theo trainer thực |

Ưu tiên tái dùng masked categorical policy, buffer và collector của
sb3-contrib 2.9.0 qua adapter/kế thừa. Hai trainer mới phải có loss đúng định
nghĩa; không sửa package trong `.venv/` hoặc sao chép một trainer PPO rồi chỉ
đổi tên. Không cần đổi Rust/FFI để làm phạm vi này.

## 3. Config và CLI

Thêm vào AlgorithmConfig:

```toml
[algorithm]
name = "ppo" # reinforce | a2c | ppo
```

Thiếu name mặc định ppo để config cũ tiếp tục chạy. Selector không hợp lệ
phải báo lỗi; không được để helper lọc field âm thầm bỏ qua selector mới.

Thêm `--algo {reinforce,a2c,ppo}` cho `model train` và `test diagnose`.
Ưu tiên CLI truyền rõ → TOML → default. Parser không tự ghi đè TOML khi
người dùng không truyền flag.

Mask luôn apply trong spec này. Metadata ghi `mask_mode="apply"`; flag
bật/tắt mask thuộc spec thứ hai.

Evaluate checkpoint tự khôi phục algorithm, encoder/history và các thiết
lập quyết định hành vi. Method truyền rõ mà mâu thuẫn checkpoint phải báo
lỗi. Phân biệt được method mặc định của parser với lựa chọn truyền rõ.

Giao diện đích sau triển khai, mỗi output phải mới:

```bash
uv run bus-rl model train \
  --config configs/studies/algorithms/reinforce.toml \
  --manifest data/generated/base/manifest.json \
  --algo reinforce --seed 11 --output experiments/algorithms/reinforce/11/train

uv run bus-rl model train \
  --config configs/studies/algorithms/a2c.toml \
  --manifest data/generated/base/manifest.json \
  --algo a2c --seed 11 --output experiments/algorithms/a2c/11/train

uv run bus-rl model evaluate \
  --manifest data/generated/base/manifest.json \
  --checkpoint experiments/algorithms/a2c/11/train/best.zip \
  --split test_id --output experiments/algorithms/a2c/11/eval/test_id
```

## 4. Hợp đồng chung của policy và dữ liệu

Mỗi trainer cung cấp learn, predict, save và loader tương ứng. Predict nhận
observation scalar/batch và action_masks; deterministic predict chọn action
hợp lệ có xác suất cao nhất.

Mỗi mẫu lưu observation đầu bước, mask cùng thời điểm, action, reward,
episode-start/termination và log probability cũ. A2C/PPO lưu thêm old value.
Khi optimize phải recompute log probability bằng observation/action/mask đã
lưu; không dùng mask của state mới nhất trong live env.

Targets và advantage không nhận gradient từ actor loss. Critic học target
đã tách gradient.

### 4.1 Mạng dùng chung

Study đầu dùng cùng encoder GRU, history 10 phút, không belief features.
Giữ cùng actor/encoder; A2C/PPO có cùng critic architecture.

REINFORCE không dùng critic. Nếu primitive chung vẫn tạo value head thì
freeze nhánh value riêng, không đưa nó vào optimizer hoặc tính value loss.
Metadata ghi `critic_used=false` và tách tổng tham số khỏi tham số thực sự
được tối ưu. Không thay actor chỉ vì REINFORCE không cần critic.

### 4.2 Rollout gồm episode hoàn chỉnh

Episode chuẩn có K = horizon_s / control_interval_s = 120 bước. Theo yêu cầu dùng đúng setup standard-v1, giữ n_steps khai báo 256.
REINFORCE thu 240 bước thực tế mỗi env (hai ca hoàn chỉnh), là bội K lớn
nhất không vượt 256; ghi rõ effective_n_steps. PPO/A2C vẫn thu 256 bước.
Cùng tổng số transitions, dữ liệu, reward và runtime:

| Tham số | Giá trị |
|---|---:|
| n_envs | 4 |
| n_steps khai báo | 256 |
| n_steps thực tế PPO/A2C / REINFORCE | 256 / 240 |
| Transitions/rollout PPO/A2C / REINFORCE | 1.024 / 960 |
| Episode/env/rollout REINFORCE | 2 |
| Full budget/seed | 245.760 |
| Screening budget/seed | 61.440 |
| Validation interval yêu cầu | 12.288 |
| Seeds xác nhận | 11, 22, 33 |

Các slot bắt đầu từ episode boundary, các scenario có cùng horizon. Tính
return riêng từng slot và từng episode; không nối qua terminal.

REINFORCE v1 yêu cầu gamma 1, horizon chia hết cho control interval,
n_steps ít nhất bằng K. Collector lấy floor(n_steps/K)*K bước;
total_timesteps phải chia hết cho n_envs*effective_n_steps. Request không
đáp ứng bị từ chối trước khi tạo output. Không đổi physical horizon để làm
vừa budget.

`fit_algorithm()` hiện thu nhỏ rollout khi smoke ít bước. Với REINFORCE,
không được thu nhỏ thành đoạn episode rồi bootstrap. Smoke cần ít nhất
một rollout hoàn chỉnh; request ngắn hơn phải báo lỗi. External truncation
không được coi như terminal hoàn tất cho Monte Carlo; v1 từ chối rollout đó.

## 5. Định nghĩa cập nhật

Ký hiệu x là đầu vào, m là mask, a là action, r là reward chuẩn hóa.
Các công thức dưới dùng gamma 1.

### 5.1 REINFORCE

Giữ policy cố định khi thu rollout. Tính return từ reward thật tới terminal:

```text
G_t = sum(r_j, j=t..K-1)
L_actor = -mean(log pi(a_t | x_t, m_t) * stopgrad(G_t))
L = L_actor - ent_coef * mean(entropy(pi))
```

Mean trên toàn rollout. Các episode cùng độ dài nên tương đương mean tổng
episode sau một hệ số hằng. Đây là REINFORCE không baseline, có entropy
regularization được ghi rõ.

Không dùng GAE, không trừ value, không normalize return theo minibatch,
không bootstrap và không scale reward lần hai.

Mỗi rollout có **một optimizer step trên toàn lô**, rồi thu dữ liệu mới.
Dùng Adam, learning rate và gradient limit chung. Không chạy nhiều epoch
policy update trên cùng return cũ.

### 5.2 A2C

```text
delta_t = r_t + (1-done_t)*V_old(x_(t+1)) - V_old(x_t)
A_t = delta_t + gae_lambda*(1-done_t)*A_(t+1)
return_target_t = A_t + V_old(x_t)
A_norm = (A - mean(A)) / (std(A) + epsilon)
L_actor = -mean(log pi(a_t | x_t, m_t) * stopgrad(A_norm_t))
L_value = mean((V(x_t) - stopgrad(return_target_t))**2)
L = L_actor + vf_coef*L_value - ent_coef*mean(entropy(pi))
```

Normalize advantage theo toàn rollout, sau khi đã tính raw return target.
Terminal value bằng 0. Mỗi rollout có **một optimizer step toàn lô**.
Không dùng probability ratio, PPO clipping hoặc nhiều epoch.

Dùng Adam để giữ optimizer chung với PPO. Metadata mô tả rõ phiên bản
`A2C + GAE + Adam`, không mặc nhận mọi default của SB3 A2C.

### 5.3 PPO

Giữ loss PPO-Clip của MaskablePPO hiện tại, gồm GAE, probability ratio,
clipping, value loss và entropy. Study mới giữ batch_size 256,
n_epochs 4 của standard-v1: mỗi rollout PPO tạo 4 minibatch/epoch, tổng 16
optimizer steps.

Không coi PPO tắt clipping là implementation A2C/REINFORCE.

### 5.4 Thiết lập chung và riêng

Common: learning rate 3e-4, gamma 1, entropy coefficient 0,01, gradient norm
limit 0,5. Giữ cùng Adam betas/epsilon với PPO control và lưu chúng trong
resolved config. A2C/PPO dùng vf_coef 0,5 và GAE lambda 0,95. PPO clip_range 0,2.

Config chung có thể còn field riêng PPO do dataclass hiện có, nhưng
resolved metadata phải chỉ rõ field thực sự dùng. Không báo A2C/REINFORCE
chạy 4 epoch vì field đó còn trong TOML.

Đối chiếu chính bằng cùng số transitions; ghi thêm optimizer steps và wall
time vì số lần cập nhật là khác biệt của thuật toán.

## 6. Validation, checkpoint và report

### 6.1 Chọn checkpoint

Dùng cùng train/validation/test days và tapes. Validation ở các mốc rollout
đã đăng ký, sau cập nhật, deterministic và mask apply. Step zero là mốc
policy chưa học, không đủ điều kiện chọn best.

Chọn mean total_cost_core nhỏ nhất sau cập nhật; bằng nhau giữ bản sớm.
Luôn đánh giá trọng số sau optimizer step cuối trước khi hoàn tất.
Callback phải phân biệt model-update version; không bỏ đánh giá cuối vì
đã có một lần đo trước update với cùng num_timesteps.

Train lại PPO control cho study bổ sung. Giữ n_steps 256 như standard-v1;
REINFORCE có effective_n_steps 240 được ghi rõ trong report. Validation
được đo sau update ở rollout boundary đầu tiên đạt mốc yêu cầu 12.288;
ghi step thực tế, không giả định các thuật toán cập nhật ở cùng step.

### 6.2 Metadata và tương thích

Lưu tối thiểu:

```text
learning_schema_version = 2
algorithm.name, mask_mode=apply, trainer_class, trainer_version
encoder, history_minutes, critic_used, optimizer, effective hyperparameters
rollout_steps, completed_episodes, optimizer_steps
requested_transitions, actual_transitions
validation_timing=post_update, best_step, best_update
physical/control/reward/obs/action hashes, dataset identity, seeds
```

Checkpoint cũ thiếu algorithm.name được nhận diện PPO + apply. Giữ
schema/physical checks và legacy import mapping. Checkpoint mới load đúng
trainer; class không biết phải báo lỗi thay vì load nhầm thành PPO.
Schema 2 được dùng chung với execution metadata của phần mở rộng masking;
checkpoint cũ thiếu section này vẫn khôi phục apply/error.

### 6.3 Evaluator và diagnostics

Generalize learned controller cho ba thuật toán, bao gồm scalar/native
batch và các nhánh kiểm tra PPO_METHODS/isinstance. Không tắt batch riêng
A2C/REINFORCE khi API chung đủ khả năng.

Results giữ đúng tên từng thuật toán. Report/study tự nhận diện arm cần
so sánh thay vì luôn chọn ppo; mô tả estimator, optimizer và mask mode đúng.

Diagnostics chung kiểm finite logits/loss/gradients/return, valid-action
count, entropy và probability mass trên action sai bằng 0. Value metrics
chỉ có khi critic được dùng. Clip fraction/KL chỉ dành PPO; arm khác dùng
null hoặc bỏ trường, không tạo số 0 giả.

Giữ experiments cho checkpoint/logs, runs cho evidence export.
Resume checks so cả algorithm và trainer version/config thực tế.

## 7. Preset và thí nghiệm

Tạo mới các file sau, không sửa study lịch sử:

```text
configs/studies/algorithms/reinforce.toml
configs/studies/algorithms/a2c.toml
configs/studies/algorithms/ppo.toml
configs/studies/algorithm-comparison.toml
```

Runtime chung: Rust, native batch bật, torch_threads 2, eval batch size 16,
reuse pool bật. Forecast và reward shaping tắt.

Thứ tự: correctness smoke → screening 61.440 transitions → xác nhận
245.760 transitions/seed, seeds 11/22/33. Screening dùng validation. Khóa
cấu hình trước khi đánh giá test_id, test_ood_burst, test_ood_traffic.

Báo paired cost, mean/P95 wait, worst-route mean wait,
completed/abandoned/unfinished share, seed spread, optimizer steps, wall
time. Không đặt yêu cầu thuật toán mới phải thắng PPO để được nghiệm thu.

## 8. Kiểm tra và acceptance

1. Return tính đúng trên chuỗi biết trước; `[-1,-2,-3]` tạo `[-6,-5,-3]`.
   Return của episode sau không ảnh hưởng episode trước.
2. REINFORCE không update nhánh value riêng. Thay value head không đổi
   return/actor loss; gradient vẫn đi vào actor/encoder.
3. GAE/target A2C khớp ví dụ tính tay, terminal không bootstrap.
4. Số optimizer steps và cách dùng loss đúng định nghĩa từng thuật toán.
5. Logit cao nhất thuộc action sai vẫn không làm actor lấy mẫu/chọn action
   sai; update dùng mask lưu trong rollout.
6. Prediction và checkpoint cost tái lập sau save/load; scalar và native
   batch cho cùng deterministic action trace/cost.
7. CLI/TOML chọn đúng trainer; config cũ vẫn chọn PPO. Selector sai và budget
   REINFORCE không tròn effective rollout bị từ chối trước khi train.
8. Train → best checkpoint → evaluate → report → export/verify giữ đúng
   nhãn/hashes của ba arm.
9. Trọng số cuối được đánh giá sau update, không bị bỏ vì trùng step count.

Unit tests theo module mới trong tests/unit/bus_rl/learning; integration
tests dưới tests/integration/bus_rl/learning/training và entry-point
CLI/report tương ứng. Dùng env Rust, không thêm test cho oracle deprecated.

Chạy Ruff và full pytest theo AGENT.md khi triển khai. Nếu không sửa
Rust/FFI thì không cần đổi kernel chỉ để thêm trainer.

Pipeline ba arm, kiểm tra toán học, checkpoint round-trip và scalar/native
parity đã đạt. Đã chạy screening 61.440 transitions cho từng arm seed 11,
sau đó chạy đầy đủ 245.760 transitions với seeds 11/22/33 trong ma trận A/B.
Không thay hyperparameter sau screening.

Evidence có mask tương ứng tại
[`runs/standard-v1/evidence/reinforce/apply`](../../runs/standard-v1/evidence/reinforce/apply),
[`a2c/apply`](../../runs/standard-v1/evidence/a2c/apply) và
[`ppo/apply`](../../runs/standard-v1/evidence/ppo/apply).
Mỗi seed có metadata/lịch sử validation và 200 ngày cho từng split test.
[`report.html`](../../runs/standard-v1/report.html) phần III so sánh trực tiếp
ba thuật toán với PPO control được train lại; không dùng PPO snapshot cũ
để kết luận hiệu quả thuật toán.

## 9. Nguồn đối chiếu

- [Problem](../problem_solution.md).
- [Training](../../src/bus_rl/learning/training/train.py),
  [policy](../../src/bus_rl/learning/policy.py),
  [config](../../src/bus_rl/config.py), [CLI](../../src/bus_rl/cli.py).
- [Checkpoint](../../src/bus_rl/learning/training/checkpoint.py),
  [callback](../../src/bus_rl/learning/training/callbacks.py).
- [Evaluator](../../src/bus_rl/evaluation/runner.py),
  [report](../../src/bus_rl/evaluation/report.py),
  [study runner](../../src/bus_rl/experiments.py).
- [Factory](../../src/bus_rl/execution/environments/factory.py),
  [native batch](../../src/bus_rl/execution/environments/rust/batch.py),
  [diagnostics](../../src/bus_rl/learning/training/diagnostics.py).
- [MaskablePPO 2.9.0](https://sb3-contrib.readthedocs.io/en/v2.9.0/modules/ppo_mask.html).
