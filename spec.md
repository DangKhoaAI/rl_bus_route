# Spec: RL cho thiết kế mạng tuyến bus cố định

Phiên bản: **0.1 — thiết kế ban đầu, 2026-09-12**.

Phạm vi do người dùng chọn: **thiết kế tuyến bus cố định**. Các mặc định dưới đây là quyết định thiết kế để bắt đầu triển khai, chưa được xác nhận bằng thực nghiệm. Trạng thái hiện tại: tài liệu; chưa có implementation hoặc model được train.

Đọc cùng [research](docs/research.md) và [implementation plan](plan.md).

## 1. Mục tiêu và phạm vi

Xây một pipeline Python có thể sinh thành phố tổng hợp, tạo tuyến ứng viên, đánh giá mạng tuyến, huấn luyện RL để chọn mạng, và so sánh với heuristic trên cùng giao thức.

**MVP giải bài toán chọn K tuyến từ một pool hữu hạn**, là một phiên bản hạn chế của TNDP. Tuyến cố định nghĩa là sau khi thiết kế, xe chạy theo chuỗi trạm đã chọn; agent không điều hướng xe theo yêu cầu thời gian thực.

### Trong phạm vi bắt buộc

- Graph trạm–đường vô hướng, thời gian cạnh xác định, tuyến hai chiều.
- Ma trận nhu cầu OD có hướng, cố định trong một instance.
- Chọn đúng K tuyến khác nhau, giới hạn số trạm/tuyến và tổng route-time.
- Headway cố định; hành khách đi tối đa hai lần chuyển tuyến.
- Synthetic generator, evaluator, baselines, Gymnasium environment, Maskable PPO.
- Training nhiều seed, ID/OOD evaluation, ablation reward và báo cáo hình/CSV.

### Ngoài MVP

- Đặt vị trí trạm mới, thiết kế đường giao thông hoặc tối ưu timetable.
- Đón/trả theo yêu cầu, multi-agent điều khiển từng xe.
- Tối ưu headway, số xe, sức chứa, hàng chờ, chi phí nhiên liệu thực.
- Congestion thay đổi theo lưu lượng; mô phỏng vi mô; triển khai vận hành thực.
- Web backend/frontend, tài khoản, cloud deployment.
- GNN, tuyến tự xây từng trạm và dữ liệu thành phố thật: hướng mở rộng, không chặn nghiệm thu MVP.

## 2. Mặc định chung

Các giá trị thuộc profile `base`; benchmark khác phải dùng config riêng và ghi rõ trong báo cáo.

| Tham số | Giá trị |
|---|---|
| Python | 3.11; khóa patch version khi tạo môi trường |
| OS mục tiêu | Linux; CPU là cấu hình kiểm thử bắt buộc |
| Số trạm train/base | N = 20 |
| Giới hạn encoding | N_max = 32; M_max = 512 |
| Số tuyến chọn | K = 4 |
| Số trạm/tuyến | 2 ≤ L ≤ 8 |
| Thời gian một cạnh | 1–8 phút |
| Ngân sách B | 100 phút tổng thời gian đi một chiều của các tuyến |
| Headway h | 10 phút cho từng tuyến |
| Penalty chuyển tuyến τ | 3 phút mỗi lần, ngoài thời gian chờ |
| Số lần chuyển tuyến tối đa q | 2 |
| Tổng OD Q | 10,000 lượt trong một kỳ nhu cầu giả định |
| Trọng số α, λ_u | 0.7 và 2.0 |
| Discount γ | 1.0; episode hữu hạn đúng K hành động |
| Training seeds | 11, 22, 33 |

**Đơn vị:** thời gian là phút; tọa độ chỉ phục vụ hình học tương đối; OD là lượt/kỳ. B là surrogate cho vận hành, không phải số tiền hay số xe. Route-time là một chiều; hai chiều chỉ là hệ số 2 khi h cố định, không cộng hai lần trong objective.

## 3. Mathematical foundation

### 3.1 Đầu vào, tuyến và không gian nghiệm

Cho graph liên thông \(G=(V,E,t)\), \(|V|=N\), \(t_{uv}>0\); \(t_{uv}=t_{vu}\). \(D\in\mathbb R_+^{N\times N}\), \(D_{ii}=0\), \(Q=\sum_{i\ne j}D_{ij}>0\).

Một tuyến là simple path \(r=(v_1,\ldots,v_L)\), các trạm không lặp, mỗi cặp liên tiếp thuộc E. Tuyến và tuyến đảo chiều được coi là một ứng viên:

\[
\operatorname{canon}(r)=\min_{\mathrm{lex}}(r,\operatorname{reverse}(r)).
\]

Chi phí tuyến:

\[
\ell(r)=\sum_{a=1}^{L-1}t_{v_a,v_{a+1}}.
\]

Pool \(\mathcal C=\{r_1,\ldots,r_M\}\). Tìm tập \(\mathcal R\subseteq\mathcal C\):

\[
|\mathcal R|=K,\qquad \sum_{r\in\mathcal R}\ell(r)\le B.
\]

Ràng buộc chiều dài đã được kiểm tra khi sinh pool. Không cấm tuyến giao nhau hoặc trùng một phần vì chúng có thể tạo chuyển tuyến. Không yêu cầu toàn bộ mạng hay mọi OD được kết nối như hard constraint; thiếu phục vụ bị phản ánh trong objective và metrics.

### 3.2 Mô hình hành trình hành khách

Với mỗi OD i→j, tìm hành trình có generalized time thấp nhất trên các tuyến đã chọn:

\[
c_{ij}(\mathcal R)=\min_{p\in\mathcal P_{ij}(\mathcal R),\,m(p)\le q}
\left[t_{\mathrm{ride}}(p)+(m(p)+1)\frac h2+m(p)\tau\right].
\]

\(m(p)\) là số lần chuyển tuyến. Nếu tập hành trình rỗng, \(c_{ij}=+\infty\) nội bộ. Không export infinity/NaN vào JSON hoặc tensor observation.

Giả định hành khách đến ngẫu nhiên so với headway đều cho kỳ vọng chờ h/2 mỗi lần lên xe; không xét đồng bộ lịch hoặc gộp tần suất nhiều tuyến chung hành lang. Đi bộ tiếp cận và chuyển trạm khác vị trí đều bằng 0/không được mô hình hóa. Chỉ chuyển tuyến tại cùng một trạm.

**Graph evaluator:** state `(stop, route_id, transfers_used)` với `transfers_used ∈ {0,1,2}`. Thêm virtual source riêng cho origin, nối đến mỗi tuyến tại origin với chi phí h/2. Cạnh ride nối trạm kề nhau trên cùng tuyến, hai chiều, giữ nguyên transfers. Cạnh transfer đổi route tại cùng stop, tăng transfers và cộng h/2 + τ. Destination lấy minimum trên mọi route/layer tại stop đích. Dijkstra dùng trọng số không âm. Tie-break: generalized time, ít chuyển tuyến, route-id theo thứ tự.

Ví dụ bắt buộc: đường 0–1–2, mỗi cạnh 4 phút; tuyến `(0,1,2)` cho 0→2 chi phí 5+8=13. Hai tuyến `(0,1)` và `(1,2)` cho chi phí 5+4+5+3+4=21. Thiếu tuyến chạm trạm 2 thì 0→2 chưa phục vụ.

### 3.3 Objective có giá trị hữu hạn

\(d^G_{ij}\) là thời gian ngắn nhất trên graph đường, dùng làm reference chứ không thay cho hành trình bus:

\[
T_{\mathrm{ref}}=\frac{\sum_{i\ne j}D_{ij}d^G_{ij}}Q+\frac h2>0.
\]

Chọn penalty chưa phục vụ lớn hơn mọi hành trình hợp lệ theo giới hạn route/transfer:

\[
C_{\max}=(q+1)(L_{\max}-1)t_{\max}+(q+1)h/2+q\tau,
\qquad P=C_{\max}+T_{\mathrm{ref}}.
\]

Với profile base, \(C_{\max}=189\) phút. Điều kiện bound: tuyến simple, tối đa L_max trạm, mỗi cạnh ≤ t_max, tối đa q chuyển. Với dataset khác, dùng bound suy ra từ config thực; reject dữ liệu vượt bound.

Đặt \(\bar c_{ij}=c_{ij}\) nếu phục vụ được, bằng P nếu không. Sau đó:

\[
C_p(\mathcal R)=\frac{\sum_{i\ne j}D_{ij}\bar c_{ij}}{QT_{\mathrm{ref}}},\quad
C_o(\mathcal R)=\frac{\sum_{r\in\mathcal R}\ell(r)}B,
\]

\[
U(\mathcal R)=\frac{\sum_{i\ne j}D_{ij}\mathbf 1[c_{ij}=\infty]}Q,
\]

\[
\boxed{J(\mathcal R)=\alpha C_p(\mathcal R)+(1-\alpha)C_o(\mathcal R)+\lambda_u U(\mathcal R).}
\]

Mục tiêu là **minimize J**. P bảo đảm OD không được miễn chi phí khi bị bỏ; λ_u là mức ưu tiên bổ sung, được báo cáo và ablate. Đây là weighted objective, không bảo đảm ưu tiên coverage tuyệt đối kiểu lexicographic. T_ref, B và P cố định trong một episode, tính chỉ từ instance/config. T_ref phụ thuộc OD nên giữa các instance metric chuẩn hóa không bằng số phút thực; luôn báo cáo cả số phút.

### 3.4 MDP và Bellman

MDP thiết kế mạng: \((\mathcal S,\mathcal A,\mathcal T,r,\gamma)\).

- State \(s_t=(G,D,\mathcal C,x_t,B_{\rm remaining},K-t,\mathrm{config})\), \(x_t\in\{0,1\}^M\) chỉ tuyến đã chọn.
- Action a_t là index một tuyến chưa chọn và feasible theo §3.5.
- Transition xác định: thêm tuyến, trừ route-time, tính lại evaluator.
- Reset chọn instance từ train; validation/test nhận instance cố định.
- Episode có đúng K action hợp lệ; không có STOP, không có chạy xe theo thời gian thực.
- Terminal sau action thứ K. Không dùng time-limit truncation trong run chuẩn.

\[
V^\pi(s)=\mathbb E_\pi\left[\sum_{k=t}^{K-1}\gamma^{k-t}r_k\mid s_t=s\right],
\qquad
Q^\pi(s,a)=r(s,a)+\gamma\mathbb E[V^\pi(s')].
\]

\(V(s_K)=0\). Policy học lợi ích của tổ hợp tuyến, không chỉ score riêng mỗi tuyến.

### 3.5 Mask có bảo đảm hoàn thành K tuyến

Giả sử trước action đã chọn t tuyến, action thử là a. Gọi \(b'=B_{\rm remaining}-\ell(r_a)\), \(k'=K-t-1\), S là các tuyến chưa chọn trừ a. Cho phép a khi:

1. a là tuyến thật, chưa chọn; b' ≥ 0.
2. |S| ≥ k'.
3. Tổng chi phí k' tuyến rẻ nhất trong S ≤ b'. Tổng rỗng bằng 0.

Điều kiện này đủ và cần để hoàn thành đối với các hard constraints hiện tại: cardinality, uniqueness và additive budget. Nó không chứng minh full coverage. Nếu thêm ràng buộc connectivity, chứng minh này không còn đủ.

Reset từ chối pool có M<K hoặc tổng K tuyến rẻ nhất>B. Mask của mọi state chưa terminal phải có ít nhất một action. Không sửa ngầm budget, không tự chọn fallback trong evaluation. Action invalid từ caller gây `ValueError` với action, step và lý do; không biến lỗi integration thành experience train.

### 3.6 Reward và credit assignment

Default dense reward:

\[
r_t=J(\mathcal R_t)-J(\mathcal R_{t+1}).
\]

Với γ=1 và episode hoàn thành:

\[
\sum_{t=0}^{K-1}r_t=J(\varnothing)-J(\mathcal R_K).
\]

J(∅) hữu hạn và độc lập policy trên cùng instance, nên tối đa expected return tương ứng tối thiểu expected final J. Không cộng thêm `-J(final)` ở cuối vì sẽ tính lại objective. Không clip reward trong default.

Ablation terminal-only: r_t=0 trước bước cuối, bước cuối r=−J(final). Hai reward có cùng xếp hạng terminal network trên một instance. Không đổi γ để tiện dùng mặc định thư viện vì sẽ phá đẳng thức telescope.

Ví dụ đơn vị reward: chi phí đi 6→4→3 cho rewards 2 và 1, return=3. Giá trị J không được hard-code thành reward thực của dataset.

## 4. Algorithm và model

### 4.1 Maskable PPO

Policy phân phối categorical trên M_max logits. Tuyến bị mask nhận xác suất 0. Xác suất action hợp lệ được chuẩn hóa lại trên mask tại state đó. Lưu/recompute cùng mask khi cập nhật log-probability.

Với \(\rho_t=\pi_\theta(a_t|s_t)/\pi_{\theta_{old}}(a_t|s_t)\):

\[
L^{clip}=\mathbb E_t[\min(\rho_t\hat A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t)].
\]

Tối thiểu hóa loss \(-L^{clip}+c_v\mathbb E[(V_\theta-\hat G)^2]-c_H H(\pi_\theta)\).

GAE: \(\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)\), \(\hat A_t=\sum_l(\gamma\lambda)^l\delta_{t+l}\), cắt ở terminal và dùng V_terminal=0. PPO là thuật toán policy gradient với value baseline; không phải Q-learning.

Tham khảo lý thuyết [R4](https://arxiv.org/abs/1707.06347), masking [R5](https://arxiv.org/abs/2006.14171), integration [R6](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html). Công thức/objective TNDP trong spec là thiết kế riêng.

### 4.2 Observation và kiến trúc baseline

Gymnasium `Dict` chứa float32 arrays, bool arrays được encode 0/1 khi cần; padding luôn bằng 0. Thông tin static lặp lại ở mỗi observation để policy quan sát đủ instance.

| Key | Shape | Nội dung |
|---|---|---|
| `nodes` | [32,4] | x, y, outgoing demand/Q, incoming demand/Q |
| `node_valid` | [32] | Mask trạm thật |
| `road_time` | [32,32] | Thời gian cạnh/t_max, nonedge=0 |
| `road_adj` | [32,32] | Phân biệt nonedge và diagonal |
| `od` | [32,32] | D/Q |
| `route_incidence` | [512,32] | Trạm nằm trên tuyến |
| `route_sequence` | [512,8] | Thứ tự trạm, giá trị (node_id+1)/32, padding=0 |
| `route_meta` | [512,2] | ℓ/B, số trạm/L_max |
| `route_valid` | [512] | Ứng viên thật |
| `selected` | [512] | Tuyến đã chọn |
| `context` | [8] | t/K, budget còn/B, K/4, α, λ_u/2, h/10, τ/3, q/2 |

`route_sequence` và incidence cùng mô tả đầy đủ simple path hai chiều; evaluator luôn dùng tuple route gốc. Observation giữ graph + OD + tập đã chọn, không chỉ aggregate coverage, để tránh thiếu tính Markov. Config giới hạn L_max/t_max/N_max cố định cho một model; không load checkpoint với config encoding khác.

**Encoder khả thi cho MVP:** custom `BaseFeaturesExtractor`. Với mỗi route, concat incidence [32], sequence [8], route_meta [2], selected [1], valid [1] → MLP dùng chung 44→64→16; nhân embedding với route_valid để padding không sinh tín hiệu từ bias. Flatten 512 route embeddings (8192 giá trị); concat flatten nodes/adj/time/OD/node_valid/context → MLP 256→128. Actor/critic mỗi nhánh hidden 128; actor xuất 512 logits, critic xuất một scalar. Sequence dùng node-id là baseline đơn giản, chưa có inductive bias graph. Kiểm tra profiler trước khi tăng batch/env count.

MVP không bảo đảm permutation equivariance. Candidate IDs được sắp ổn định theo canonical path; khi relabel city phải dựng lại pool và observation. OOD size nằm trong N_max nhưng kết quả không được coi là bảo đảm tổng quát quy mô. Shared scorer/attention hoặc GNN là mở rộng riêng sau baseline.

### 4.3 Tham số training khởi đầu

| Tham số | Mặc định pilot |
|---|---|
| learning_rate | 3e-4 |
| n_envs | 4, DummyVecEnv trước; tăng sau profiling |
| n_steps mỗi env | 128 |
| batch_size | 128 |
| n_epochs | 4 |
| gamma / gae_lambda | 1.0 / 0.95 |
| clip_range | 0.2 |
| ent_coef / vf_coef | 0.01 / 0.5 |
| max_grad_norm | 0.5 |
| pilot budget | 20,480 environment steps |
| full budget ban đầu | 204,800 steps/seed, 3 seeds |
| validation interval | mỗi 10,240 environment steps |

Đây là tổng transitions trên các env, không phải số episode; với K=4, full budget tương ứng 51,200 episode nếu luôn chạy đầy đủ. Ghi số thực tế thư viện thực hiện; tính callback interval theo n_envs. Không hứa hội tụ trong budget này. Smoke test dùng budget nhỏ khác để xác nhận plumbing.

## 5. Synthetic data và candidate generation

### 5.1 Generator

- Hai họ graph: `jittered_grid` và `geometric_knn`. Grid chọn ô gần hình vuông với N điểm, giữ cạnh lưới và bổ sung MST nếu cần; jitter tọa độ không làm mất topology.
- Geometric: uniform points, k=3 nearest neighbors, đối xứng hóa rồi hợp với Euclidean MST.
- Với cạnh uv: raw=`euclidean_distance * uniform(0.9,1.1)`; scale min/max raw về [1,8] phút. Nếu tất cả raw bằng nhau dùng 4 phút. Nhiễu draw một lần mỗi cạnh vô hướng.
- OD theo công thức [research §4](docs/research.md), hoặc uniform. Vai trò P,A trong [0.5,2.0]; chọn 25% node làm residential và 25% khác làm employment, nhân lần lượt P hoặc A với 4. σ=median positive road shortest-time; η lognormal với log-mean −0.125, log-std 0.5.
- Q=10,000, diagonal=0. Không ép OD đối xứng dù tuyến/đường hai chiều.
- Mỗi instance có `schema_version`, `instance_id`, graph-family, seed, generator-config, hash graph và manifest split.

### 5.2 Candidate pool

Sinh một weighted shortest simple path cho mỗi unordered pair i<j; equal-cost tie-break theo node-id bằng graph được xây có thứ tự ổn định. Canonicalize, lọc 2≤L≤8, deduplicate, sort lexicographic. Với N≤32 có tối đa 496 cặp, nằm trong M_max=512. Không lọc theo D ở MVP để baseline không được lợi khác nhau từ candidate generation.

Một pool thiếu tuyến dài hoặc vòng qua vùng nhu cầu cao là hạn chế đã biết. Tuyến qua waypoint chỉ làm ở extension có `candidate_version` mới; tuyệt đối không so hai model/pool khác nhau mà gọi là ảnh hưởng riêng của thuật toán.

Reject instance nếu K tuyến rẻ nhất không nằm trong B; tối đa 100 lần sinh lại bằng child seed, sau đó báo lỗi có config. Ghi tỷ lệ rejection để phát hiện selection bias. Lưu pool một lần để mọi phương pháp đọc cùng candidate IDs/hash.

### 5.3 Split

| Tập | Số instance | Nội dung |
|---|---:|---|
| tiny | Các fixture N=3–6, M≤12, K=2–3 | Tính tay hoặc enumeration |
| train | 2,000 | N=20, 50/50 hai họ graph; uniform/clustered OD 50/50 |
| validation | 200 | Thành phố/seed riêng, cùng phân phối train |
| test_id | 300 | Thành phố riêng, cùng phân phối |
| test_ood_demand | 300 | N=20 mới; hệ số hotspot từ 4 lên 8 |
| test_ood_size | 300 | N=30 mới, K/B giữ nguyên; ghi cả hiệu ứng thiếu tài nguyên/trạm |

Seed gốc: train=1001, validation=2001, test_id=3001, test_ood_demand=4001, test_ood_size=5001. Hash graph không được trùng giữa split, kể cả khác OD. Group theo base graph trước mọi augmentation. Một nghiên cứu size thuần với resources scale theo N cần profile riêng; không trộn với test_ood_size mặc định.

## 6. Code architecture và thư viện

Ngôn ngữ chính **Python 3.11**, package tên `bus_rl`, layout `src/`. Config TOML đọc bằng `tomllib`; dataclass typed cho domain. Dependency cụ thể khóa bằng uv khi implementation và lưu `uv.lock`; không khẳng định các phiên bản chưa cài đã tương thích.

| Công cụ/thư viện | Vai trò |
|---|---|
| NumPy | Ma trận, RNG, .npz |
| NetworkX | Graph, MST, shortest path, reference evaluator |
| PyTorch | Neural policy/encoder |
| Gymnasium | Env API reset/step và spaces |
| stable-baselines3 + sb3-contrib | Maskable PPO, vector env, callbacks |
| pandas | CSV tổng hợp thí nghiệm |
| Matplotlib | Hình tuyến, đường học, trade-off |
| pytest | Domain/evaluator/env/integration tests |
| Ruff | Lint và formatting |
| uv | Environment và dependency locking |

Chọn wheel PyTorch theo CPU/CUDA thực tế; CPU import và smoke bắt buộc. Không thêm PyTorch Geometric, Ray, SUMO, web framework ở MVP.

```text
docs/research.md
spec.md
plan.md
pyproject.toml                  # project metadata, deps, bus-rl entrypoint
uv.lock                        # resolved khi setup
configs/{base,pilot,train,eval}.toml
src/bus_rl/
  domain.py                    # City, Route, ProblemConfig, Instance, Evaluation
  data/{generate,candidates,io}.py
  transit/{paths,metrics}.py   # passenger routing và objective thuần
  env/{masking,observation,network_design}.py
  baselines/{random,greedy,local_search,exact}.py
  models/features.py
  training/{train,callbacks,checkpoint}.py
  evaluation/{runner,statistics,plots}.py
  cli.py
tests/{fixtures,test_data,test_candidates,test_paths,test_metrics,
       test_masking,test_env,test_baselines,test_model,test_pipeline}.py
data/{generated,manifests}/
runs/<run_id>/
reports/<experiment_id>/
```

Luồng phụ thuộc: domain ← data/transit ← env/baselines ← training/evaluation ← CLI. Evaluator không import RL; generator không gọi training. Cùng một hàm đánh giá được dùng cho mọi phương pháp.

### 6.1 Data contracts

```python
Route = tuple[int, ...]

# dataclasses dự kiến; array shapes được validate khi khởi tạo/load
City(coords: ndarray, adjacency: ndarray, edge_minutes: ndarray,
     demand: ndarray, instance_id: str, metadata: dict)
ProblemConfig(k: int, min_stops: int, max_stops: int,
              budget_minutes: float, headway_minutes: float,
              transfer_penalty_minutes: float, max_transfers: int,
              alpha: float, unserved_weight: float, edge_max_minutes: float)
Instance(city: City, candidates: tuple[Route, ...], config: ProblemConfig)
Evaluation(objective: float, passenger_cost: float, operator_cost: float,
           unserved_share: float, served_mean_minutes: float | None,
           mean_transfers: float | None, route_minutes: float,
           direct_share: float, one_transfer_share: float,
           two_transfer_share: float)
```

`ndarray` ở contract chỉ `numpy.ndarray`. Evaluation transfer shares có mẫu số Q cho mọi OD; `served_mean_minutes` và `mean_transfers` có mẫu số demand phục vụ, bằng None khi không ai được phục vụ. Tie-break của path quyết định transfer category. Tổng direct+one+two+unserved=1 trong tolerance.

```python
generate_city(seed: int, n: int, graph_family: str,
              demand_mode: str, hotspot_factor: float = 4.0) -> City
build_candidates(city: City, config: ProblemConfig) -> tuple[Route, ...]
save_instance(instance: Instance, directory: Path) -> None
load_instance(directory: Path) -> Instance
passenger_paths(instance: Instance, selected: tuple[int, ...]) -> PathResult
evaluate(instance: Instance, selected: tuple[int, ...]) -> Evaluation
action_mask(instance: Instance, selected: tuple[int, ...]) -> ndarray
make_observation(instance: Instance, selected: tuple[int, ...]) -> dict[str, ndarray]
solve_random(instance: Instance, seed: int) -> tuple[int, ...]
solve_greedy(instance: Instance) -> tuple[int, ...]
solve_local_search(instance: Instance, max_evaluations: int) -> tuple[int, ...]
solve_exact(instance: Instance) -> tuple[int, ...]
```

`PathResult` chứa cost_minutes[N,N], transfers[N,N], served[N,N]; diagonal cost=0, served=False, transfers=0 và bị loại khỏi thống kê. Unserved ngoài diagonal cost=inf, transfers=−1 nội bộ. `evaluate` chấp nhận mạng partial để tính reward; final validator tách riêng yêu cầu đúng K. `action_mask` trả bool[512]; terminal trả toàn false nhưng caller không được sample nữa.

Env `NetworkDesignEnv(instances, reward_mode="dense")`:

- `reset(seed=None, options=None) -> (obs, info)`; `options={"instance_index": i}` chọn xác định.
- `step(action) -> (obs, reward, terminated, truncated, info)`; info chứa selected IDs và từng thành phần Evaluation.
- `action_masks() -> ndarray` gọi mask thuần; phù hợp MaskablePPO.
- `action_space=Discrete(512)`; `observation_space` khớp §4.2.

File instance: `city.npz` không pickle + `instance.json` chứa config/metadata/routes. JSON lưu schema version, hashes và route IDs; path mặc định tương đối workspace. Checkpoint chỉ load từ artifact tin cậy của dự án.

## 7. Baselines và đánh giá

### Baselines bắt buộc

1. Random uniform trên mask; 10 rollouts/instance, báo mean và best-of-10 riêng.
2. Greedy: tại mỗi bước chọn action giảm J nhiều nhất trên mask, tie theo ID.
3. Greedy + local search: từ mạng greedy, xét thay một tuyến bằng một ứng viên chưa chọn; chỉ nhận swap hợp lệ giảm J > 1e-9. Best improvement, tối đa 1,000 lần gọi evaluator, dừng khi không cải thiện.
4. Exact enumeration cho M≤12, K≤3: xét mọi subset đúng K trong budget, lấy min J. Đây là optimum **trong candidate pool**.

Baseline và RL dùng cùng config, evaluator, pool, test instances. RL inference mặc định greedy argmax logits sau mask. Best-of-10 RL là experiment riêng, báo chi phí sampling; không so ngầm với một lần chạy baseline.

### Metrics và statistics

- Primary: mean final J trên test_id; lower is better.
- Quality: demand-weighted coverage=1−U; served_mean_minutes; generalized-time phân vị có trọng số; direct/one/two transfer shares; tổng route_minutes.
- Robustness: invalid network rate; tỷ lệ lỗi rollout; OOD degradation.
- Compute: training wall time; candidate generation time; evaluator calls; inference wall time cả evaluator và policy. CPU/GPU và caching regime phải ghi rõ.
- So sánh paired theo instance; với 3 training seeds, báo riêng từng seed và mean/std giữa seed. CI paired bootstrap theo instance sau khi lấy mean model-seed cho từng instance, nhãn CI phản ánh sampling instance, không giả vờ 3×N điểm độc lập.
- Checkpoint tốt nhất theo mean validation J trên 200 instance; tie lấy checkpoint sớm hơn. Không dùng test để chọn seed/checkpoint.

Reward experiments: dense và terminal-only tại α=0.7, λ_u=2; thêm α∈{0.3,0.9} và λ_u=0 khi budget cho phép. Mỗi cấu hình đổi objective phải train/evaluate có nhãn riêng; so raw metrics và cùng objective tham chiếu, không so trực tiếp J khác trọng số rồi kết luận tốt hơn.

## 8. Reproducibility, hiệu năng và kiểm tra

Run artifact phải có resolved config, data/pool hashes, git commit/dirty status, dependency lock hash, package versions, seed, device, elapsed time, checkpoint metadata, learning CSV và evaluation CSV. Full run lưu model cuối và model tốt nhất; rerun evaluation phải dùng đúng encoding/config.

MVP chỉ hỗ trợ fresh training và load checkpoint để inference; không resume training. Output directory đã tồn tại bị từ chối để bảo toàn run cũ.

Cache evaluator theo `(instance_hash, sorted(selected_ids), evaluator_version, config_hash)`. Không cache chỉ theo số tuyến. Policy permutation hoặc augmentation không được làm lệch ID/cache.

Profile serialized observations và rollout buffer trước full run; không materialize tensor [M,N,N] cho từng batch khi sequence [M,L_max] đã đủ lưu tuyến. Thay encoding phải tăng spec version và chạy lại observation/model tests.

Acceptance bắt buộc:

- Fixture đi thẳng=13, chuyển tuyến=21 và unserved đúng theo §3.2.
- Mọi reward hữu hạn; telescope đúng tolerance 1e-6; không reward terminal kép.
- Mask bảo đảm completion, không all-false trước terminal với instance hợp lệ.
- Routes valid, budget đúng, đủ K ở 1,000 rollout test ngẫu nhiên.
- Tiny exact không kém mọi baseline; RL so với oracle với gap được báo cáo.
- Hai lần generation cùng seed/config cho cùng dữ liệu và hash.
- Model save/load cho cùng greedy action trên cùng obs/mask.
- Pilot hoàn thành, có profiler output và metrics; full training ít nhất 3 seeds.
- Báo cáo test/ablation có nguồn artifact và kết luận trung thực, kể cả khi RL thua baseline.

## 9. Định nghĩa hoàn thành và thay đổi thiết kế

Hoàn thành MVP khi tất cả acceptance trên chạy được bằng CLI, dữ liệu/model/report tái lập theo manifest và người đọc có thể lần từ config đến kết quả. Không dùng “reward tăng” làm bằng chứng duy nhất.

Nếu cần thay encoding, candidate generator, objective, headway hoặc transfer semantics: tăng config/evaluator/schema version tương ứng, cập nhật spec và lưu lineage. Không ghi đè kết quả cũ dưới cùng experiment ID.

Kế hoạch triển khai cụ thể, file và lệnh kiểm chứng nằm trong [plan.md](plan.md). Không bắt đầu mở rộng GNN hoặc mạng đường thật trước khi qua gate evaluator, baseline và pilot.
