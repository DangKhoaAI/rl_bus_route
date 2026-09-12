# Spec v0.2: Dynamic Bus Fleet Reallocation and Frequency Control

Ngày: **2026-09-12**. Trạng thái: đặc tả dự kiến; chưa implement hoặc train.

Định nghĩa mới của người dùng thay thế hoàn toàn mục tiêu chọn K tuyến từ candidate pool. Bản cũ nằm trong commit `a806bb6`. Xem [research](docs/research.md) và [plan](plan.md).

## 1. Mục tiêu

Trên một mạng tuyến bus có sẵn, xây controller RL phân bổ đội xe hữu hạn và điều chỉnh dịch vụ theo nhu cầu hành khách biến động. Controller giảm thời gian chờ và khách không được phục vụ, cân bằng với chi phí vận hành và chất lượng từng tuyến.

Đầu ra là **policy vận hành** và chuỗi nhiệm vụ xe theo thời gian; route geometry không phải biến quyết định. “Frequency control” là thay mục tiêu xuất bến và cấp xe thực hiện, không tăng năng lực bằng một biến số độc lập với fleet.

### 1.1 Phạm vi hoàn chỉnh

- Ba tuyến hai chiều có sẵn, một depot, đội xe hữu hạn có sức chứa.
- Demand theo thời gian, boarding/alighting, queue FIFO, travel time biến động.
- Dispatch reserve, headway target, reassign xe rỗng tại bến, recall và short-turn được định nghĩa trước.
- Synthetic scenarios; simulator 30 giây; RL control mỗi 2 phút.
- Maskable PPO, baselines vận hành, multi-seed evaluation, action/reward ablations.
- Forecast lịch sử đơn giản là experiment thêm sau core; không bắt buộc deep forecasting model.

### 1.2 Các mốc có thể chạy độc lập

| Mốc | Khả năng |
|---|---|
| M1 | Simulator + dispatch reserve + headway target + recall |
| M2 | M1 + reassign giữa các tuyến tại bến hợp lệ |
| M3 | M2 + short-turn mission; đây là core MVP hoàn chỉnh |

M1/M2 chưa phải hoàn thành toàn bộ phạm vi. Core training cuối dùng M3. Các action chưa bật vẫn giữ slot trong action table nhưng luôn bị mask.

Feature flags: M1 `enable_reassign=false, enable_short_turn=false`; M2 `true,false`; M3 `true,true`. Ablation no-reassign tại M3 dùng `false,true`, không đồng nhất với M1. Recall/dispatch/headway luôn bật trong cả ba mốc.

### 1.3 Ngoài phạm vi

Không thiết kế tuyến mới; không đổi hành trình xe đang chở khách; không điều khiển metro. Metro/trường học chỉ là nguồn arrivals. Không passenger transfer giữa tuyến, dynamic route choice, driver rostering, ca nghỉ lao động, breakdown repair, tín hiệu giao thông hay congestion nội sinh. Tất cả tài xế sẵn sàng trong 4 giờ; compatibility khai báo trong dữ liệu. Không web app, deployment hoặc real-city claim.

## 2. Mặc định để bắt đầu

| Tham số | Giá trị |
|---|---|
| Ngôn ngữ/runtime | Python 3.11, Linux, CPU smoke bắt buộc |
| R tuyến / số trạm mỗi tuyến | 3 / 6, hai chiều |
| F xe | 12: 3 xe được gán mỗi tuyến + 3 reserve tại depot |
| Giới hạn encoding | R_max=4, F_max=16, S_max=8 |
| Capacity / comfort | 40 / 30 người mỗi xe |
| Simulation tick δ | 30 giây |
| Control interval Δ | 120 giây = 4 ticks |
| Horizon H | 240 phút = 120 decisions |
| Demand window | [0,180) phút; [180,240) không sinh khách mới |
| Edge travel nền | 180 giây mỗi cặp trạm liền kề |
| Intermediate dwell | 30 giây mỗi lần dừng, cố định |
| Terminal/turnpoint layover | 120 giây, không bỏ qua |
| Headway target ban đầu | 15 phút mỗi hướng |
| Headway target có thể chọn | 6, 10, 15 phút; áp dụng cả hai hướng tuyến |
| Minimum departure spacing | 2 phút cùng tuyến/hướng/điểm xuất phát |
| Service guard headway | 20 phút, dùng dự báo khả thi và báo vi phạm thực tế |
| Fleet floor | 2 xe cam kết FULL trên mỗi tuyến |
| Allocation cooldown | 20 phút/xe sau dispatch/reassign/recall/short-turn |
| Headway-change cooldown | 10 phút/tuyến |
| Passenger patience / excessive-wait threshold | 45 / 15 phút |
| Discount | γ=1.0 cho horizon hữu hạn |
| Reward normalization | N_ref=3,000 hành khách tham chiếu, cố định |

Các giá trị là đề xuất cấu hình, chưa hiệu chỉnh bằng dữ liệu thật. Tất cả thời gian nội bộ là integer giây; cost/report đổi sang phút. Đồng hồ/timers luôn bội δ. Không clip queue hoặc bỏ khách vì tensor đầy; simulator lưu số lượng không giới hạn theo integer/cohort, observation scale không đổi conservation.

## 3. Network, fleet và passenger state

### 3.1 Network

Route r có ordered stops `(s0,s1,...,s5)`. Hướng `+` đi index tăng; `−` đi index giảm. Mỗi xe FULL đến endpoint, khách xuống hết, layover rồi đủ điều kiện chạy hướng ngược lại.

Base fixture: A dùng node 0–5, B dùng 6–11, C dùng 12–17, depot=18. Cạnh dọc tuyến có base-time 180s; depot nối hai endpoint của mỗi tuyến bằng cạnh 360s. Deadhead dùng weighted shortest path trên graph này và traffic field. Tuyến không chia sẻ hành khách. Cấu hình mới có thể dùng hub chung với travel=0 cho cùng địa điểm thật; tuyệt đối không đặt travel=0 giữa hai bến khác nhau.

Short-turn của mỗi tuyến: `s0→s1→s2→s3→s2→s1→s0`, có turnpoint cho phép tại s3 và layover 120s. Chỉ định mission trước khi xuất phát. Sau khi hoàn thành và layover ở s0, xe trở lại mode FULL trên tuyến đó.

### 3.2 Vehicle state machine

Mỗi xe có ID cố định, capacity, route assignment hoặc depot, pattern, direction, position/edge, remaining travel/dwell, next-stop, trip destination limit, destination passenger cohorts, ready time và cooldown.

Các phase loại trừ nhau:

```text
DEPOT_IDLE -> DEADHEAD -> TERMINAL_IDLE -> SERVICE_MOVING
                                         ^                  |
                                         |                  v
                                      LAYOVER <- SERVICE_DWELL
```

SERVICE_DWELL chỉ ở trạm giữa; endpoint/turnpoint chuyển qua LAYOVER. DEADHEAD không chở khách. TERMINAL_IDLE là xe đã hoàn tất layover và rỗng; đây mới là trạng thái có thể reassign/recall. Depot reserve cũng phải hết cooldown mới được điều đi.

Dispatch/reassign ghi target route ngay để tránh cấp nhiệm vụ trùng, nhưng xe DEADHEAD incoming chưa tính vào floor xe FULL đang bảo vệ tuyến. Khi đến target s0, xe trở thành extra-departure pending; chỉ chở khách sau khi dispatcher thực hiện departure thật.

### 3.3 Passenger records/cohorts

Mỗi passenger hoặc cohort cùng đặc điểm có `(id, route, direction, origin, destination, arrival_tick, count, first_denied_flag, status)`, kèm boarding/completion/abandonment tick khi xảy ra. Destination phải nằm phía trước origin theo direction. Status: WAITING, ONBOARD, COMPLETED, ABANDONED.

Cohort có thể split khi capacity còn ít hơn count; mọi phần giữ lineage và first_denied_flag. Simulator biết destination để boarding đúng hành trình; controller aggregate không được tự xem toàn bộ destination tương lai/ẩn.

Board FIFO theo `(arrival_tick, passenger_id)` trong số khách đủ điều kiện. Người không đủ điều kiện đi short-turn vẫn chờ FULL và không được đánh dấu denied-capacity. Ở turnpoint chỉ đón khách theo chiều quay về. Xe không nhận người có đích ngoài short-turn pattern; không ép người xuống giữa đường.

Waiting age ≥45 phút thì abandon trước boarding tại cùng tick. Boarding/alighting là sự kiện một lần; còn trong queue thì waiting vẫn tiếp tục sau denied.

### 3.4 Conservation

Ở mỗi tick:

\[
N_{generated}=N_{waiting}+N_{onboard}+N_{completed}+N_{abandoned}.
\]

\[
F=N_{depot}+N_{deadhead}+N_{terminal}+N_{moving}+N_{dwell}+N_{layover}.
\]

\(0\le L_b\le40\), tất cả counts là integer không âm. Xe không có hai nhiệm vụ đồng thời; cohort không ở cả queue và xe.

## 4. Simulation timing và dispatcher

### 4.1 Thứ tự sự kiện xác định

Tại boundary t:

1. Hoàn thành movement/dwell/layover hết hạn; alight đúng destination; cập nhật ready state.
2. Sinh arrivals tại t nếu t<180 phút; đánh dấu abandonment đủ patience.
3. Nếu là control boundary, tạo observation/mask, nhận đúng một action; không nhìn arrivals sau t.
4. Xử lý service boarding và autonomous terminal dispatcher; resolve tie theo vehicle ID. Không xử lý một visit/departure hai lần.
5. Tích phân costs trên `[t,t+δ)` từ trạng thái sau sự kiện; timers tiến δ.

`step(action)` thực hiện 4 ticks, đến observation boundary kế tiếp sau bước 1–2, trước bước 3–4. `reset` tạo boundary t=0 cùng quy ước. Ở t=H xử lý completion/abandonment đến hạn, không sinh arrivals hoặc departure mới, rồi terminal settlement. Các counter event ở đúng H thuộc step cuối, không bị mất.

Travel duration khi vào edge được lấy từ traffic field tại entry time, làm tròn lên bội δ, tối thiểu δ; giữ nguyên đến cuối edge. Không resample mỗi tick làm xe không bao giờ đến. Intermediate dwell cố định, không mô hình hóa thời gian boarding theo số người trong MVP. Sau deadhead xe có thể ready ngay tại bến; layover 120s bắt buộc sau service leg/short-turn leg, không cộng thêm vào deadhead arrival.

Terminal boarding diễn ra tại thời điểm departure; layover trước đó bao gồm recovery/terminal service. Intermediate boarding tại lúc đến sau alight/arrivals rồi dwell 30s. Đếm headway bằng actual departure timestamp, không bằng thời điểm agent đặt target.

### 4.2 Autonomous dispatcher

Normal operation luôn tiếp tục khi agent chọn NOOP. Ở mỗi terminal/hướng:

- Ưu tiên extra-departure pending từ injection/reassign/short mission.
- Sau đó xét xe FULL ready nếu gap từ last FULL departure ≥ headway target.
- Mọi departure phải giữ spacing ≥2 phút từ last ANY departure cùng tuyến/hướng/terminal. Short mission cũng chiếm spacing.
- Chỉ một xe mỗi departure opportunity; tie theo ID. Xe không có mặt/đang layover không thể xuất bến.
- Full departure cập nhật last_FULL và last_ANY; short chỉ last_ANY. Target là mong muốn, không bảo đảm thực hiện nếu fleet không đủ.

Horizon finite có departure lịch sử ban đầu `last_FULL=last_ANY=−900s` ở các endpoint. Ba xe mỗi tuyến: hai ở s0, một ở s5, đều empty/ready; ba reserve ở depot. Không có khách đầu ca. Peak bắt đầu sau khoảng startup, nhưng metrics vẫn tính cả ca cho mọi phương pháp.

Fleet/headway sanity check dùng cycle gồm travel+dwell+layover. Không dùng \(C/h\) làm năng lực cả mạng: đó chỉ là xấp xỉ năng lực qua một điểm/hướng và bỏ qua quay vòng/tải theo đoạn.

## 5. Action space, constraints và service protection

### 5.1 Action table tĩnh

Một quyết định mỗi 120s; không cho đồng thời dispatch hai xe trong một action. Muốn điều hai xe phải dùng hai bước.

| Action | Hiệu ứng |
|---|---|
| `NOOP` | Giữ kế hoạch, dispatcher vẫn chạy |
| `DISPATCH(b,r)` | Reserve b ở depot deadhead tới r.s0; first FULL departure là extra |
| `SET_HEADWAY(r,h)` | Cập nhật target 6/10/15 phút cho hai hướng, không sinh xe |
| `REASSIGN(b,r)` | Xe rỗng ready ở terminal tuyến khác deadhead tới r.s0, extra FULL đầu tiên |
| `RECALL(b)` | Xe rỗng ready quay depot, bỏ assignment khi rời tuyến |
| `SHORT_TURN(b,r)` | Reserve từ depot hoặc xe rỗng FULL ready tại r.s0 nhận một short mission; sau đó quay về FULL |

Không có HOLD mid-route, arbitrary reroute hoặc đổi short pattern đang chạy. Những điều này cần extension spec.

Encode Discrete(221): NOOP=0; 64 slots DISPATCH b-major/r-minor; 64 REASSIGN; 64 SHORT_TURN; 16 RECALL; 12 SET_HEADWAY r-major/choice-minor, với F_max=16,R_max=4. Entity padding luôn masked; ordering lưu trong checkpoint schema.

### 5.2 Mask

Luôn có NOOP hợp lệ trước terminal. Mask action khi sai phase, xe còn khách, cooldown, route compatibility, pattern entry/turnpoint, entity tồn tại, hoặc feature chưa bật. SET_HEADWAY đang bằng target hiện tại hoặc trong 10-minute cooldown bị mask. Target thấp nhưng hiện thiếu xe vẫn có thể yêu cầu; actual service không được giả vờ đáp ứng.

DISPATCH chỉ từ DEPOT_IDLE. REASSIGN chỉ giữa hai tuyến khác nhau. RECALL/REASSIGN/SHORT lấy xe từ tuyến phải thỏa **donor guard** sau khi loại xe đó khỏi nguồn lực FULL:

1. Còn ít nhất 2 xe FULL committed, không tính incoming deadhead, short mission, reserve hoặc xe đang bị chuyển đi.
2. Còn một xe FULL empty/ready khác tại đúng donor terminal; earliest feasible departure theo spacing không muộn hơn `last_FULL + 20 phút`.

Guard bảo vệ nguồn lực và departure sắp tới; không chứng minh queue không overload hoặc mọi future headway ≤20 phút dưới traffic ngẫu nhiên. Luôn báo actual violations. Baselines và RL dùng cùng guard, không cho RL hưởng helper riêng.

Action request không hợp lệ gây `ValueError` khi API dùng sai, không sửa ngầm thành NOOP. Không teleport khi bắt đầu deadhead. Cooldown đặt tại action acceptance; không chặn completion/scheduler bình thường.

## 6. Mathematical foundation và reward

### 6.1 Dynamics

Cho Q_{r,d,i,j}(t) là số người chờ, L_b(t) là load xe. Với arrivals A, boarding B và abandonment E trong một tick:

\[
Q(t+\delta)=Q(t)+A(t)-B(t)-E(t).
\]

Với alighting D_b và boarding B_b:

\[
L_b(t+\delta)=L_b(t)-D_b(t)+B_b(t),\qquad 0\le L_b\le C_b.
\]

World state S_t gồm toàn bộ cohorts, xe/timers, assignments, dispatcher clocks, target/cooldown và latent demand/traffic state. Dynamics stochastic từ exogenous process. Action làm thay đổi cả năng lực hiện tại và vị trí nguồn lực tương lai.

### 6.2 Observation và POMDP

Controller nhận o_t=O(S_t), gồm queue counts/age summaries, load/position/ETA, boarding/alighting và arrivals quá khứ, actual headways, reserve/assignment/cooldown và thời gian còn lại. Demand latent và destination chi tiết của người đang chờ không được lộ.

Vì o_t không mô tả đầy đủ S_t, đây là **partial-observation control**. Policy baseline là \(\pi(a_t\mid o_t,\mathrm{recent\ history})\); không tuyên bố aggregate features thỏa Markov đầy đủ. Giả định queue/load sensing hoàn hảo ở MVP; noisy sensing là OOD extension.

Với state đầy đủ, Bellman hữu hạn:

\[
V_t^\pi(s)=\mathbb E[r_t+\gamma V_{t+1}^\pi(S_{t+1})\mid S_t=s],\quad V_T=0.
\]

PPO critic trong triển khai ước lượng từ observation/history, nên có thể chịu sai số do thông tin ẩn. Remaining time xuất hiện trong observation để phân biệt đầu/cuối ca.

### 6.3 Cost theo interval

Mọi tích phân dưới đây tính bằng phút trên interval control. W: tổng queue passenger-minutes; V: tổng onboard passenger-minutes; O: excess comfort passenger-minutes; C: bus-minutes ngoài depot; C_D: deadhead bus-minutes; F: passenger-minutes của người đã chờ ≥15 phút.

\[
W_t=\int_t^{t+\Delta}\sum_{r,d,i,j} Q_{r,d,i,j}(u)\,du,
\qquad V_t=\int_t^{t+\Delta}\sum_b L_b(u)\,du,
\]

\[
O_t=\int_t^{t+\Delta}\sum_b\max(0,L_b(u)-30)\,du,
\qquad F_t=\int_t^{t+\Delta}\sum_{p\in waiting(u)}\mathbf1[age_p(u)\ge15]\,du.
\]

U_t là số người **lần đầu** bị capacity-denied trong interval; E_t là số người abandon; M_t là số mission-change actions được nhận (dispatch/reassign/recall/short). Không đếm NOOP/headway update như một mission. Không đếm short-ineligible như denied. Mỗi người có thể first-denied một lần rồi abandon sau đó: hai penalty mô tả hai hậu quả khác nhau, không lặp cùng event.

Default cost (đơn vị passenger-minute equivalent):

\[
c_t=W_t+0.25V_t+0.5O_t+0.5C_t+0.5C_{D,t}+F_t+5U_t+60E_t+2M_t.
\]

Deadhead chịu base operating cost và phụ phí riêng có chủ đích. C bao gồm terminal idle/layover đã gán tuyến, không chỉ lúc xe chuyển động; recall mới giải phóng nguồn lực về depot. Đây là proxy, không phải chi phí tài chính thật.

Reward:

\[
\boxed{r_t=-c_t/N_{ref}},\qquad N_{ref}=3000.
\]

Ở bước cuối cộng thêm `−60*(N_waiting(H)+N_onboard(H))/N_ref`. Người đã abandon không có mặt trong settlement. Không xóa khách để tránh penalty. Chi phí waiting/ride đã phát sinh vẫn giữ nguyên; settlement là phí chưa hoàn thành, không phải tính lại cùng thời gian.

Với γ=1:

\[
\sum_t r_t=-\frac{\sum_t c_t+60N_{unfinished}(H)}{N_{ref}}.
\]

Không dùng reward telescope của v0.1, không thưởng “đã đón khách” độc lập. Trọng số là điểm xuất phát phải ablate; floor/guard là hard constraints, reward fairness không thay thế chúng.

### 6.4 Termination

H=240 phút là terminal hữu hạn thuộc định nghĩa nhiệm vụ: `terminated=True`, `truncated=False`, bootstrap V_terminal=0. Arrivals dừng ở phút 180 nhưng controller vẫn hoạt động đến H để phục vụ phần còn lại. Simulator không auto-clear passengers; logging final residual/abandoned/completed bắt buộc.

Giới hạn ngoài nhiệm vụ khi debug là truncation và phải ghi lý do; không đưa episode bị cắt này vào quality comparison như đã hoàn thành. Gymnasium phân biệt hai trường hợp [R8](https://gymnasium.farama.org/main/tutorials/handling_time_limits/).

## 7. Synthetic scenarios và forecast

### 7.1 Demand/traffic

Mỗi scenario-day gồm network, fleet initial state, demand tape và exogenous traffic field. Poisson count mỗi stop/direction/tick với λ ở đơn vị khách/phút, nhân δ/60. Baseline tổng route demand khoảng A=180, B=160, C=140 khách/giờ cho cả hai hướng; tăng cường Gaussian peak ở một tuyến với multiplier 1.5–3.0, peak center lấy trong [45,135] phút, width 15–30 phút. Route rates chia cho direction/origin bằng probability vectors tổng 1. Destination được sample ở downstream với xác suất tỷ lệ `exp(-abs(j-i)/2)`; không sinh OD đi ngược hướng hoặc origin=end-of-direction. Baselines có thể dùng prior khoảng cách đã khai báo này để ước lượng short-turn eligibility, không xem destination thực của người đang chờ.

Base noise: edge/time-bucket multiplier lognormal log-std=0.15, mean=1, chặn [0.7,2.5], bucket=5 phút. Base route edge-time khác nhau trong [150,210] giây cho ngày khác; depot connectors [300,480] giây. Round up sau khi nhân traffic. Không resample travel cho cùng edge/time trong cùng scenario.

OOD burst: arrivals theo nhóm tại một trạm trong 5–10 phút, không chỉ tăng đều λ. OOD traffic: hành lang một tuyến multiplier 1.5–2.0 trong 30 phút; đây là chậm xe, không breakdown. Stress thiếu fleet đổi F bằng config riêng, không dùng checkpoint encoding khác.

Pre-generate demand theo scenario_seed độc lập action. Traffic được indexed `(scenario_seed, edge_id, time_bucket)`; policy đi edge khác thời điểm khác có travel khác một cách nhân quả, nhưng các policy chia sẻ cùng field. Tách RNG của policy khỏi RNG môi trường. Không expose future tape hoặc seed có thể tra future trong observation/info dùng bởi controller.

### 7.2 Split

| Split | Số ngày | Nội dung |
|---|---:|---|
| train | 500 | Base peak/traffic distributions |
| validation | 100 | Ngày mới, cùng phân phối |
| test_id | 200 | Ngày mới, cùng phân phối |
| test_ood_burst | 200 | Batch peak; tổng demand được báo riêng |
| test_ood_traffic | 200 | Traffic disruption |

Seeds gốc lần lượt 1001/2001/3001/4001/5001; training algorithm seeds=11/22/33. Cùng base network giữa các ngày là có chủ đích vì controller vận hành mạng cố định; đây là tổng quát theo ngày/demand, **không tuyên bố tổng quát topology**. Scenario/tape hash không trùng giữa split. Biến thể của cùng ngày phải nằm cùng split. Không chọn baseline thresholds hoặc checkpoint trên test.

### 7.3 Forecast optional

Forecaster chỉ dùng train-day arrival logs và history ≤t. Baseline: time-bin mean của train ở cùng stop/direction cho 15 phút kế tiếp, nhân correction `clip((recent_10min+1)/(historical_10min+1),0.5,2.0)`. Không dùng waiting Q làm nhãn arrivals vì Q phụ thuộc control. Fit một lần trên train, freeze ở validation/test. Báo MAE/bias, forecast/no-forecast cùng controller architecture/masks/budget. Full future tape chỉ được dùng cho oracle diagnostic riêng nếu bổ sung sau.

## 8. Model, PPO và observation encoding

### 8.1 Feature contract

Fixed Dict float32, padding=0 có entity masks. Node/route IDs ổn định. Chia count/load theo 40, age/time theo H hoặc ngưỡng được nêu; giá trị count có thể >1, không clip mất overload. `obs_version=2`.

- `stops [4,2,8,7]`: queue count/40, mean age/2700s, max age/2700s, count age≥900s/40, boarding last Δ/40, alighting last Δ/40, arrivals last Δ/40.
- `arrival_history [4,2,8,5]`: counts 5 control bins vừa qua/40, chỉ thời gian đã xảy ra.
- `forecast [4,2,8]`: expected next-15-min arrivals/40; zero khi disabled và có forecast-enabled flag.
- `vehicles [16,27]`: phase one-hot 6; route one-hot 5 gồm depot; pattern one-hot 2; direction one-hot 2; current/next node IDs scaled theo số node; target-route one-hot 5 gồm none; 5 scalar load/40, phase_remaining/H, ready_remaining/H, cooldown_remaining/1200s, nominal time-to-terminal/H. Node khi không áp dụng dùng 0; ID thật encode (id+1)/(node_count+1).
- `routes [4,8]`: headway target/1200s, actual full departure gap hai hướng/1200s, FULL committed count/16, incoming count/16, short count/16, target cooldown/600s, reserve-compatible count/16.
- `stop_valid [4,2,8]`, `vehicle_valid [16]`, `route_valid [4]`, `context [3]`: time/H, remaining/H, forecast-enabled.

27 vehicle features =6+5+2+2+2+5+5. SERVICE_MOVING/SERVICE_DWELL tách phase; 6 phase đúng §3.4. Queue destination distribution không đưa vào observation; mask có thể lộ điều kiện hành động hợp lệ nhưng không future demand. Bus load theo tổng, destination cohorts chỉ simulator dùng. Không claim feature đủ Markov.

### 8.2 Architecture

MVP dùng MaskablePPO `MultiInputPolicy`, flatten các Dict tensors → shared MLP [256,128], actor/critic mỗi nhánh [128], actor logits 221, critic scalar. Custom feature extractor chỉ cần normalize/flatten/mask padding đúng; không GNN và không recurrent trong core. Static route geometry cố định; flatten IDs hạn chế generalization sang mạng mới.

Masked policy: logits invalid=−∞ trước categorical; valid actions được normalize lại. `NOOP` giữ nonempty support. Train và evaluation phải dùng cùng masks; MaskableEvalCallback/evaluate hoặc runner tự truyền mask. Không coi ordinary unmasked evaluate là hợp lệ.

PPO dùng:

\[
\rho_t=\pi_\theta(a_t|o_t)/\pi_{old}(a_t|o_t),
\quad L^{clip}=\mathbb E[\min(\rho_t\hat A_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t)].
\]

Loss minimize `−Lclip + c_v*MSE(value,return) − c_H*entropy`; GAE với γ=1, λ=0.95, terminal value=0. Observation/history critic là approximation trong POMDP. Xem [PPO](https://arxiv.org/abs/1707.06347), [masking](https://arxiv.org/abs/2006.14171).

### 8.3 Training defaults

| Tham số | Giá trị |
|---|---|
| lr / clip / GAE λ | 3e-4 / 0.2 / 0.95 |
| n_envs / n_steps mỗi env | 4 / 256 |
| batch_size / n_epochs | 256 / 4 |
| entropy / value coeff / max_grad_norm | 0.01 / 0.5 / 0.5 |
| pilot / full transitions mỗi seed | 12,288 / 245,760 |
| validation interval | 12,288 tổng env transitions |
| gamma / seeds | 1.0 / 11,22,33 |

Transitions là control decisions, không micro-ticks hoặc passengers. Horizon=120, nên full budget tương đương 2,048 episode nếu chia hết theo episode, nhưng vector rollout có thể kết thúc giữa episode; log completed episodes thực tế. Chọn best mean validation cost, tie lấy checkpoint sớm. Training samples scenario days có replacement. Không hứa hội tụ; profile simulator trước full run.

## 9. Code architecture và data contracts

Python 3.11, `src/bus_rl`, TOML config qua `tomllib`. Dependencies: NumPy, NetworkX (deadhead graph), PyTorch, Gymnasium, stable-baselines3, sb3-contrib, pandas, Matplotlib; pytest/Ruff dev; uv lock. Không bắt buộc SimPy/SUMO vì fixed-tick engine đủ cho scope. CPU smoke trước chọn CUDA.

```text
src/bus_rl/
  domain.py                       # immutable network/config và runtime dataclasses
  data/{scenario,io}.py            # generation, tapes, manifests
  sim/{engine,passengers,vehicles,dispatcher,travel}.py
  control/{actions,guards}.py
  env/{observation,bus_dispatch}.py
  rewards/costs.py
  baselines/{fixed,threshold,proportional,random}.py
  forecasting/historical.py
  models/features.py             # shared MLP feature extractor
  training/{train,callbacks,checkpoint}.py
  evaluation/{runner,statistics,plots,profile}.py
  cli.py
configs/{base,pilot,train,eval}.toml
configs/experiments/
tests/{fixtures,test_data,test_passengers,test_vehicles,test_engine,
       test_control,test_rewards,test_env,test_baselines,test_model,
       test_forecast,test_pipeline}.py
data/{generated,manifests}/
runs/<run_id>/
reports/<experiment_id>/
```

Dependency flow: domain → data/sim → control/rewards → env → training/evaluation/CLI. Baselines gọi cùng action API, không sửa simulator state để có advantage. Forecast không đọc future tape. Cost calculator không quyết định hành động.

Core dataclasses: `Network`, `Route`, `Vehicle`, `PassengerCohort`, `Scenario`, `SimConfig`, `WorldState`, `Action`, `StepCosts`, `EpisodeMetrics`. Scenario chứa network/initial fleet/exogenous tapes/config; WorldState là mutable runtime riêng cho mỗi reset, không sửa Scenario.

SimConfig chia namespace `physical`, `control`, `reward`; config train thêm namespace `algorithm`. Physical gồm mạng, fleet, tick, horizon, capacity và tapes. Control gồm feature flags/guards/cooldowns; reward gồm hệ số §6.3. Scenario hash chỉ phụ thuộc physical data và generation config. Run có physical_hash, control_hash, reward_hash riêng; env chỉ được override control/reward từ run config, không âm thầm đổi physical khi load dataset. Nhờ đó action/reward ablations dùng đúng cùng tapes, còn khác biệt controller được truy vết rõ.

```python
generate_scenario(seed: int, config: SimConfig) -> Scenario
save_scenario(scenario: Scenario, directory: Path) -> None
load_scenario(directory: Path) -> Scenario
initial_state(scenario: Scenario) -> WorldState
advance_interval(state: WorldState, scenario: Scenario, action: Action) -> StepCosts
build_action_table() -> tuple[Action, ...]  # 221 slots
valid_action_mask(state: WorldState, config: SimConfig) -> ndarray  # bool[221]
observe(state: WorldState, history: ObservationHistory,
        forecast: Forecast | None) -> dict[str, ndarray]
interval_cost(costs: StepCosts, config: SimConfig) -> float
summarize_episode(state: WorldState) -> EpisodeMetrics
```

`advance_interval` mutate state đúng Δ, return **raw component increments** và terminal counts nếu đến H, không reset counters cả episode. `StepCosts`: waiting_pm, onboard_pm, crowding_pm, active_bus_min, deadhead_bus_min, excessive_wait_pm, first_denied_count, abandoned_count, mission_changes, terminal_unfinished_count. `interval_cost` dùng công thức §6, normalization ở env một lần.

`WorldState` chứa current_time_s, vehicle map, queues/cohorts, dispatcher clocks, history bins, cumulative costs, conserved counts và reference read-only tới network/compatibility; `ObservationHistory` chỉ record past sensor aggregates. `Forecast` là tensor expected counts và model/version, không latent parameters. WorldState cung cấp properties `generated_count`, `waiting_count`, `onboard_count`, `completed_count`, `abandoned_count`, `depot_count`; Vehicle cung cấp `load`. Count properties phải đối chiếu với cohort/fleet records, không dùng số cache thiếu cập nhật.

`BusDispatchEnv(scenarios, config, forecaster=None)` implement `reset(seed, options={scenario_index})`, `step(action_index)`, `action_masks()`; action_space Discrete(221). Return obs/reward/terminated/truncated/info theo Gymnasium. Info không chứa future tapes/destinations ẩn cho controller; audit logs lưu riêng.

Scenario storage: JSON metadata/topology/fleet + numeric NPZ arrays, không pickle; schema_version=2. Validate timestamps, direction/destination, nonnegative counts, tick multiples, compatibility, shape bounds. Raw artifacts giữ scenario hash, config hash, code commit/dirty status, uv.lock hash, device, package versions và RNG seed. Fresh training only; checkpoint load cho inference phải kiểm obs/action/config version. Output đã tồn tại bị reject.

## 10. Baselines, evaluation và acceptance

### 10.1 Baselines

- `Fixed`: NOOP mọi bước, dispatcher target ban đầu, fleet assignment cố định; reserve không dùng.
- `Threshold`: urgency bằng queue pressure + age + headway gap; dispatch reserve trước, rồi reassign guarded, short-turn khi ước lượng demand đủ điều kiện từ queue/prior lịch sử chiếm đa số; không đọc destination ẩn. Threshold tune validation.
- `Proportional`: desired fleet chia theo estimated demand, vẫn phải tạo các action cụ thể và đi qua guards/travel. Không fractional bus hoặc chuyển assignment tức thời.
- `Random-valid`: kiểm env, không baseline duy nhất.

Threshold/proportional được dùng cùng action set và thông tin policy; forecast setting ghi riêng. Không dùng actual future arrivals hoặc state ẩn. Log budget/tuning và chi phí compute.

### 10.2 Metrics

Primary objective: total passenger-minute-equivalent cost/3000. Báo riêng:

- Mean/P95 observed waiting của **mọi khách sinh trong demand window**, gồm abandonment và right-censored queue cuối H; pending queue chờ đến H được gắn censor flag. Không gọi giá trị censor này là waiting hoàn chỉnh ngoài H.
- Completed share, abandoned share, unfinished share và unique-denied share.
- Per-route và worst-route mean/P95 wait; excessive-wait share là tỷ lệ người từng chờ ≥15 phút trên mọi người sinh của tuyến, không chỉ số đang chờ ở cuối ca.
- Onboard time, comfort-overload integral, load≤capacity violation count.
- Actual headway distribution theo tuyến/hướng; headway target violation và >20-minute service gap.
- Fleet utilization, active/deadhead bus-min, reserve use, intervention count, switching/cooldown violations.
- Wall time, decisions/s, simulator ticks/s, inference latency CPU/GPU.

Conservation ở đầu ca empty nên denominator demand generated; scenario zero-arrival có mean/share=None và finite costs, không chia zero. Settlement không cộng vào observed waiting, chỉ cost.

### 10.3 Experiments

Core M3 dùng 3 seeds. Bắt buộc ablation no-reassign, no-short-turn, fairness-weight=0 (guards giữ nguyên), mỗi cấu hình train 3 seeds cùng budget. Stage M1/M2 là integration gates và có thể dùng làm action ablation nếu configs khớp. Forecast/no-forecast là experiment bổ sung 3 seeds nếu thực hiện. Không chuyển baseline data/action set để làm đẹp kết quả.

Test paired trên cùng days/tapes; mean mỗi instance qua model seeds trước paired bootstrap theo scenario, 2,000 resamples seed6001. Báo mean/std giữa seeds riêng; CI theo day không đại diện đầy đủ training randomness. Checkpoint/thresholds chọn validation, test frozen.

### 10.4 Acceptance

1. Queue/fleet conservation từng tick, không tải vượt 40, không teleport, không reassignment xe có khách.
2. 10 người chờ 5 phút tạo 50 passenger-minutes; interval splitting không làm đổi tổng.
3. 45 khách chờ + xe rỗng capacity40 → board40, queue5, unique_denied5; không lặp denied nếu không có visit mới và không đếm lại first_denied lần sau.
4. Deadhead 6 phút không thể phục vụ target sau 2 phút; ready sau arrival/layover semantics đúng.
5. Short-turn không nhận khách vượt turnpoint; họ không bị mất khỏi queue hoặc bị tính capacity-denied.
6. Fleet floor/guard/cooldown thực thi cho mọi controller; NOOP luôn hợp lệ trước terminal.
7. Cùng scenario/action trace → cùng event logs/metrics; đổi actions không đổi exogenous tapes.
8. 120 decisions đúng 4 giờ; terminal/truncation/cost settlement đúng, không xóa người cuối ca.
9. Core observation không đọc future tape; thay future tape giữ past không đổi observation/action hiện tại với policy deterministic. Nếu bật forecast, cùng kiểm tra này áp dụng thêm cho forecaster.
10. PPO smoke/save/load qua; pilot có profiler; core+ablations có đủ metrics và artifact lineage.

Hoàn thành là có simulator đúng, baseline mạnh và kết luận tái lập; không yêu cầu RL phải thắng. Hạn chế sensing, driver, traffic và route-choice phải giữ trong final report.
