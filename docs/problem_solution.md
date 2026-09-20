# Problem và Solution: Điều phối đội xe buýt bằng Reinforcement Learning

Bản đầy đủ dùng để giải thích bài toán, các biến và cách dùng MaskablePPO để giải. Bản giải thích nhanh trình bày cùng nội dung ở độ dài khoảng một trang A4. Các giá trị setup dưới đây là bộ mặc định được dùng trong phần giải thích, không phải xác nhận cấu hình của mọi experiment. Ký hiệu được giữ thống nhất giữa định nghĩa và config.

# A. Problem: Điều phối đội xe buýt theo nhu cầu biến động

## 1. Goal

Trên một mạng tuyến xe buýt có sẵn, với số lượng xe và sức chứa hữu hạn, cần quyết định cách phân bổ xe và điều chỉnh dịch vụ theo nhu cầu hành khách thay đổi theo thời gian. Mục tiêu là giảm thời gian chờ, tình trạng quá đông và số hành khách không được phục vụ, đồng thời kiểm soát chi phí vận hành và điều chuyển xe. Các quyết định phải tuân thủ giới hạn sức chứa, thời gian di chuyển và các điều kiện bảo vệ dịch vụ trên từng tuyến. Kết quả cần tìm là một quy tắc điều khiển: từ thông tin hiện có tại mỗi thời điểm, xác định quyết định vận hành phù hợp cho đội xe trong phần thời gian còn lại.

## 2. Các biến và cách xây dựng mô hình bài toán

### 2.1. Nhóm biến mô tả mạng lưới và đội xe

| Ký hiệu | Ý nghĩa |
| --- | --- |
| $G=(V,E)$ | Mạng giao thông gồm các nút $V$ và cạnh $E$ |
| $\mathcal R$ | Tập tuyến buýt có sẵn |
| $R=\lvert\mathcal R\rvert$ | Số tuyến |
| $\mathcal S_r=(s_{r,0},\ldots,s_{r,n_r-1})$ | Dãy trạm theo thứ tự trên tuyến $r$ |
| $n_r$ | Số trạm của tuyến $r$ |
| $d\in\{+1,-1\}$ | Hướng di chuyển trên tuyến |
| $v_{\mathrm{dep}}$ | Vị trí depot |
| $\mathcal B$ | Tập xe buýt |
| $F=\lvert\mathcal B\rvert$ | Tổng số xe |
| $C_b$ | Sức chứa tối đa của xe $b$ |
| $C_b^{\mathrm{comfort}}$ | Ngưỡng tải thoải mái của xe $b$ |
| $\bar\tau_e$ | Thời gian di chuyển cơ sở trên cạnh $e$ |
| $\mathcal P_r^{\mathrm{short}}$ | Hành trình short-turn được phép trên tuyến $r$ |

Các đại lượng này xác định hạ tầng và nguồn lực mà người điều phối có thể sử dụng.

### 2.2. Nhóm biến thời gian và quy tắc vận hành

Dùng $t$ cho thời gian vật lý và $k$ cho bước ra quyết định, với $t_k=k\Delta$.

| Ký hiệu | Ý nghĩa |
| --- | --- |
| $\delta$ | Độ dài một tick mô phỏng |
| $\Delta$ | Khoảng thời gian giữa hai lần ra quyết định |
| $H$ | Tổng thời gian vận hành |
| $K=H/\Delta$ | Số bước ra quyết định |
| $H_D$ | Thời điểm ngừng sinh hành khách mới |
| $\tau^{\mathrm{dwell}}$ | Thời gian dừng tại trạm giữa tuyến |
| $\tau^{\mathrm{layover}}$ | Thời gian nghỉ bắt buộc tại endpoint/turnpoint |
| $\tau^{\mathrm{patience}}$ | Thời gian chờ tối đa trước khi hành khách bỏ cuộc |
| $\tau^{\mathrm{excess}}$ | Ngưỡng xác định chờ quá lâu |
| $\mathcal H$ | Tập headway mục tiêu được phép chọn |
| $h_0$ | Headway mục tiêu ban đầu |
| $h_{\min}$ | Khoảng cách tối thiểu giữa hai lần xuất phát cùng luồng dịch vụ |
| $h_{\mathrm{guard}}$ | Ngưỡng headway dùng trong điều kiện bảo vệ tuyến cho xe đi |
| $F_{\min}$ | Số xe FULL committed tối thiểu cần giữ trên một tuyến |
| $\tau^{\mathrm{alloc}}$ | Cooldown sau thay đổi nhiệm vụ xe |
| $\tau^{\mathrm{headway}}$ | Cooldown giữa hai lần thay đổi headway tuyến |

Headway là khoảng thời gian giữa hai lần xe xuất phát. Headway mục tiêu là yêu cầu điều phối; headway thực tế còn phụ thuộc xe có sẵn, traffic và layover.

### 2.3. Nhóm biến nhu cầu và traffic

| Ký hiệu | Ý nghĩa |
| --- | --- |
| $\lambda_{r,d,i}(t)$ | Cường độ khách đến trạm $i$, tuyến $r$, hướng $d$, tại thời điểm $t$ |
| $\bar\lambda_r$ | Tổng demand cơ sở của tuyến $r$, tính cho cả hai hướng |
| $p_{r,d,i,j}$ | Xác suất khách tại $i$ muốn đến $j$ |
| $A_{r,d,i,j}(t)$ | Số khách mới đến trong tick bắt đầu tại $t$, đi từ $i$ đến $j$ |
| $m_{\mathrm{peak}}$ | Hệ số tăng demand tại đỉnh cao điểm |
| $\mu_{\mathrm{peak}}$ | Thời điểm trung tâm cao điểm |
| $\sigma_{\mathrm{peak}}$ | Độ rộng cao điểm |
| $\xi_{e,\ell}$ | Hệ số traffic trên cạnh $e$, bucket thời gian $\ell$ |
| $\sigma_{\mathrm{traffic}}$ | Độ lệch chuẩn trong không gian log của traffic multiplier |
| $\Delta_{\mathrm{traffic}}$ | Độ dài một traffic bucket |
| $\tau_e(t)$ | Thời gian đi qua cạnh $e$ nếu bắt đầu đi tại $t$ |

Ví dụ, thời gian di chuyển được xác định từ:

$$
\tau_e(t)=\delta\left\lceil\frac{\bar\tau_e\,\xi_{e,\ell(t)}}{\delta}\right\rceil.
$$

Demand và traffic là các yếu tố ngoại sinh: controller có thể phản ứng với chúng nhưng không trực tiếp lựa chọn giá trị của chúng.

### 2.4. Nhóm biến trạng thái hành khách

| Ký hiệu | Ý nghĩa |
| --- | --- |
| $Q_{r,d,i,j}(t)$ | Số khách đang chờ đi từ $i$ đến $j$ trên tuyến $r$, hướng $d$ |
| $a_p$ | Thời điểm hành khách $p$ đến trạm |
| $w_p(t)=t-a_p$ | Tuổi chờ của khách $p$ nếu khách vẫn đang chờ |
| $\chi_p(t)$ | Trạng thái khách: WAITING, ONBOARD, COMPLETED hoặc ABANDONED |
| $\eta_p(t)$ | Cờ cho biết khách đã từng bị từ chối lên xe vì hết chỗ hay chưa |
| $B_{r,d,i,j}(t)$ | Số khách từ queue tương ứng được lên xe trong tick |
| $E_{r,d,i,j}(t)$ | Số khách từ queue tương ứng bỏ cuộc trong tick |
| $N_{\mathrm{waiting}}(t)$ | Tổng khách đang chờ |
| $N_{\mathrm{onboard}}(t)$ | Tổng khách trên xe |
| $N_{\mathrm{completed}}(t)$ | Tổng khách đã hoàn thành hành trình |
| $N_{\mathrm{abandoned}}(t)$ | Tổng khách đã bỏ cuộc |
| $N_{\mathrm{generated}}(t)$ | Tổng khách đã xuất hiện |

Queue count chưa đủ mô tả trạng thái: hai queue cùng 20 người nhưng một queue mới chờ 1 phút và một queue đã chờ 40 phút dẫn đến các quyết định khác nhau. Vì vậy cần giữ cả thông tin tuổi chờ và trạng thái hành khách.

### 2.5. Nhóm biến trạng thái xe và điều phối

| Ký hiệu | Ý nghĩa |
| --- | --- |
| $x_b(t)$ | Vị trí hiện tại của xe $b$: nút hoặc cạnh đang đi |
| $z_b(t)$ | Tuyến được gán cho xe $b$, hoặc không có tuyến |
| $d_b(t)$ | Hướng chạy của xe |
| $L_b(t)$ | Số khách trên xe |
| $\phi_b(t)$ | Phase: depot idle, deadhead, terminal idle, moving, dwell hoặc layover |
| $p_b(t)$ | Pattern FULL hoặc SHORT |
| $\rho_b(t)$ | Thời gian còn lại của movement/dwell/layover hiện tại |
| $c_b(t)$ | Cooldown phân bổ xe còn lại |
| $\mathcal P_b(t)$ | Các khách trên xe, gồm thông tin đích đến |
| $B_b(t),D_b(t)$ | Số khách lên/xuống xe trong tick |
| $h_r(t)$ | Headway mục tiêu hiện tại của tuyến $r$ |
| $\ell^{\mathrm{ANY}}_{r,d,v}(t)$ | Thời điểm departure gần nhất từ $v$, tuyến $r$, hướng $d$ |
| $\ell^{\mathrm{FULL}}_{r,d,v}(t)$ | Thời điểm FULL departure gần nhất của cùng luồng |
| $c_r^h(t)$ | Cooldown thay đổi headway còn lại |

### 2.6. Từ các biến xây dựng state

Trạng thái đầy đủ tại bước quyết định $k$ là:

$$
s_k=\left(t_k,\;\text{passenger records},\;
\{x_b,z_b,d_b,L_b,\phi_b,p_b,\rho_b,c_b,\mathcal P_b\}_{b\in\mathcal B},\;
\{h_r,\ell^{\mathrm{ANY}},\ell^{\mathrm{FULL}},c_r^h\},\;
\text{trạng thái ngoại sinh}\right).
$$

“Trạng thái ngoại sinh” ở đây là thông tin của quá trình sinh demand/traffic cần để mô tả dynamics đầy đủ. Nó không mặc nhiên là thông tin được cung cấp cho controller. Sau một quyết định, hệ thống tiến triển theo:

$$
s_{k+1}=f(s_k,u_k,\omega_k),
$$

trong đó $\omega_k$ biểu diễn arrivals và traffic tác động trong khoảng $[t_k,t_{k+1})$. Ở mức tick, queue và tải xe tuân theo:

$$
Q_{r,d,i,j}(t+\delta)=Q_{r,d,i,j}(t)+A_{r,d,i,j}(t)-B_{r,d,i,j}(t)-E_{r,d,i,j}(t),
$$

$$
L_b(t+\delta)=L_b(t)-D_b(t)+B_b(t).
$$

### 2.7. Từ các biến xây dựng action

Một quyết định vận hành có dạng:

$$
u_k=(\kappa_k,b_k,r_k,h_k),
$$

với $\kappa_k$ là loại quyết định. Các trường còn lại chỉ có ý nghĩa khi loại quyết định cần chúng.

| Quyết định | Ý nghĩa |
| --- | --- |
| $\mathrm{NOOP}$ | Giữ kế hoạch; dispatcher tiếp tục vận hành |
| $\mathrm{DISPATCH}(b,r)$ | Đưa xe dự phòng $b$ vào tuyến $r$ |
| $\mathrm{REASSIGN}(b,r)$ | Chuyển xe rỗng đủ điều kiện sang tuyến $r$ |
| $\mathrm{RECALL}(b)$ | Đưa xe đủ điều kiện về depot |
| $\mathrm{SHORT\_TURN}(b,r)$ | Giao một nhiệm vụ short-turn đã định nghĩa |
| $\mathrm{SET\_HEADWAY}(r,h)$ | Đặt $h_r=h$, với $h\in\mathcal H$ |

Tập quyết định khả thi phụ thuộc state:

$$
u_k\in\mathcal U(s_k).
$$

Các điều kiện xác định $\mathcal U(s_k)$ gồm sức chứa, phase, cooldown, vị trí, xe rỗng, fleet floor và donor guard. Việc chuyển xe phải tiêu tốn thời gian deadhead. Hai bất biến chính:

$$
0\le L_b(t)\le C_b,
$$

$$
N_{\mathrm{generated}}=N_{\mathrm{waiting}}+N_{\mathrm{onboard}}+N_{\mathrm{completed}}+N_{\mathrm{abandoned}}.
$$

### 2.8. Từ các biến xây dựng objective

Trong mỗi khoảng điều khiển $k$, định nghĩa:

| Ký hiệu | Đại lượng chi phí |
| --- | --- |
| $W_k$ | Tổng passenger-minutes chờ |
| $V_k$ | Tổng passenger-minutes trên xe |
| $O_k$ | Tổng passenger-minutes vượt ngưỡng comfort |
| $C_k$ | Tổng bus-minutes ngoài depot |
| $C_{D,k}$ | Tổng bus-minutes deadhead |
| $X_k$ | Tổng passenger-minutes chờ khi tuổi chờ đã đạt $\tau^{\mathrm{excess}}$ |
| $U_k$ | Số khách lần đầu bị từ chối vì xe đầy |
| $E_k$ | Số khách bỏ cuộc |
| $M_k$ | Số mission-change được chấp nhận |
| $N_H$ | Số khách chưa hoàn thành tại $H$: còn chờ hoặc còn trên xe |

Ví dụ, nếu thời gian nội bộ tính bằng giây:

$$
W_k=\frac1{60}\int_{t_k}^{t_{k+1}}N_{\mathrm{waiting}}(t)\,dt,
$$

$$
O_k=\frac1{60}\int_{t_k}^{t_{k+1}}\sum_b\max(0,L_b(t)-C_b^{\mathrm{comfort}})\,dt.
$$

Đặt trọng số chi phí:

$$
\mathbf w=(w_W,w_V,w_O,w_C,w_D,w_X,w_U,w_E,w_M,w_H).
$$

Chi phí mỗi khoảng:

$$
c_k=w_WW_k+w_VV_k+w_OO_k+w_CC_k+w_DC_{D,k}+w_XX_k+w_UU_k+w_EE_k+w_MM_k.
$$

Objective của bài toán là:

$$
\boxed{\min_\mu\;\mathbb E_\mu\left[\sum_{k=0}^{K-1}c_k+w_HN_H\right],\qquad u_k=\mu(\mathcal I_k)\in\mathcal U(s_k)}
$$

Ở đây $\mathcal I_k$ là thông tin được phép biết đến thời điểm $t_k$, còn $\mu$ là quy tắc điều khiển cần tìm. Quy tắc này phải quyết định từ thông tin hiện có, không biết trước demand và traffic tương lai.

## 3. Config sử dụng đúng các biến đã định nghĩa

### Network, fleet và trạng thái ban đầu

| Biến | Giá trị setup |
| --- | --- |
| $R$ | 3 tuyến |
| $n_r$ | 6 trạm/tuyến, hai hướng |
| $F$ | 12 xe |
| $C_b$ | 40 khách/xe |
| $C_b^{\mathrm{comfort}}$ | 30 khách/xe |
| $z_b(0)$ | 3 xe mỗi tuyến; 3 xe dự phòng |
| $x_b(0)$ | Mỗi tuyến: 2 xe ở $s_{r,0}$, 1 xe ở $s_{r,5}$; reserve ở depot |
| $L_b(0)$, $Q_{r,d,i,j}(0)$ | 0 |
| $\bar\tau_e$ | 180 giây cho cạnh giữa hai trạm liền kề trong base fixture |
| $\mathcal P_r^{\mathrm{short}}$ | $s_0\to s_1\to s_2\to s_3\to s_2\to s_1\to s_0$ |

### Thời gian và vận hành

| Biến | Giá trị setup |
| --- | --- |
| $\delta$ | 30 giây |
| $\Delta$ | 120 giây |
| $H$ | 14.400 giây |
| $K=H/\Delta$ | 120 quyết định |
| $H_D$ | 10.800 giây |
| $\tau^{\mathrm{dwell}}$ | 30 giây |
| $\tau^{\mathrm{layover}}$ | 120 giây |
| $\tau^{\mathrm{patience}}$ | 2.700 giây |
| $\tau^{\mathrm{excess}}$ | 900 giây |
| $h_0$ | 900 giây |
| $\mathcal H$ | $\{360,600,900\}$ giây |
| $h_{\min}$ | 120 giây |
| $h_{\mathrm{guard}}$ | 1.200 giây |
| $F_{\min}$ | 2 xe FULL committed/tuyến |
| $\tau^{\mathrm{alloc}}$ | 1.200 giây |
| $\tau^{\mathrm{headway}}$ | 600 giây |

### Demand, traffic và objective

| Biến | Giá trị setup |
| --- | --- |
| $(\bar\lambda_1,\bar\lambda_2,\bar\lambda_3)$ | $(180,160,140)$ khách/giờ |
| $m_{\mathrm{peak}}$ | 1,5–3,0 |
| $\mu_{\mathrm{peak}}$ | Phút 45–135 |
| $\sigma_{\mathrm{peak}}$ | 15–30 phút |
| $p_{r,d,i,j}$ | Tỷ lệ với $\exp(-\lvert j-i\rvert/2)$, chuẩn hóa trên các đích hợp lệ |
| $\sigma_{\mathrm{traffic}}$ | 0,15 |
| $\Delta_{\mathrm{traffic}}$ | 300 giây |
| $\xi_{e,\ell}$ | Lognormal, chặn trong $[0,7;2,5]$ ở base scenario |
| $\mathbf w$ | $(1;\,0,25;\,0,5;\,0,5;\,0,5;\,1;\,5;\,60;\,2;\,60)$ |

Các giới hạn tensor như tối đa 4 tuyến, 16 xe, 8 trạm thuộc phần encoding của solution; chúng không phải quy luật vận hành của bài toán.

# B. Solution: Dùng MaskablePPO để học quy tắc điều khiển

## 1. Input, output và flow cơ bản

Solution RL nhận **observation của môi trường và danh sách action hợp lệ**, sau đó chọn một quyết định vận hành. Có hai loại output cần phân biệt:

| Giai đoạn | Input | Output |
| --- | --- | --- |
| Khi điều khiển | Observation $o_k$, action mask $m_k$ | Một action $a_k$, được giải mã thành $u_k$ của phần A |
| Khi huấn luyện | Các tương tác với môi trường và training config | Tham số policy đã học $\theta^*$, lưu thành checkpoint |

Flow huấn luyện:

1. Môi trường tạo observation $o_k$ và mask $m_k$.
2. Actor tính xác suất của các action hợp lệ.
3. Chọn action $a_k$, giải mã thành quyết định $u_k$.
4. Simulator thực hiện $u_k$ và tiến thêm $\Delta$ giây.
5. Môi trường trả reward $r_k$, observation mới $o_{k+1}$ và tín hiệu kết thúc.
6. Thu thập nhiều bước thành rollout.
7. Critic và reward được dùng để ước lượng return, advantage.
8. PPO cập nhật actor và critic.
9. Lặp lại với policy vừa cập nhật.

Khi sử dụng policy đã học, chỉ cần chu trình nhận observation/mask → chọn action → thực thi. Không cần cập nhật trọng số sau mỗi quyết định.

## 2. MaskablePPO nằm ở đâu trong landscape RL?

| Trục phân loại | MaskablePPO thuộc nhóm nào? | Ý nghĩa trong bài toán này |
| --- | --- | --- |
| Model-based / model-free | **Model-free** | Không học hoặc dùng mô hình chuyển trạng thái để lập kế hoạch bằng cách thử các tương lai |
| On-policy / off-policy | **On-policy** | Cập nhật từ rollout của policy vừa dùng để thu thập dữ liệu |
| Value-based / policy-based | **Tối ưu policy trực tiếp** | Actor trực tiếp biểu diễn xác suất chọn action |
| Actor–critic | **Actor–critic** | Actor chọn action; critic ước lượng giá trị để hỗ trợ cập nhật actor |
| Thông tin trạng thái | **Partial observation** | Policy chỉ nhận aggregate/history, không thấy full simulator state |

Việc có simulator không làm PPO trở thành model-based: ở đây simulator cung cấp trải nghiệm như một môi trường. PPO không truy vấn simulator để tìm kiếm cây hành động trước mỗi quyết định. PPO dùng nhiều epoch trên cùng rollout, nhưng vẫn được xếp là on-policy: nó giới hạn độ lệch so với policy thu thập rollout và thay rollout cũ bằng dữ liệu mới.

## 3. Các biến của thuật toán

| Ký hiệu | Ý nghĩa |
| --- | --- |
| $o_k=O(s_k)$ | Observation được tạo từ state |
| $m_k(a)\in\{0,1\}$ | Action $a$ có hợp lệ tại bước $k$ hay không |
| $a_k$ | Chỉ số action được policy chọn |
| $\theta$ | Tham số actor |
| $\varphi$ | Tham số critic |
| $\pi_\theta(a\mid o,m)$ | Phân phối xác suất trên action hợp lệ |
| $V_\varphi(o_k)$ | Ước lượng tổng reward tương lai từ observation |
| $r_k$ | Reward môi trường trả sau action |
| $\gamma$ | Discount factor |
| $\lambda_{\mathrm{GAE}}$ | Tham số GAE |
| $\hat A_k$ | Advantage ước lượng |
| $\hat G_k$ | Return target cho critic |
| $\theta_{\mathrm{old}}$ | Actor đã thu thập rollout hiện tại |
| $\rho_k(\theta)$ | Tỷ lệ xác suất action giữa policy mới và policy cũ |
| $\epsilon$ | Biên clip của PPO |
| $\alpha$ | Learning rate |
| $c_V,c_{\mathrm{ent}}$ | Trọng số value loss và entropy |
| $N_{\mathrm{ref}}$ | Hệ số chuẩn hóa reward |
| $n_{\mathrm{env}},n_{\mathrm{step}}$ | Số môi trường và số bước thu thập trên mỗi môi trường |
| $n_{\mathrm{batch}},n_{\mathrm{epoch}}$ | Kích thước minibatch và số lượt cập nhật trên rollout |

Actor và critic có thể dùng chung phần trích xuất feature; $\theta,\varphi$ ở đây biểu diễn hai vai trò chức năng.

## 4. Input từ môi trường và cách xử lý

Observation gồm các nhóm:

| Input | Thông tin mang vào policy |
| --- | --- |
| `stops` | Queue count, mean/max age, số khách chờ lâu, boarding/alighting/arrivals gần đây |
| `arrival_history` | Arrivals trong 5 control interval gần nhất |
| `vehicles` | Phase, tuyến, hướng, pattern, vị trí, tải và timers của xe |
| `routes` | Headway target, departure gaps, số xe committed/incoming/short và cooldown |
| `context` | Thời gian hiện tại, thời gian còn lại, cờ forecast |
| Entity masks | Vị trí tensor nào tương ứng với tuyến/trạm/xe có thật |
| `forecast` | Dự báo arrivals nếu bật; bằng 0 khi tắt |
| Action mask $m_k$ | Các quyết định hiện có thể thực hiện |

Arrival history là một phần của observation. Core policy dùng MLP xử lý các feature này; không tự có bộ nhớ recurrent. Các feature được scale, ghép thành biểu diễn đầu vào rồi đi qua mạng:

```
Observation
    ↓
Feature extractor / MLP
    ├── Actor → action logits → action mask → xác suất action
    └── Critic → Vφ(o)
```

Với logits $z_\theta(o,a)$, phân phối sau masking là:

$$
\pi_\theta(a\mid o,m)=\frac{m(a)\exp(z_\theta(o,a))}{\sum_{a'}m(a')\exp(z_\theta(o,a'))}.
$$

Action không hợp lệ có xác suất bằng 0. Mask thực thi điều kiện khả thi của phần A. Encoding hiện dùng 221 slot:

$$
1+64+64+64+16+12=221
$$

tương ứng NOOP, DISPATCH, REASSIGN, SHORT_TURN, RECALL và SET_HEADWAY. Slot dành cho entity không tồn tại cũng bị mask.

## 5. Reward mà RL tối ưu

Từ cost đã định nghĩa trong phần A, môi trường trả:

$$
r_k=-\frac{c_k+\mathbf1[k=K-1]\,w_HN_H}{N_{\mathrm{ref}}}.
$$

Với $N_{\mathrm{ref}}=3000$, objective của RL là:

$$
\boxed{J(\theta)=\mathbb E_{\pi_\theta}\left[\sum_{k=0}^{K-1}\gamma^kr_k\right]}
$$

Vì $\gamma=1$:

$$
J(\theta)=-\frac1{3000}\mathbb E_{\pi_\theta}\left[\sum_{k=0}^{K-1}c_k+w_HN_H\right].
$$

Do đó, tối đa hóa tổng reward tương đương tối thiểu hóa objective vận hành ở phần A. Chuẩn hóa bằng một hằng số dương không thay đổi thứ tự tốt/xấu của các policy theo objective này.

## 6. PPO cập nhật policy như thế nào?

**Bước 1: Thu thập rollout.** Với policy hiện tại, lưu các mẫu:

$$
(o_k,m_k,a_k,r_k,o_{k+1},\mathrm{done}_k,\log\pi_{\theta_{\mathrm{old}}}(a_k\mid o_k,m_k),V_\varphi(o_k)).
$$

**Bước 2: Ước lượng advantage.** Đặt $d_k=1$ nếu bước kết thúc nhiệm vụ:

$$
e_k=r_k+\gamma(1-d_k)V_\varphi(o_{k+1})-V_\varphi(o_k).
$$

GAE tính ngược:

$$
\hat A_k=e_k+\gamma\lambda_{\mathrm{GAE}}(1-d_k)\hat A_{k+1}.
$$

Advantage dương nghĩa là kết quả tốt hơn mức critic dự kiến; advantage âm nghĩa là kém hơn. Return target:

$$
\hat G_k=\hat A_k+V_\varphi(o_k).
$$

Ở cuối horizon, giá trị tương lai bằng 0. **Bước 3: Cập nhật actor có giới hạn.** Tính:

$$
\rho_k(\theta)=\frac{\pi_\theta(a_k\mid o_k,m_k)}{\pi_{\theta_{\mathrm{old}}}(a_k\mid o_k,m_k)}.
$$

PPO dùng objective:

$$
L^{\mathrm{clip}}(\theta)=\mathbb E_k\left[\min\left(\rho_k\hat A_k,\;\operatorname{clip}(\rho_k,1-\epsilon,1+\epsilon)\hat A_k\right)\right].
$$

Actor được khuyến khích tăng xác suất action có advantage tốt. Clipping hạn chế động lực thay đổi xác suất quá mạnh trong một lần cập nhật. **Bước 4: Cập nhật critic và duy trì exploration.**

$$
L_V(\varphi)=\mathbb E_k\left[(V_\varphi(o_k)-\hat G_k)^2\right].
$$

Loss tổng để minimize:

$$
\boxed{L=-L^{\mathrm{clip}}+c_VL_V-c_{\mathrm{ent}}\,\mathbb E_k[\mathcal H(\pi_\theta(\cdot\mid o_k,m_k))]}
$$

Entropy khuyến khích policy duy trì sự đa dạng trong lựa chọn action. Đây là loss phục vụ học; tiêu chí đánh giá lời giải cuối cùng vẫn là chi phí vận hành của phần A.

## 7. Config và final output của solution

| Biến thuật toán | Config code | Giá trị cơ sở |
| --- | --- | --- |
| $\alpha$ | `learning_rate` | $3\times10^{-4}$ |
| $\gamma$ | `gamma` | 1,0 |
| $\lambda_{\mathrm{GAE}}$ | `gae_lambda` | 0,95 |
| $\epsilon$ | `clip_range` | 0,2 |
| $n_{\mathrm{env}}$ | `n_envs` | 4 |
| $n_{\mathrm{step}}$ | `n_steps` | 256 |
| $n_{\mathrm{batch}}$ | `batch_size` | 256 |
| $n_{\mathrm{epoch}}$ | `n_epochs` | 4 |
| $c_V$ | `vf_coef` | 0,5 |
| $c_{\mathrm{ent}}$ | `ent_coef` | 0,01 |
| $N_{\mathrm{ref}}$ | `n_ref` | 3.000 |

Mỗi rollout có $4\times256=1024$ transitions. Sau cập nhật, policy mới tiếp tục thu thập rollout tiếp theo. Final output của training là policy đã học $\pi_{\theta^*}$. Khi nhận một observation và mask mới, policy trả action:

$$
a_k^*=\arg\max_{a:m_k(a)=1}\pi_{\theta^*}(a\mid o_k,m_k)
$$

nếu chạy deterministic evaluation. Action index được giải mã thành một quyết định cụ thể, chẳng hạn:

$$
u_k=\mathrm{DISPATCH}(b=10,r=2).
$$

Simulator thực thi quyết định đó theo thời gian và constraint của bài toán; chuỗi quyết định qua cả episode tạo ra kết quả vận hành dùng để đo cost, waiting time, completed share và các chỉ số dịch vụ.
