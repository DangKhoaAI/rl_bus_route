# Điều phối đội xe buýt bằng RL — bản giải thích nhanh

## A. Problem: Cần giải quyết điều gì?

### 1. Goal

Điều phối đội xe hữu hạn trên mạng tuyến cố định theo nhu cầu biến động, để giảm chờ đợi, quá đông và khách không được phục vụ, đồng thời kiểm soát chi phí vận hành. Kết quả cần tìm là quy tắc chọn quyết định từ thông tin hiện có, có tính đến ảnh hưởng lên phần thời gian còn lại.

### 2. Các biến → state → action → objective

Các biến chia thành nhóm; $r,d,i,j,b$ lần lượt chỉ tuyến, hướng, trạm đi, trạm đến và xe. $t$ là thời gian vật lý; $k$ là bước quyết định.

| Nhóm | Các biến và ý nghĩa |
|---|---|
| Nguồn lực | $R$: số tuyến; $n_r$: số trạm/tuyến; $F$: tổng xe; $C_b$: sức chứa; $C_b^{\mathrm{comfort}}$: ngưỡng tải thoải mái |
| Thời gian | $\delta$: tick; $\Delta$: khoảng điều khiển; $H$: horizon; $K=H/\Delta$: số quyết định; $H_D$: lúc ngừng sinh khách |
| Ngoại sinh | $A_{r,d,i,j}(t)$: khách mới; $\bar\lambda_r$: demand cơ sở; $\xi(t)$: traffic multiplier tác động lên thời gian đi |
| Hành khách | $Q_{r,d,i,j}(t)$: queue; $w_p(t)$: tuổi chờ; trạng thái chờ/đang đi/hoàn thành/bỏ cuộc |
| Xe, điều phối | $L_b(t)$: tải; $x_b(t)$: vị trí; $z_b(t)$: tuyến; $\phi_b(t)$: phase; timers/cooldown; $h_r(t)$: headway mục tiêu |

**State.** $s_k$ gộp thời gian, hồ sơ khách, trạng thái xe, dispatcher và trạng thái ngoại sinh. Cần cả queue count lẫn tuổi chờ: 20 khách mới đến và 20 khách đã chờ 40 phút tạo mức khẩn cấp khác nhau. Dynamics là $s_{k+1}=f(s_k,u_k,\omega_k)$, với $\omega_k$ là arrivals/traffic trong khoảng điều khiển. Queue tăng khi khách đến, giảm khi lên xe hoặc bỏ cuộc; tải xe tăng khi boarding và giảm khi alighting.

**Action.** Mỗi bước chọn một quyết định $u_k$ trong tập khả thi $\mathcal U(s_k)$:

| Quyết định | Tác động |
|---|---|
| `NOOP` / `DISPATCH(b,r)` | Tiếp tục kế hoạch / đưa xe dự phòng vào tuyến |
| `REASSIGN(b,r)` / `RECALL(b)` | Chuyển xe sang tuyến khác / đưa xe về depot |
| `SHORT_TURN(b,r)` | Chạy hành trình ngắn định trước rồi trở lại FULL |
| `SET_HEADWAY(r,h)` | Đặt khoảng cách xuất phát mục tiêu $h_r=h$ |

**Constraints.** Không vượt sức chứa; bảo toàn khách và xe; chỉ chuyển xe rỗng đủ điều kiện; deadhead tốn thời gian. Phải tuân thủ cooldown và giữ $F_{\min}$ xe FULL committed/tuyến cùng donor guard. Đặt headway thấp chỉ tạo yêu cầu: cần xe có mặt và sẵn sàng mới thực sự xuất phát được.

**Objective.** Tìm quy tắc $\mu$ từ thông tin hiện có để giảm tổng chi phí kỳ vọng:

$$
\min_\mu\;\mathbb E_\mu\left[\sum_{k=0}^{K-1}c_k+60N_H\right],\qquad u_k\in\mathcal U(s_k).
$$

Chi phí cơ sở mỗi khoảng là:

$$
c_k=W_k+0.25V_k+0.5O_k+0.5C_k+0.5C_{D,k}+X_k+5U_k+60E_k+2M_k.
$$

$W,V,O,X$ là passenger-minutes chờ, trên xe, quá comfort và chờ lâu; $C,C_D$ là bus-minutes hoạt động và deadhead; $U,E,M$ đếm lần đầu bị từ chối vì đầy xe, khách bỏ cuộc và đổi mission. $N_H$ đếm khách còn chờ hoặc còn trên xe cuối ca. Ví dụ, 10 khách chờ 5 phút tạo $W=50$ passenger-minutes.

### 3. Setup cơ sở theo các biến trên

- $R=3$, $n_r=6$, $F=12$: 3 xe/tuyến và 3 reserve; $C_b=40$, $C_b^{\mathrm{comfort}}=30$; ban đầu không có khách.
- $\delta=30$ giây, $\Delta=120$ giây; $H=240$ phút, $K=120$; $H_D=180$ phút để còn 60 phút phục vụ khách tồn.
- $\bar\lambda_r=(180,160,140)$ khách/giờ cho cả hai hướng, có cao điểm; traffic biến động theo thời gian.
- $h_r(0)=15$ phút; chọn $h\in\mathcal H=\{6,10,15\}$ phút; $F_{\min}=2$. Cooldown xe/tuyến là 20/10 phút; khách bỏ cuộc sau 45 phút chờ.

## B. Solution: RL học cách quyết định như thế nào?

### 1. Input, output và flow

Mỗi bước, solution nhận observation $o_k$ và action mask $m_k$, trả chỉ số $a_k$ được giải mã thành quyết định $u_k$. Observation gồm queue/tuổi chờ, lịch sử arrivals, trạng thái xe, headway và thời gian còn lại. Destination chi tiết của khách chờ và demand/traffic tương lai bị ẩn; $o_k$ chỉ phản ánh một phần $s_k$, nên đây là POMDP.

Flow: **quan sát → chọn action → simulator chạy 2 phút → nhận reward và observation mới → gom rollout → cập nhật policy → lặp lại**. Training cuối cùng trả policy đã học, còn mỗi lần sử dụng policy trả một quyết định vận hành.

### 2. Thuật toán và các biến chính

MaskablePPO thuộc **model-free**: học từ tương tác, không lập kế hoạch bằng mô hình chuyển trạng thái. Nó là **on-policy**: dùng rollout của policy vừa chạy; tối ưu policy trực tiếp bằng **actor–critic**. Simulator cung cấp trải nghiệm, không khiến PPO trở thành model-based.

- Actor $\pi_\theta(a\mid o,m)$ có tham số $\theta$, tính xác suất action; mask loại lựa chọn không hợp lệ trong 221 slot.
- Critic $V_\varphi(o)$ có tham số $\varphi$, ước lượng tổng reward tương lai để hỗ trợ actor học.
- $\hat A_k$ là advantage: kết quả tốt hơn/kém hơn critic dự kiến; $\gamma$ là discount; $\lambda_{\mathrm{GAE}}$ điều khiển cách kết hợp các sai số dự đoán qua nhiều bước.

MLP xử lý observation và lịch sử được đưa sẵn vào feature; core policy không có bộ nhớ recurrent. Khi train, actor lấy mẫu action; khi đánh giá deterministic, chọn action hợp lệ có xác suất cao nhất.

### 3. PPO xử lý dữ liệu và cập nhật

Rollout lưu observation, mask, action, reward, tín hiệu kết thúc, value estimate và xác suất action của policy cũ. GAE dùng reward cùng critic để tính advantage. PPO tăng xác suất action có advantage tốt; so sánh xác suất mới/cũ và dùng clipping $\epsilon=0.2$ để hạn chế thay đổi quá mạnh.

Critic học khớp return target; entropy khuyến khích exploration. Loss kết hợp ba phần: âm objective PPO đã clip, sai số value và âm entropy. Setup cơ sở: learning rate $\alpha=3\times10^{-4}$, $\lambda_{\mathrm{GAE}}=0.95$; 4 môi trường × 256 bước tạo 1.024 transitions/rollout, cập nhật 4 epoch với minibatch 256.

### 4. Reward optimize và final output

$$
r_k=-\frac{c_k+\mathbf1[k=K-1]\,60N_H}{3000},\qquad
\max_\theta J(\theta)=\mathbb E_{\pi_\theta}\left[\sum_k\gamma^kr_k\right].
$$

Với $\gamma=1$, tối đa hóa tổng reward tương đương tối thiểu hóa objective phần A. Phạt cuối ca ngăn việc để khách tồn mà không chịu chi phí; critic bootstrap bằng 0 khi kết thúc horizon.

Output training là checkpoint chứa policy $\pi_{\theta^*}$. Khi sử dụng, policy nhận observation/mask và trả quyết định như `DISPATCH(10,2)`, không cần cập nhật trọng số. Chất lượng được đánh giá bằng cost và các chỉ số phục vụ trên scenario đánh giá.

<!-- Bản mở rộng khoảng gấp đôi bản brief ban đầu; không còn giới hạn một trang A4. Bản đầy đủ: problem_solution.md. -->
