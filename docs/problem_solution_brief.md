# Điều phối đội xe buýt bằng RL

RL (Reinforcement Learning, học tăng cường) học cách ra quyết định bằng cách thử hành động trong mô phỏng và nhận điểm đánh giá. Phần A định nghĩa bài toán vận hành; phần B giải thích cách dùng RL để giải bài toán đó.

## A. Problem: Cần giải quyết điều gì?

### 1. Goal

Điều phối đội xe hữu hạn trên mạng tuyến cố định theo nhu cầu biến động, để giảm chờ đợi, quá đông và khách không được phục vụ, đồng thời kiểm soát chi phí vận hành. Kết quả cần tìm là quy tắc chọn quyết định từ thông tin hiện có, có tính đến ảnh hưởng lên phần thời gian còn lại.

Ví dụ: một tuyến đông khách đột ngột, tuyến khác ít khách. Người điều phối cần cân nhắc đưa xe dự phòng vào hay chuyển xe từ tuyến khác, vì chuyển xe mất thời gian và có thể khiến khách ở tuyến cho xe phải chờ lâu hơn.

### 2. Các biến → state → action → objective

Các biến chia thành nhóm; $r,d,i,j,b$ lần lượt chỉ tuyến, hướng, trạm đi, trạm đến và xe. $t$ là thời gian vật lý; $k$ là bước quyết định.

| Nhóm | Các biến và ý nghĩa |
|---|---|
| Nguồn lực | $R$: số tuyến; $n_r$: số trạm/tuyến; $F$: tổng xe; $C_b$: sức chứa; $C_b^{\mathrm{comfort}}$: ngưỡng tải thoải mái |
| Thời gian | $\delta$: bước mô phỏng (tick); $\Delta$: khoảng điều khiển; $H$: thời lượng ca (horizon); $K=H/\Delta$: số quyết định; $H_D$: lúc ngừng sinh khách |
| Ngoại sinh | $A_{r,d,i,j}(t)$: khách mới; $\bar\lambda_r$: nhu cầu cơ sở; $\xi(t)$: hệ số giao thông làm thay đổi thời gian đi; controller không điều khiển các yếu tố này |
| Hành khách | $Q_{r,d,i,j}(t)$: số người trong hàng chờ (queue); $w_p(t)$: thời gian khách $p$ đã chờ; trạng thái chờ/đang đi/hoàn thành/bỏ cuộc |
| Xe, điều phối | $L_b(t)$: tải; $x_b(t)$: vị trí; $z_b(t)$: tuyến; $\phi_b(t)$: trạng thái hoạt động; thời gian còn lại; $h_r(t)$: khoảng cách xuất phát mục tiêu (headway) |

**State (trạng thái).** $s_k$ gộp thời gian, hồ sơ khách, trạng thái xe, bộ điều phối xuất bến và trạng thái ngoại sinh. Cần cả số người lẫn tuổi chờ: 20 khách mới đến và 20 khách đã chờ 40 phút tạo mức khẩn cấp khác nhau. Quy luật chuyển trạng thái là $s_{k+1}=f(s_k,u_k,\omega_k)$, với $\omega_k$ là khách mới/giao thông trong khoảng điều khiển. Hàng chờ giảm khi khách lên xe hoặc bỏ cuộc; tải xe giảm khi khách xuống xe.

**Action (hành động).** Mỗi bước chọn một quyết định $u_k$ trong tập khả thi $\mathcal U(s_k)$:

| Quyết định | Tác động |
|---|---|
| `NOOP` / `DISPATCH(b,r)` | Tiếp tục kế hoạch / đưa xe dự phòng vào tuyến |
| `REASSIGN(b,r)` / `RECALL(b)` | Chuyển xe sang tuyến khác / đưa xe về bãi xe (depot) |
| `SHORT_TURN(b,r)` | Quay đầu ở trạm cho phép để phục vụ đoạn ngắn, rồi trở lại chạy toàn tuyến (FULL) |
| `SET_HEADWAY(r,h)` | Đặt khoảng cách xuất phát mục tiêu $h_r=h$ |

**Constraints (ràng buộc).** Không vượt sức chứa; không làm mất khách hoặc sinh thêm xe; chỉ chuyển xe rỗng đủ điều kiện. Chạy chuyển vị trí không chở khách (deadhead) vẫn tốn thời gian. Phải đợi đủ thời gian giữa hai lần đổi nhiệm vụ (cooldown), giữ $F_{\min}$ xe cam kết chạy toàn tuyến và kiểm tra tuyến cho xe còn xe sẵn sàng để xuất bến kịp (donor guard). Headway thấp chỉ là mục tiêu; cần xe có mặt mới xuất phát được.

**Objective (mục tiêu tối ưu).** Tìm quy tắc $\mu$ từ thông tin hiện có để giảm tổng chi phí kỳ vọng; $\mathbb E$ biểu diễn trung bình trên các tình huống nhu cầu/giao thông:

$$
\min_\mu\;\mathbb E_\mu\left[\sum_{k=0}^{K-1}c_k+60N_H\right],\qquad u_k\in\mathcal U(s_k).
$$

Chi phí cơ sở mỗi khoảng là:

$$
c_k=W_k+0.25V_k+0.5O_k+0.5C_k+0.5C_{D,k}+X_k+5U_k+60E_k+2M_k.
$$

$W,V,O,X$ lần lượt đo người-phút chờ, trên xe, vượt ngưỡng thoải mái và chờ từ 15 phút trở lên; $C,C_D$ đo xe-phút ngoài depot và deadhead. $U,E,M$ đếm khách lần đầu không lên được vì đầy xe, khách bỏ cuộc và lần đổi nhiệm vụ. $N_H$ là khách còn chờ hoặc trên xe cuối ca. Ví dụ, 10 khách chờ 5 phút tạo $W=50$ người-phút. Các hệ số quy đổi thành điểm chi phí tổng hợp, không phải tiền thực tế; chúng thể hiện mức ưu tiên giữa phục vụ khách và sử dụng xe.

### 3. Setup cơ sở theo các biến trên

- $R=3$, $n_r=6$, $F=12$: 3 xe/tuyến và 3 xe dự phòng; $C_b=40$, $C_b^{\mathrm{comfort}}=30$; ban đầu không có khách.
- $\delta=30$ giây, $\Delta=120$ giây; $H=240$ phút, $K=120$; $H_D=180$ phút để còn 60 phút phục vụ khách tồn.
- $\bar\lambda_r=(180,160,140)$ khách/giờ cho cả hai hướng, có cao điểm; giao thông biến động theo thời gian. Đây là dữ liệu mô phỏng tổng hợp.
- $h_r(0)=15$ phút; chọn $h\in\mathcal H=\{6,10,15\}$ phút; $F_{\min}=2$. Cooldown xe/tuyến là 20/10 phút; khách bỏ cuộc sau 45 phút chờ.

## B. Solution: RL học cách quyết định như thế nào?

### 1. Input, output và flow

Mỗi bước, RL nhận thông tin quan sát $o_k$ (observation) và danh sách đánh dấu hành động hợp lệ $m_k$ (action mask), trả mã hành động $a_k$ được đổi thành quyết định $u_k$. Quan sát gồm hàng chờ/tuổi chờ, lượng khách đến gần đây, trạng thái xe, headway và thời gian còn lại. Đích đến chi tiết của khách chờ và nhu cầu/giao thông tương lai bị ẩn: đây là bài toán ra quyết định chỉ quan sát một phần trạng thái (POMDP).

Flow: **quan sát → chọn hành động → mô phỏng chạy 2 phút → nhận điểm reward và quan sát mới → gom chuỗi tương tác (rollout) → cập nhật quy tắc lựa chọn (policy) → lặp lại**. Huấn luyện trả policy đã học; mỗi lần sử dụng policy trả một quyết định vận hành.

### 2. Thuật toán và các biến chính

MaskablePPO là PPO (Proximal Policy Optimization) có thêm action mask. Nó thuộc **model-free**: học từ tương tác, không dùng mô hình chuyển trạng thái để lập kế hoạch; **on-policy**: cập nhật từ rollout của policy vừa chạy; và **actor–critic**: tối ưu policy trực tiếp với sự hỗ trợ của bộ ước lượng giá trị. Có simulator không khiến PPO trở thành model-based.

- Actor $\pi_\theta(a\mid o,m)$ có tham số $\theta$, tính xác suất hành động; mask loại lựa chọn không hợp lệ trong 221 mã dành cho các tổ hợp loại hành động/xe/tuyến/headway.
- Critic $V_\varphi(o)$ có tham số $\varphi$, ước lượng tổng reward tương lai để hỗ trợ actor học.
- $\hat A_k$ là advantage: kết quả tốt hơn/kém hơn critic dự kiến; $\gamma$ là hệ số giảm trọng số reward tương lai; $\lambda_{\mathrm{GAE}}$ điều khiển cách ước lượng advantage từ nhiều bước.

Mạng nơ-ron truyền thẳng nhiều lớp (MLP) xử lý quan sát và lịch sử được đưa sẵn vào các đặc trưng đầu vào. Khi huấn luyện, actor lấy mẫu hành động theo xác suất; khi đánh giá theo chế độ xác định, chọn hành động hợp lệ có xác suất cao nhất.

### 3. PPO xử lý dữ liệu và cập nhật

Rollout lưu quan sát, mask, hành động, reward, tín hiệu kết thúc, giá trị dự đoán và xác suất hành động của policy cũ. GAE (Generalized Advantage Estimation) kết hợp reward với critic để tính advantage. PPO khuyến khích tăng xác suất hành động có advantage tốt; dùng tỷ lệ xác suất mới/cũ và biên cắt (clipping) $\epsilon=0.2$ để hạn chế thay đổi quá mạnh.

Critic học dự đoán tổng reward tương lai; entropy khuyến khích thử nhiều lựa chọn thay vì sớm chỉ chọn một. Hàm mất mát kết hợp mục tiêu actor, sai số critic và entropy. Setup: tốc độ học $\alpha=3\times10^{-4}$, $\lambda_{\mathrm{GAE}}=0.95$; 4 môi trường × 256 bước tạo 1.024 mẫu/rollout, học qua dữ liệu 4 lượt (epoch), mỗi nhóm cập nhật 256 mẫu (minibatch).

### 4. Reward optimize và final output

$$
r_k=-\frac{c_k+\mathbf1[k=K-1]\,60N_H}{3000},\qquad
\max_\theta J(\theta)=\mathbb E_{\pi_\theta}\left[\sum_k\gamma^kr_k\right].
$$

Ở đây $r_k$ là điểm sau một quyết định, $J$ là tổng điểm kỳ vọng; $\mathbf1[k=K-1]$ bằng 1 ở bước cuối và bằng 0 ở các bước khác. Với $\gamma=1$, tối đa hóa tổng reward tương đương giảm objective phần A. Chia cho 3000 chỉ chuẩn hóa thang điểm. Cuối ca phạt khách tồn và đặt giá trị tương lai bằng 0 vì nhiệm vụ đã kết thúc.

Kết quả huấn luyện là file lưu tham số (checkpoint) của policy $\pi_{\theta^*}$. Khi sử dụng, policy có thể trả `DISPATCH(10,2)`: đưa xe số 10 vào tuyến số 2. Chất lượng phải được kiểm tra trên các ngày mô phỏng đánh giá bằng cost, thời gian chờ và tỷ lệ khách hoàn thành; huấn luyện xong chưa chứng minh RL tốt hơn cách điều phối khác.

<!-- Bản mở rộng khoảng gấp đôi bản brief ban đầu; không còn giới hạn một trang A4. Bản đầy đủ: problem_solution.md. -->
