# Research: Dynamic Bus Fleet Reallocation and Frequency Control

Phiên bản **0.2 — 2026-09-12**. Định nghĩa mới do người dùng cung cấp thay thế hướng thiết kế mạng tuyến trong v0.1. Thiết kế cũ còn trong Git commit `a806bb6`; không còn là mục tiêu hiện hành.

Đọc cùng [spec](../spec.md) và [plan](../plan.md). Trạng thái: nghiên cứu/thiết kế; chưa có simulator, model hay kết quả thực nghiệm.

## 1. Vấn đề hiện tại

Mạng tuyến đã tồn tại. Nhu cầu và thời gian di chuyển thay đổi theo thời gian. Bộ điều khiển quan sát hàng chờ, tải xe, headway và nguồn lực rồi quyết định điều xe dự phòng, phân bổ lại xe, thay đổi mục tiêu giãn cách hoặc chạy chuyến tăng cường trên một đoạn được phép.

Tên đề tài:

> **Dynamic Bus Fleet Reallocation and Frequency Control Using Reinforcement Learning under Time-Varying Passenger Demand.**

Đối tượng tối ưu là **chuỗi quyết định vận hành trong một ca**, với số xe hữu hạn. Route geometry là đầu vào. RL có lý do để thử vì một quyết định hôm nay làm thay đổi nơi xe có mặt, lượng khách tồn và lựa chọn khả thi ở các bước sau; điều này chưa chứng minh RL sẽ thắng heuristic hoặc tối ưu hóa cổ điển.

## 2. Kiểm tra các lập luận trong định nghĩa mới

| Nội dung | Kết luận và cách dùng |
|---|---|
| Điều xe giữa các tuyến là một bài toán thực tế | Có cơ sở nghiên cứu; cần giới hạn điểm chuyển và lịch nhiệm vụ |
| Dynamic interlining | Bài được dẫn nghiên cứu shared fleet tại một hub chung; không phải bằng chứng cho phép chuyển xe tùy ý ở mọi nơi [R1] |
| Short-turning và interlining | Có nghiên cứu allocation kết hợp hai lựa chọn này, nhưng bài được dẫn là tactical optimization, không phải phương pháp RL [R2] |
| Deadheading | Có nghiên cứu vận hành chạy rỗng; bài được dẫn không tự chứng minh mọi thiết kế bus trong dự án [R3] |
| “Giảm 20 xuống 10 phút/chuyến” | Nếu nói khoảng cách xuất bến thì đó là headway; không phải thời gian một chuyến đi |
| Tăng frequency | Phải có xe và lịch di chuyển để thực hiện; không được sửa tham số để tăng năng lực ảo |
| Công bằng giữa các tuyến | Penalty giúp thể hiện ưu tiên nhưng không bảo đảm không hy sinh tuyến; cần floor nguồn lực và guard dịch vụ |
| Forecast demand | Có ích để thử, nhưng dự báo phải causal và được đo lỗi; không đưa realized future vào state |

Ví dụ định lượng: với chu kỳ xe 40 phút và headway mục tiêu 10 phút, xấp xỉ cần 4 xe hoạt động đều; headway 5 phút cần khoảng 8 xe. Quan hệ này là sanity check ở steady state, không thay thế lịch từng xe trong simulator.

## 3. Các nguồn nghiên cứu phù hợp

### Dynamic interlining

Zahedi, Koutsopoulos và Ma mô tả chia sẻ đội xe giữa các tuyến cùng bến hub và điều phối để thực hiện các chuyến theo lịch. Bài dùng tối ưu hóa và simulation để đánh giá tính tin cậy. Dự án tham khảo nguyên tắc chia sẻ nguồn lực có ràng buộc; deadhead giữa các bến riêng là giả định mở rộng phải mô hình hóa riêng. Công bố online 2023, volume journal 2025. [R1]

### Fleet allocation với short-turn/interlining

Gkiotsalitis, Wu và Cats xem xét tạo các lựa chọn short-turn/interlining và phân bổ fleet để cân bằng thời gian chờ với chi phí vận hành. Bài dùng mô hình tổ hợp và genetic algorithm, nên đây là nguồn về bài toán/baseline chứ không phải bằng chứng hiệu quả của RL. [R2]

### Deadheading

Bài *The real-time deadheading problem in transit operations control* nghiên cứu chọn xe chạy rỗng và số trạm bỏ qua nhằm giảm passenger cost trong điều kiện headway bất thường. Dự án chỉ mượn ý tưởng accounting cho thời gian xe chạy rỗng; không cho bus chở khách biến mất hoặc dịch chuyển tức thời. [R3]

### RL và departure intervals

*Deep Reinforcement Learning based Dynamic Optimization of Bus Timetable* nghiên cứu DQN cho điều chỉnh departure intervals theo dòng khách. Điều này là tiền lệ trực tiếp hơn neural route design, nhưng phạm vi fleet/reassignment của dự án cần simulator và kiểm chứng riêng. Không dùng mức cải thiện paper làm mục tiêu đảm bảo cho dự án. [R4]

## 4. Formulation phù hợp với đồ án

Ba mức có thể triển khai:

| Mức | Khả năng | Vai trò |
|---|---|---|
| M1 | Dispatch reserve + recall + target headway trên tuyến có sẵn | Milestone nhỏ để kiểm chứng fleet accounting và RL |
| M2 | Thêm reassign xe rỗng giữa tuyến tại bến, có travel time | Bài toán phân bổ nguồn lực liên tuyến đầy đủ hơn |
| M3 | Thêm short-turn mission trên đoạn được phép | MVP hoàn chỉnh theo định nghĩa mới |

Forecast đơn giản từ dữ liệu lịch sử là experiment bổ sung sau M3, không cần train một deep forecasting model trước khi train controller. Không xem M1 là đã hoàn thành toàn bộ đề tài. Không triển khai thiết kế tuyến mới hoặc multi-agent từng xe ở giai đoạn này.

Các mặc định trong spec (3 tuyến, 12 xe, 4 giờ) là quyết định bắt đầu có thể thay bằng config; người dùng chưa đưa hạn mức compute hay lịch nộp cụ thể.

## 5. Synthetic data cần chứa gì?

Một scenario gồm:

1. Tuyến, thứ tự trạm hai chiều, thời gian cạnh và các điểm quay đầu được phép.
2. Đội xe, sức chứa, vị trí ban đầu, nhiệm vụ và điều kiện tương thích.
3. Dòng khách theo origin, destination, route, direction và thời điểm đến.
4. Traffic field theo edge và time bucket; deadhead dùng đường đi được phép.
5. Sự kiện có thể tạo peak, trường tan học, metro đổ khách hoặc chậm xe.

Có thể sinh số khách mỗi tick bằng:

\[
A_{r,d,i,k}\sim\operatorname{Poisson}(\lambda_{r,d,i}(t_k)\,\delta),
\]

trong đó đơn vị của λ và δ phải khớp. λ kết hợp nền, giờ cao điểm và hotspot theo trạm. Sau đó sinh destination ở phía trước theo hướng đi. Metro trong MVP là nguồn khách ngoại sinh tại trạm kết nối; không điều khiển đoàn tàu metro.

Poisson là baseline có kiểm soát, không phải khẳng định hành khách thực đến độc lập. Batch arrivals tại metro/trường học nên được mô hình hóa thành nhóm hoặc burst có overdispersion, rồi đặt thành OOD test.

**Phân biệt ba thứ:** generator sinh thế giới; simulator phản ứng với hành động; forecaster chỉ học từ dữ liệu quá khứ. Không huấn luyện controller trên một công thức λ rồi cung cấp cho nó toàn bộ λ tương lai như một dự báo “thực tế”.

## 6. Simulator và conservation là phần khó nhất

Mỗi bus phải nằm đúng một trạng thái: depot, deadhead, terminal idle, service moving/dwell hoặc layover. Xe chỉ chuyển route ở bến hợp lệ khi rỗng và sẵn sàng. Short-turn chỉ áp dụng nhiệm vụ mới, không ép người đang trên xe xuống trước đích.

Cần đảm bảo ở mọi tick:

\[
N_{generated}=N_{waiting}+N_{onboard}+N_{completed}+N_{abandoned},
\]

\[
N_{depot}+N_{deadhead}+N_{terminal}+N_{service}+N_{layover}=N_{fleet}.
\]

Waiting time phải tích phân số khách thực sự chờ qua thời gian. Boarding/alighting là sự kiện, không phải demand tự mất khỏi queue. Người bị từ chối lên do đầy vẫn ở queue cho đến khi lên được hoặc hết patience.

MVP dùng fixed-tick simulator 30 giây và control mỗi 2 phút. Đây là xấp xỉ thời gian có thể kiểm chứng; không cần SUMO trước khi mô hình conservation đúng.

## 7. Reward engineering

Các thành phần có vai trò riêng:

- Queue passenger-minutes: chờ của mọi người, kể cả người không được phục vụ.
- In-vehicle passenger-minutes: tránh đón sớm rồi giữ trên xe để làm đẹp waiting metric.
- Comfort overload: số người vượt mức thoải mái, vẫn dưới hard capacity.
- Operating/deadhead bus-minutes: chi phí nguồn lực thực, không chỉ đếm nút dispatch.
- First denied boarding: đánh dấu lần đầu, không đếm cùng người mỗi tick.
- Abandonment và unfinished cuối episode: tránh trì hoãn khách đến sau horizon.
- Excessive-wait passenger-minutes: tăng ưu tiên cho đuôi phân phối chờ.

Không dùng variance waiting giữa tuyến làm fairness chính: variance thấp cũng có thể đạt bằng làm mọi tuyến đều tệ. Dùng ngưỡng chất lượng, worst-route metrics và ràng buộc bảo vệ tuyến cho xe là cách dễ diễn giải hơn. Reward vẫn không bảo đảm fairness tuyệt đối.

Dense reward ở bài này là âm chi phí phát sinh trong interval; công thức chênh lệch J của bài chọn K tuyến cũ không còn phù hợp.

## 8. State, partial observability và forecast

Simulator biết destination từng hành khách và demand latent; bộ điều khiển không nhất thiết biết. Q, load, boarding/alighting history không đủ để gọi observation là Markov state đầy đủ.

Dự án phân biệt **state simulator** và **observation controller**; với aggregate sensors, đây là bài điều khiển quan sát không đầy đủ. Baseline dùng feed-forward MaskablePPO với cửa sổ lịch sử; không tuyên bố đã giải POMDP tối ưu. Recurrent policy là mở rộng cần integration masking riêng, không giả định MaskablePPO sẵn có recurrent support.

Forecast baseline: trung bình lịch sử theo time-bin từ tập train, kết hợp arrivals gần đây. Chỉ dùng observations có timestamp ≤ t. Đánh giá MAE và bias trên demand-only held-out days trước khi đưa forecast vào controller. Perfect-future forecast nếu làm chỉ là oracle diagnostic, gắn nhãn riêng, không baseline online công bằng.

## 9. Giao thức thực nghiệm

Baselines cốt lõi:

- Fixed assignment/headway, không intervention.
- Threshold queue controller có reserve/reassign guards.
- Demand-proportional allocation dùng cùng sensors/forecast khả dụng.
- Random valid action để kiểm tra env, không phải baseline chất lượng duy nhất.

So sánh trên cùng scenario và exogenous demand/traffic tapes. Không dùng một RNG stream bị thay số lần gọi theo action vì sẽ khiến mỗi policy gặp thời tiết/demand khác nhau. Traffic noise nên indexed theo `(edge, time_bucket, scenario_seed)`.

Primary quality: mean waiting trên toàn bộ demand có báo censoring, completed share và toàn bộ cost. Kèm P95 wait, worst-route excessive-wait share, denied/abandoned, deadhead, utilization, actual headway và intervention count. Không chỉ báo waiting của khách đã lên.

Ablations bắt buộc của pipeline cuối: bỏ reassign, bỏ short-turn; reward có/không fairness ở cùng action guards. Forecast/no-forecast là ablation riêng sau core. Một experiment thay nhiều thành phần cùng lúc không thể gán cải thiện cho một thành phần.

## 10. Giới hạn của kết luận

MVP giả định sensing hàng chờ/tải chính xác, tuyến riêng không đổi, không passenger transfer, tất cả tài xế khả dụng trong ca và compatibility được khai báo. Không có labor rostering, congestion nội sinh, tín hiệu giao thông hoặc hành vi hành khách đổi tuyến.

Thực nghiệm synthetic chỉ chứng minh trong simulator đã định nghĩa. Kết quả không đủ để tuyên bố triển khai được cho một hệ thống vận tải thực hoặc thuật toán tốt nhất.

## 11. Nguồn và mức kiểm chứng

Đối chiếu ngày 2026-09-12. Các quyết định mô hình trong spec là của dự án; các nguồn dưới đây hỗ trợ bối cảnh hoặc API, không phải chứng nhận thiết kế.

| ID | Nguồn | Đã xem/dùng |
|---|---|---|
| R1 | Zahedi, Koutsopoulos, Ma, [Dynamic interlining in bus operations](https://link.springer.com/article/10.1007/s11116-023-10440-x), online 2023 / journal 2025 | Bài open-access, abstract/introduction; shared hub scope |
| R2 | Gkiotsalitis, Wu, Cats, [A cost-minimization model for bus fleet allocation featuring the tactical generation of short-turning and interlining options](https://trid.trb.org/View/1570024), 2019; [DOI](https://doi.org/10.1016/j.trc.2018.11.007) | TRID abstract do publisher cung cấp, không giả vờ đã tái hiện full paper |
| R3 | [The real-time deadheading problem in transit operations control](https://www.sciencedirect.com/science/article/abs/pii/S0191261597000131), 1998 | Publisher search abstract; mở trực tiếp trang có lỗi; không dùng nội dung full text |
| R4 | [Deep Reinforcement Learning based Dynamic Optimization of Bus Timetable](https://arxiv.org/abs/2107.07066), 2021 | Abstract; DQN/interval control |
| R5 | [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017 | PPO theory |
| R6 | [Invalid Action Masking in Policy Gradient Algorithms](https://arxiv.org/abs/2006.14171), 2020 | Masking |
| R7 | [SB3-Contrib MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html) | Library integration và evaluation |
| R8 | [Gymnasium: handling time limits](https://gymnasium.farama.org/main/tutorials/handling_time_limits/) | Finite horizon, termination/truncation |

Các nguồn TNDP ở v0.1 được giữ trong Git history; không còn là nền tảng cho mục tiêu vận hành hiện tại. Chưa tải dataset hoặc chạy code của tác giả.
