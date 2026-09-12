# Research: Reinforcement Learning cho thiết kế mạng tuyến bus cố định

Ngày đối chiếu nguồn: **2026-09-12**. Trạng thái: nghiên cứu và đề xuất thiết kế; chưa có code, thí nghiệm hoặc kết quả huấn luyện của dự án.

Tài liệu liên quan: [đặc tả dự án](../spec.md), [kế hoạch thực hiện](../plan.md).

## 1. Bài toán và câu hỏi nghiên cứu

**Transit Network Design Problem (TNDP)**: cho một mạng đường, các trạm tiềm năng và nhu cầu đi lại origin–destination (OD), tìm một tập tuyến bus phục vụ nhu cầu với chất lượng hành trình tốt và chi phí vận hành hợp lý.

Một nghiệm là **cả mạng tuyến**, không phải đường ngắn nhất của một xe. Hành khách có thể đi trực tiếp hoặc chuyển tuyến. Hai trạm cùng xuất hiện trong mạng chưa có nghĩa là đi được giữa chúng trong giới hạn số lần chuyển tuyến.

Phạm vi đã chọn là **tuyến cố định**. Thời gian trong một episode RL là thứ tự quyết định thiết kế mạng, không phải thời gian xe chạy ngoài đường. Không dùng mô hình đón/trả từng yêu cầu của dial-a-ride làm định nghĩa dự án.

Câu hỏi chính:

> Với cùng tập tuyến ứng viên và ngân sách, policy học bằng RL có tạo mạng tuyến có chi phí tổng hợp thấp hơn greedy và random, đồng thời duy trì chất lượng trên thành phố tổng hợp chưa thấy không?

Câu hỏi phụ:

1. Reward phản ánh cả hành khách và vận hành tạo ra đánh đổi gì so với reward chỉ tối đa độ phủ?
2. Dense reward dựa trên chênh lệch objective có cải thiện hiệu quả học so với terminal reward không?
3. Policy suy giảm thế nào khi thay phân bố OD, độ tập trung nhu cầu hoặc topology?
4. Chất lượng bị giới hạn bao nhiêu bởi tập tuyến ứng viên, thay vì bởi thuật toán học?

Đây là giả thuyết cần thực nghiệm; không giả định RL chắc chắn vượt heuristic.

## 2. Nền tảng tài liệu

### 2.1 RL cho transit network design

Holliday, El-Geneidy và Dudek nghiên cứu GNN được huấn luyện bằng RL để hướng dẫn thay đổi mạng tuyến trong tìm kiếm tiến hóa. Bản v6 sử dụng PPO và đánh giá trên các benchmark transit. Điểm rút ra cho dự án là khả năng kết hợp biểu diễn graph, objective hành khách–vận hành và heuristic học được; thiết kế MVP dưới đây không phải bản tái hiện phương pháp đó. [R1]

Luận án của Holliday trình bày construction MDP và kết hợp policy với metaheuristic. Đây là nguồn để tìm hiểu sự khác nhau giữa xây nghiệm và cải thiện nghiệm. [R2]

### 2.2 Neural combinatorial optimization

Kool và cộng sự dùng attention cùng REINFORCE với greedy rollout baseline cho các bài toán routing. Đây là tiền lệ cho học heuristic từ instance mà không cần nhãn nghiệm tối ưu; kết quả ở TSP/VRP không tự động chuyển sang TNDP có chuyển tuyến. [R3]

### 2.3 PPO và action masking

PPO dùng surrogate objective có clipping để hạn chế thay đổi policy trong cập nhật. Dự án ưu tiên thư viện đã có triển khai thay vì tự viết optimizer RL. [R4]

Invalid action masking loại các lựa chọn không hợp lệ trước khi lấy mẫu từ policy. SB3-Contrib cung cấp MaskablePPO; khi đánh giá phải truyền mask và dùng callback/evaluation hỗ trợ masking. Đây là khác biệt quan trọng so với chỉ phạt hành động sai bằng reward. [R5, R6]

## 3. Ba thiết kế thuật toán có thể chọn

| Thiết kế | Action | Ưu điểm | Hạn chế |
|---|---|---|---|
| **Chọn tuyến ứng viên — MVP được chọn** | Thêm một tuyến từ pool vào mạng | Episode ngắn; kiểm tra ràng buộc rõ; baseline công bằng | Chỉ tìm nghiệm trong pool; không học trực tiếp hình dạng tuyến |
| Tự xây tuyến từng trạm | Chọn trạm kế tiếp, kết thúc tuyến | Không phụ thuộc hoàn toàn vào pool | Horizon dài; dead end; mask và credit assignment khó |
| Cải thiện mạng có sẵn | Thay tuyến, thêm/bớt trạm | Tận dụng nghiệm heuristic; gần nghiên cứu hybrid | Cần thiết kế move set và ngân sách tìm kiếm |

**Quyết định:** hoàn thành pipeline candidate-selection trước. Nếu thí nghiệm cho thấy candidate pool là nút thắt, mới bổ sung tuyến đi qua waypoint hoặc construction policy. Không gọi MVP là bộ giải TNDP không hạn chế.

MVP vẫn có quyết định tuần tự: tuyến đầu có thể chưa có nhiều lợi ích riêng lẻ nhưng tạo điểm chuyển tuyến hữu ích khi thêm tuyến thứ hai. Greedy theo lợi ích tức thời có thể bỏ lỡ hiệu ứng này. Đây là động cơ hợp lý để thử RL, chưa phải bằng chứng RL tốt hơn.

## 4. Synthetic data có khả thi không?

**Có.** Đầu vào cơ bản là graph và OD; không cần dữ liệu nhãn “mạng tuyến tối ưu”. Nhưng cần phân biệt:

- **Đúng cấu trúc:** graph liên thông, trọng số dương, OD không âm, tuyến đi qua các cạnh có thật.
- **Đủ đa dạng:** không chỉ một grid hoặc OD đều.
- **Gần thực tế:** cần hiệu chỉnh và kiểm chứng bằng nguồn thực; MVP chưa đưa ra tuyên bố này.

### 4.1 Thành phố tổng hợp

Đề xuất riêng của dự án:

1. Sinh vị trí trạm trong một vùng chuẩn hóa.
2. Tạo hai họ topology: grid có jitter và graph hình học k-nearest-neighbor được nối bằng minimum spanning tree.
3. Sinh thời gian cạnh dương theo khoảng cách và nhiễu nhỏ. Cố định trọng số trong một instance.
4. Gắn vai trò dân cư, việc làm và dịch vụ cho trạm.
5. Sinh OD từ độ hút của nơi đến và trở kháng thời gian đi đường, sau đó chuẩn hóa về tổng nhu cầu.

Mô hình OD gợi ý:

\[
w_{ij}=P_i A_j \exp(-d^G_{ij}/\sigma)\eta_{ij},\quad i\ne j
\]

\[
D\sim\mathrm{Multinomial}\left(Q,\{w_{ij}/\sum_{u\ne v}w_{uv}\}\right),\quad D_{ii}=0.
\]

\(P_i\) là nguồn phát sinh, \(A_j\) là sức hút; \(d^G\) là thời gian ngắn nhất trên mạng đường; \(\eta\) là nhiễu dương. Công thức là giả định generator, không phải mô hình nhu cầu đã được hiệu chỉnh.

### 4.2 Các chế độ nhu cầu

| Chế độ | Mục đích |
|---|---|
| Uniform OD | Kiểm tra tính trung lập và baseline đơn giản |
| Residential → employment | Tạo nhu cầu hướng tâm/bất đối xứng |
| Multi-center | Buộc mạng phục vụ nhiều cụm |
| Shifted demand | Kiểm tra ngoài phân phối huấn luyện |

Không dùng GAN hoặc LLM để sinh số liệu trong MVP: procedural generation đủ để kiểm soát seed, thông số và giải thích cấu trúc dữ liệu.

### 4.3 Feasibility và chia tập

- Bảo đảm tồn tại K tuyến ứng viên khác nhau trong ngân sách bằng kiểm tra tổng chi phí K tuyến rẻ nhất.
- Điều kiện này **không bảo đảm phục vụ 100% OD**; độ phủ là mục tiêu mềm và phải báo cáo riêng.
- Split theo thành phố gốc trước khi tạo biến thể OD; không để cùng graph ở train và test rồi tuyên bố tổng quát topology.
- Lưu seed, generator version, config và hash graph trong manifest.
- Có tập tiny để tính nghiệm tối ưu bằng enumeration trong pool; đây là oracle kiểm chứng, không phải bộ train chính.

## 5. Mô hình hành khách là thành phần cần đầu tư nhất

Không tính thời gian hành khách bằng shortest path trên toàn bộ mạng đường sau khi chọn tuyến. Cách đó cho phép đi qua đường không có bus và bỏ qua phí chuyển tuyến.

Đề xuất dùng graph nhiều lớp theo tuyến và số lần chuyển tuyến:

- Đi trên một tuyến: cộng thời gian cạnh liên tiếp.
- Lên tuyến đầu: cộng thời gian chờ kỳ vọng \(h/2\).
- Chuyển tuyến tại trạm chung: cộng \(h/2+\tau\), tăng bộ đếm chuyển tuyến.
- Giới hạn hai lần chuyển; nếu không tồn tại hành trình, OD đó được coi là chưa phục vụ.

Giả định \(h/2\) phù hợp với mô hình hành khách đến ngẫu nhiên so với headway đều; không biểu diễn timetable đồng bộ. Không mô phỏng tải xe, xếp hàng hay ùn tắc nội sinh. Thời gian chờ là số kỳ vọng theo giả định, không phải quan sát từ event simulator.

Đây là evaluator xác định cho thiết kế mạng. Simulator vi mô như SUMO không cần cho giai đoạn chứng minh pipeline; có thể nghiên cứu sau khi evaluator đơn giản đã được kiểm chứng.

## 6. Reward và rủi ro đánh giá

Ưu tiên định nghĩa objective cuối cùng trước rồi chuyển sang reward. Spec quy định chi phí hành khách có phạt OD chưa phục vụ, chi phí tuyến chuẩn hóa và penalty độ phủ.

| Sai lầm | Hậu quả | Biện pháp |
|---|---|---|
| Chỉ tính thời gian của khách được phục vụ | Bỏ OD khó để làm đẹp trung bình | Tính chi phí cho mọi OD, báo cáo coverage |
| Coi mọi đường nối là bus | Đánh giá quá lạc quan | Graph theo tuyến và transfer layer |
| Thêm reward phủ và reward thời gian tùy ý | Mục tiêu học lệch objective | Dense reward bằng chênh lệch cùng một objective |
| Cho RL pool lớn hơn baseline | So sánh không công bằng | Dùng cùng pool, budget và evaluator |
| Chọn checkpoint theo test | Rò rỉ dữ liệu | Validation riêng, test sau khi khóa lựa chọn |
| So số cuối paper khác cost/headway | Kết luận vượt nghiên cứu thiếu căn cứ | Chỉ so trực tiếp khi giao thức tương thích |

## 7. Thực nghiệm có giá trị học thuật

Mức tối thiểu:

1. Random feasible, greedy marginal objective, greedy + one-route swap search.
2. Maskable PPO trên cùng input/action space.
3. Dense reward so với terminal-only reward, cùng terminal objective.
4. Ít nhất ba seed huấn luyện; paired evaluation trên cùng test instances.
5. Báo cáo objective, coverage OD có trọng số, thời gian hành trình, chuyển tuyến, chiều dài tuyến và latency.

RL không thắng greedy vẫn là kết quả có ích nếu làm rõ do pool hạn chế, không đủ mẫu, reward hay model thiếu biểu diễn. Không đặt điều kiện nghiệm thu là “phải thắng”. Điều kiện nghiệm thu là pipeline đúng, kết quả tái lập và kết luận có bằng chứng.

## 8. Compute, giới hạn và hướng mở rộng

CPU đủ để bắt đầu kiểm tra graph, evaluator và tiny cases. Chưa có phép đo để cam kết thời gian training. Cần pilot đo env steps/s, thời gian evaluator, bộ nhớ và tốc độ policy trước khi chọn CPU/GPU hay tăng ngân sách.

Ưu tiên tối ưu cache theo tập tuyến đã chọn vì nhiều thứ tự hành động dẫn tới cùng mạng. Nếu evaluator chiếm phần lớn thời gian, GPU lớn hơn có thể không giúp đáng kể.

Mở rộng có điều kiện:

- Shared route scorer/attention để giảm phụ thuộc thứ tự tuyến ứng viên.
- GNN và policy tự xây tuyến, khi MVP đã có baseline mạnh.
- Dataset Mandl/Mumford để đánh giá ngoài generator; phải kiểm tra nguồn, quyền sử dụng, quy ước route và objective trước khi nhập.
- Mạng đường thực và OD được hiệu chỉnh; chưa đưa vào cam kết MVP.
- Tối ưu frequency/fleet/capacity là một bài toán mở rộng cần spec mới.

## 9. Nguồn tham khảo và phạm vi sử dụng

Nguồn đã đọc qua trang bài báo/tài liệu chính thức; chưa tải dataset, clone hay chạy repo của tác giả. Các quyết định MVP là đề xuất của dự án, không phải thông số sao chép từ paper.

| ID | Nguồn | Dùng cho |
|---|---|---|
| R1 | Holliday et al., [Learning Heuristics for Transit Network Design and Improvement with Deep Reinforcement Learning, v6](https://arxiv.org/html/2404.05894v6), 2025; bản đầu 2024 | Bối cảnh TNDP, RL/hybrid và benchmark |
| R2 | Holliday, [Applications of deep reinforcement learning to urban transit network design](https://arxiv.org/abs/2502.17758), 2025 | Construction MDP và hướng hybrid |
| R3 | Kool et al., [Attention, Learn to Solve Routing Problems!](https://arxiv.org/abs/1803.08475), ICLR 2019 | Neural routing và REINFORCE |
| R4 | Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017 | PPO clipped objective |
| R5 | Huang & Ontañón, [A Closer Look at Invalid Action Masking in Policy Gradient Algorithms](https://arxiv.org/abs/2006.14171), 2020 | Action masking |
| R6 | [SB3-Contrib: Maskable PPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html) | Integration và masked evaluation |
| R7 | [Gymnasium Env API](https://gymnasium.farama.org/api/env/) | reset/step, termination/truncation |
| R8 | [NetworkX shortest_path](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.generic.shortest_path.html) | Weighted shortest paths |
| R9 | [PyTorch installation](https://pytorch.org/get-started/locally/) | Chọn wheel theo phần cứng |
| R10 | [uv locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/) | Lock môi trường tái lập |

Các URL `stable`/`master` có thể thay đổi; lúc implementation phải lưu phiên bản dependency thực tế và lockfile. Không xem tài liệu web hiện tại là bằng chứng môi trường dự án đã cài hoặc chạy được.
