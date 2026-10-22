# standard-v1

Một báo cáo tổng hợp cho MLP10, GRU10 và Belief20-fade.

- [Báo cáo Markdown](report.md): đọc ngay trong editor.
- [Báo cáo HTML](report.html): bảng và biểu đồ, tự chứa hình ảnh.
- [Báo cáo PDF](report.pdf): bản 6 trang để đọc, in hoặc chia sẻ.

Ba định dạng dùng cùng nội dung và số liệu; bản Markdown trình bày bằng bảng,
HTML/PDF có thêm biểu đồ. Báo cáo gồm thiết lập chạy, từng điều kiện test,
kết quả cuối và cách tính. Không cần mở các file dữ liệu để đọc báo cáo.

`evidence/` giữ dữ liệu gốc: config, protocol, manifest kịch bản, metadata/lịch sử
train, kết quả từng ngày và các bảng tổng hợp. Không chứa weights hoặc logs.
Các report riêng từng policy đã được thay bằng báo cáo chung ở trên.

Kiểm tra tính toàn vẹn từ thư mục gốc dự án:

```bash
uv run bus-rl artifact verify --path runs/standard-v1
```

Tạo lại cả ba định dạng và dọn lại bundle:

```bash
uv run --with reportlab python scripts/build_study_report.py --study runs/standard-v1
```

Chương trình kiểm tra dữ liệu, dựng đủ các định dạng ở thư mục tạm, rồi mới thay
bundle hiện tại; bản trước đó được giữ trong `/tmp/bus-study-previous-*/`.
Dữ liệu này là snapshot tái lập trên bộ test tham chiếu đã dùng trước đây;
không tự thay thế kết luận lịch sử ở `reports/`.
