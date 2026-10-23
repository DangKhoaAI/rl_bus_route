# Nguồn số liệu của trang giới thiệu

Bảng nguồn và trạng thái kiểm chứng cho từng con số trong
[`web/intro/index.html`](../../web/intro/index.html), kiểm ngày 2026-10-22.
Số Hình/Bảng/Bảng đánh số theo trang giới thiệu. Đường dẫn tính từ gốc repo;
`rerun/` = `reports/provenance/rerun-2026-10-22/`.

- Ba luật viết tay: chạy lại ngày 2026-10-22, khớp bảng đóng băng tới chữ số cuối.
- PPO L3: checkpoint không còn; số lấy từ bảng đóng băng.
- Huấn luyện lại cùng cấu hình (core-11): validation 13.428,0 (L3 seed 11: 13.199,1), ngày thường 13.371,2, kém L3 ≈ 2,3%. Dùng cho Hình 4 và Hình 8.

| Số liệu | Nguồn | Trạng thái | Lệnh tái lập |
|---|---|---|---|
| Fixed, Threshold, Proportional: chi phí, thời gian chờ | `reports/tables/baselines_summary.csv`, `main_benchmark.csv` | chạy lại, khớp tới chữ số cuối | `uv run bus-rl baseline --config configs/eval.toml --manifest data/generated/core/manifest.json --split <split> --output <dir>` |
| PPO (L3): chi phí, thời gian chờ, chi phí theo ngày | `main_benchmark.csv`, `paired_days.csv` | bảng đóng băng; checkpoint không còn | không chạy lại được |
| Hiệu ghép cặp ba luật viết tay (Bảng 4) | `baseline_contrasts.csv` | bảng đóng băng | — |
| core-11: validation 13.428,0 | `runs/core-11/metadata.json` | đọc trực tiếp | — |
| core-11: ngày thường 13.371,2; trace ngày đầu (Hình 8) | `rerun/core11-test_id-results.csv`, `core11-test_id-day0-events.jsonl` | chạy lại | `uv run bus-rl evaluate --config configs/eval.toml --manifest data/generated/core/manifest.json --split test_id --checkpoint runs/core-11/best.zip --output <dir> --trace` |
| Mặt nạ phút 0 và 60 (Hình 4) | `rerun/core11-test_id-day0-masks.csv` | chạy lại; chuỗi lệnh khớp trace Hình 8 | `uv run python reports/provenance/rerun-2026-10-22/core11_day0_masks.py` |
| Fixed ngày đầu: 15.444,6 (Hình 8) | `runs/standard-v1/evidence/baselines/shared/eval/test_id/results.csv` | đọc trực tiếp | — |
| Trace ngày đầu của Threshold, Proportional (Hình 8) | `rerun/baselines-test_id-day0-*.{csv,jsonl}` | chạy lại | lệnh `baseline` thêm `--trace` |
| Loại lệnh của L3, 27 ngày | `reports/provenance/l3-failure-traces.json` | bảng đóng băng | — |
| Tỷ lệ bỏ về 1,8% (burst) | `main_benchmark.csv` | suy ra: 1 − completed_share = 1,82% | — |
| Siêu tham số, hệ số chi phí, ràng buộc | `configs/train.toml`, `costs.rs`, `guards.rs` | đọc trực tiếp từ mã | — |
| Hình 3, 5, 7 | công thức trong trang | minh họa | — |
