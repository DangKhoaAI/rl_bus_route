# Scripts và lệnh hiện tại

Chạy từ thư mục gốc repository. Các luồng chính dùng `uv run bus-rl`, với implementation ở `src/bus_rl/cli.py`.

## Script được giữ

| Script | Mục đích |
|---|---|
| `build_native.py` | Build/install kernel Rust; cần cho train, evaluate và web demo |
| `build_study_report.py` | Dựng bộ HTML/Markdown/PDF của study từ evidence đã có |
| `extend_standard_study.py` | Ghép study thuật toán/masking hoàn tất vào standard-v1, kiểm tra và giữ nguyên evidence gốc |
| `capture_masking_decisions.py` | Ghi ví dụ quyết định cố định cho báo cáo masking |

Không cần dùng các script study khi chỉ train/evaluate một model.
Các script benchmark Python/Rust đời cũ, tuning, L3 analysis và encoder ablation đã được bỏ khỏi cây hiện tại. Có thể xem lại ở Git commit `fbb1c83`; các lệnh trong tài liệu archive là lệnh lịch sử.

## Setup

```bash
uv sync --locked --extra dev
uv run python scripts/build_native.py
```

Cần Python 3.11 và Rust/Cargo. Mỗi output dữ liệu/train/eval/report cần là đường dẫn mới.

## Data → train → evaluate → report

```bash
uv run bus-rl data generate --config configs/base.toml \
  --counts train=500,validation=100,test_id=200,test_ood_burst=200,test_ood_traffic=200 \
  --output data/generated/base

uv run bus-rl model train --config configs/train.toml \
  --manifest data/generated/base/manifest.json \
  --output experiments/demo/train

uv run bus-rl model evaluate --config configs/eval.toml \
  --manifest data/generated/base/manifest.json \
  --checkpoint experiments/demo/train/best.zip --split test_id \
  --output experiments/demo/eval

uv run bus-rl baseline run --config configs/eval.toml \
  --manifest data/generated/base/manifest.json --split test_id \
  --methods fixed,proportional,threshold --output experiments/demo/baselines

uv run bus-rl report generate \
  --results experiments/demo/eval/results.csv experiments/demo/baselines/results.csv \
  --output experiments/demo/report
```

Lặp `model evaluate` và `baseline run` với `test_ood_burst` / `test_ood_traffic` và output riêng để đánh giá thêm hai case. `report generate` nhận nhiều CSV, tạo HTML; không train hoặc evaluate lại.

## Test và kiểm tra

```bash
uv run pytest -q
uv run ruff check .
uv run bus-rl test diagnose --help
uv run bus-rl test profile --help
```

`pytest` kiểm tra tính đúng; `test diagnose` / `test profile` là công cụ chẩn đoán hiệu năng, không thay thế test suite.

## Study đầy đủ và artifact

```bash
uv run bus-rl experiment run --config configs/studies/standard.toml \
  --manifest data/generated/base/manifest.json \
  --output experiments/new-study --artifacts runs/new-study

uv run --with reportlab python scripts/build_study_report.py --study runs/new-study
uv run bus-rl artifact verify --path runs/new-study
```

`build_study_report.py` dựng lại báo cáo trong bundle nguồn, không phải lệnh gom file sang `reports/`. Dùng đường dẫn mới để giữ nguyên bộ kết quả đã chốt.

Tra các tùy chọn bằng `uv run bus-rl <nhóm> <lệnh> --help`.
