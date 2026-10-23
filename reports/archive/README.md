# Archive — retained evidence

Chỉ giữ kết luận, bảng tổng hợp và protocol có giá trị đối chiếu.

| Thư mục | Lý do giữ |
|---|---|
| [historical-core](historical-core/README.md) | Benchmark và ablation core cũ, protocol, kiểm chứng và ledger |
| [belief-gru-ablation](belief-gru-ablation/README.md) | Thử nghiệm encoder: kết luận, summary, contrasts và protocol |
| [memory-belief-ablation](memory-belief-ablation/README.md) | Thử nghiệm cửa sổ nhớ/belief: kết luận, summary, contrasts và protocol |
| [rejected-changes](rejected-changes/README.md) | Bằng chứng các thay đổi đã thử và không được chọn |
| [infrastructure](infrastructure/README.md) | Kiểm chứng Rust và tối ưu runtime đã chấp nhận |

Pilot 12k chỉ kiểm tra pipeline (PPO validation cost 18,207 so với fixed 16,017); không phải bằng chứng chất lượng policy. Các bản đo tốc độ Python cũ, pilot, hình trùng, telemetry và dữ liệu từng ngày đã được bỏ khỏi archive.
Có thể xem lại toàn bộ bản trước khi dọn tại commit `782f3e4` trong Git.
Dữ liệu từng ngày của study hiện tại vẫn giữ ở `runs/standard-v1/evidence/`.

[Báo cáo cuối hiện tại](../README.md).

Các script benchmark/phân tích thí nghiệm cũ đã được dọn khỏi `scripts/`; các lệnh lịch sử có thể dùng lại với source ở Git commit `fbb1c83`. Các lệnh hiện tại nằm trong `scripts/README.md` và CLI `uv run bus-rl`.
