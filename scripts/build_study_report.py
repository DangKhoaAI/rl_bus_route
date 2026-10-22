#!/usr/bin/env python
"""Build a compact study bundle: README + one report in Markdown, HTML and PDF.

Run: uv run --with reportlab python scripts/build_study_report.py --study runs/standard-v1
Original evidence is retained in evidence/; per-policy presentation copies are omitted.
The complete replacement is built and verified before moving the previous bundle to /tmp.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
import tempfile
from datetime import UTC, datetime
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from bus_rl.artifacts import verify_artifacts
from bus_rl.evaluation.report_data import compare_baselines, summarize_results, validate_results
from bus_rl.provenance import canonical_hash, file_hash

POLICIES = ["mlp10", "gru10", "belief20-fade"]
LABELS = {"mlp10": "MLP10", "gru10": "GRU10", "belief20-fade": "Belief20-fade"}
SPLITS = {
    "test_id": (
        "Nhu cầu bình thường",
        "Nhu cầu và giao thông theo cấu hình nền của bộ mô phỏng, với các seed kịch bản riêng cho test.",
    ),
    "test_ood_burst": (
        "Nhu cầu tăng đột biến",
        (
            "Tại một điểm dừng ngẫu nhiên, thêm 8-20 hành khách mỗi tick trong 5-10 phút; "
            "thời điểm bắt đầu nằm trong phút 30-150. Tick mô phỏng là 30 giây."
        ),
    ),
    "test_ood_traffic": (
        "Gián đoạn giao thông",
        "Thời gian di chuyển trên hành lang của một tuyến ngẫu nhiên tăng 1,5-2,0 lần trong 30 phút.",
    ),
}
BASELINES = {"fixed": "Fixed", "proportional": "Proportional", "threshold": "Threshold"}
COLORS = {"MLP10": "#087f8c", "GRU10": "#4567a8", "Belief20-fade": "#ad6843"}


def fmt(value, digits=1):
    return f"{value:,.{digits}f}"


def table(headers, rows, widths):
    return ("table", (headers, rows, widths))


def image(plot):
    buffer = io.BytesIO()
    plot.savefig(buffer, format="png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(plot)
    return ("image", buffer.getvalue())


def savings_plot(comparisons):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    tick_labels = []
    for i, split in enumerate(SPLITS):
        for j, policy in enumerate(POLICIES):
            candidates = comparisons[(comparisons.policy == policy) & (comparisons.split == split)]
            best = candidates.loc[candidates.baseline_cost.idxmin()]
            y = i * 4 + j
            lo, hi = best.cost_saved_pct_ci95_low, best.cost_saved_pct_ci95_high
            color = COLORS[LABELS[policy]]
            ax.plot([lo, hi], [y, y], color=color, lw=2)
            ax.scatter(best.cost_saved_pct, y, color=color, s=40, zorder=3)
            tick_labels.append((y, f"{split.replace('test_', '')} / {LABELS[policy]}"))
    ax.set_yticks([x[0] for x in tick_labels], [x[1] for x in tick_labels], fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="#555555", ls="--", lw=1)
    ax.set_xlabel("Cost reduction vs lowest-cost baseline (%) - paired 95% interval", fontsize=10)
    ax.grid(axis="x", alpha=0.2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return image(fig)


def condition_plot(rows):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.1))
    for ax, (metric, label) in zip(
        axes,
        [("total_cost_core", "Mean operational cost"), ("p95_wait", "Mean daily P95 wait (min)")],
    ):
        for i, row in enumerate(rows):
            color = COLORS.get(row["label"], "#8a929c")
            ax.barh(i, row[metric], color=color, height=0.6)
            ax.text(row[metric], i, f" {row[metric]:,.1f}", va="center", fontsize=8)
        ax.set_yticks(range(len(rows)), [row["label"] for row in rows], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, max(row[metric] for row in rows) * 1.22)
        ax.set_xlabel(label, fontsize=10)
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    fig.tight_layout(w_pad=2)
    return image(fig)


def load_data(evidence):
    """Recompute from untouched per-day results and check the published aggregate CSVs."""
    summary = pd.read_csv(evidence / "summary.csv")
    comparisons = pd.read_csv(evidence / "comparisons.csv")
    protocol = json.loads((evidence / "protocol.json").read_text())
    spec = protocol["plan"]["spec"]
    if set(spec["policies"]) != set(POLICIES) or set(spec["splits"]) != set(SPLITS):
        raise ValueError(
            "this report template requires the standard three-policy/three-condition study"
        )
    metadata = {}
    for policy in POLICIES:
        paths = sorted((evidence / policy).glob("*/eval/*/results.csv"))
        paths += sorted((evidence / "baselines/shared/eval").glob("*/results.csv"))
        frames = []
        for path in paths:
            sidecar = json.loads(path.with_name("evaluation.json").read_text())
            if sidecar["results_sha256"] != file_hash(path):
                raise ValueError(f"stale evaluation metadata: {path}")
            frames.append(
                pd.read_csv(
                    path,
                    dtype={
                        "scenario_seed": "string",
                        "model_seed": "string",
                        "scenario_hash": "string",
                    },
                )
            )
        raw = validate_results(pd.concat(frames, ignore_index=True))
        for actual, expected, keys in [
            (
                summarize_results(raw),
                summary[summary.policy == policy].drop(columns="policy"),
                ["split", "method"],
            ),
            (
                compare_baselines(raw, "ppo"),
                comparisons[comparisons.policy == policy].drop(columns="policy"),
                ["split", "baseline"],
            ),
        ]:
            pd.testing.assert_frame_equal(
                actual.sort_values(keys).reset_index(drop=True),
                expected.sort_values(keys).reset_index(drop=True),
                check_dtype=False,
                check_exact=False,
                rtol=1e-10,
                atol=1e-8,
            )
        for seed in spec["seeds"]:
            metadata[policy, seed] = json.loads(
                (evidence / policy / str(seed) / "train/metadata.json").read_text()
            )
    # Shared baseline rows must actually be identical across policy-level aggregates.
    for split in SPLITS:
        for method in BASELINES:
            rows = summary[(summary.split == split) & (summary.method == method)].drop(
                columns="policy"
            )
            if len(rows.drop_duplicates()) != 1:
                raise ValueError("shared baseline summaries disagree")
    return summary, comparisons, protocol, metadata


def document(evidence):
    summary, comparisons, protocol, metadata = load_data(evidence)
    configs = protocol["resolved_configs"]
    spec = protocol["plan"]["spec"]
    first = configs["mlp10"]
    physical, algo = first["physical"], first["algorithm"]
    seeds = spec["seeds"]
    overview = []
    for split in SPLITS:
        for policy in POLICIES:
            row = summary[
                (summary.policy == policy) & (summary.split == split) & (summary.method == "ppo")
            ].iloc[0]
            candidates = comparisons[(comparisons.policy == policy) & (comparisons.split == split)]
            best = candidates.loc[candidates.baseline_cost.idxmin()]
            overview.append(
                [
                    LABELS[policy],
                    SPLITS[split][0],
                    f"{fmt(row.total_cost_core)} +/- {fmt(row.seed_std_cost)}",
                    BASELINES[best.baseline],
                    f"{best.cost_saved_pct:.1f}%",
                    f"[{best.cost_saved_pct_ci95_low:.1f}; {best.cost_saved_pct_ci95_high:.1f}]",
                ]
            )
    pages = []
    pages.append(
        (
            "Thiết lập thực nghiệm",
            [
                (
                    "p",
                    (
                        "Báo cáo tổng hợp standard-v1: so sánh ba cấu hình bộ mã hóa quan sát của MaskablePPO "
                        "với ba bộ điều khiển Fixed, Proportional và Threshold trong cùng bộ mô phỏng xe buýt."
                    ),
                ),
                (
                    "p",
                    (
                        "Kết quả chính: cả ba cấu hình RL có cost thấp hơn các baseline ở cả ba điều kiện. "
                        "MLP10 có cost trung bình thấp nhất trong các cấu hình RL ở điều kiện bình thường và gián đoạn "
                        "giao thông; Belief20-fade thấp nhất khi nhu cầu tăng đột biến. Chênh lệch giữa các cấu hình RL "
                        "nhỏ và chỉ mang tính mô tả; báo cáo không khẳng định ưu thế thống kê giữa chúng."
                    ),
                ),
                table(
                    ["Hạng mục", "Thiết lập chung"],
                    [
                        [
                            "Mạng và đội xe",
                            f"{physical['route_count']} tuyến hai chiều, {physical['stops_per_route']} điểm dừng/tuyến; {physical['fleet_size']} xe, sức chứa {physical['capacity']} khách/xe",
                        ],
                        [
                            "Thời gian mô phỏng",
                            f"{physical['horizon_s'] // 60} phút/ngày; nhận khách trong {physical['demand_end_s'] // 60} phút đầu; tick {physical['tick_s']} giây; quyết định mỗi {physical['control_interval_s']} giây",
                        ],
                        [
                            "Dữ liệu",
                            f"{spec['counts']['train']} ngày train, {spec['counts']['validation']} ngày validation, {spec['counts']['test_id']} ngày cho mỗi điều kiện test",
                        ],
                        [
                            "Ngân sách và seed",
                            f"{algo['total_timesteps']:,} transitions/seed; seeds {', '.join(map(str, seeds))}",
                        ],
                        [
                            "PPO",
                            f"learning rate {algo['learning_rate']}; gamma {algo['gamma']}; GAE lambda {algo['gae_lambda']}; clip {algo['clip_range']}",
                        ],
                        [
                            "Rollout và tối ưu",
                            f"{algo['n_envs']} envs x {algo['n_steps']} bước; batch {algo['batch_size']}; {algo['n_epochs']} epochs; entropy {algo['ent_coef']}",
                        ],
                        [
                            "Runtime",
                            f"Rust, CPU; {algo['torch_threads']} Torch threads; đánh giá theo batch {first['runtime']['eval_batch_size']}",
                        ],
                        [
                            "Điều khiển",
                            "Giữ nguyên action masking, REASSIGN và SHORT_TURN; forecast và reward shaping tắt",
                        ],
                        [
                            "Chọn checkpoint",
                            f"Cost validation thấp nhất sau khi train; hòa thì chọn sớm nhất; đánh giá mỗi {algo['eval_freq']:,} transitions. Step 0 chỉ làm mốc tham chiếu.",
                        ],
                    ],
                    [0.24, 0.76],
                ),
                ("h3", "Ba cấu hình"),
                table(
                    ["Cấu hình", "Encoder / cửa sổ", "Điểm khác biệt"],
                    [
                        ["MLP10", "MLP / 10 phút", "Mạng feed-forward"],
                        ["GRU10", "GRU / 10 phút", "Mã hóa cửa sổ quan sát bằng GRU"],
                        [
                            "Belief20-fade",
                            "Belief / 20 phút",
                            "Đặc trưng rate Gamma-Poisson; prior shape/rate = 0.1/0.1; decay = 0.8",
                        ],
                    ],
                    [0.23, 0.25, 0.52],
                ),
                (
                    "p",
                    (
                        "Belief20-fade là xấp xỉ cục bộ cường độ nhu cầu, không phải belief state đầy đủ của POMDP. "
                        "Đây là snapshot tái lập trên bộ dữ liệu tham chiếu đã được dùng trong nghiên cứu trước; không phải một test set xác nhận hoàn toàn mới."
                    ),
                ),
            ],
        )
    )
    pages.append(
        (
            "Kết quả tổng hợp",
            [
                (
                    "p",
                    (
                        "Cost càng thấp càng tốt. Mỗi giá trị RL là trung bình trên các seed model và ngày kịch bản. "
                        "Cột +/- là độ lệch chuẩn cost trung bình giữa ba seed train; baseline tham chiếu là baseline có cost thấp nhất trong từng điều kiện."
                    ),
                ),
                table(
                    [
                        "Cấu hình",
                        "Điều kiện",
                        "Cost +/- seed SD",
                        "Baseline",
                        "Giảm cost",
                        "CI 95% (%)",
                    ],
                    overview,
                    [0.16, 0.21, 0.22, 0.13, 0.12, 0.16],
                ),
                savings_plot(comparisons),
                (
                    "p",
                    (
                        "Các khoảng tin cậy trên so sánh RL với baseline, không so sánh trực tiếp MLP10, GRU10 và Belief20-fade. "
                        "Khoảng được bootstrap theo ngày sau khi lấy trung bình seed; không bao gồm bất định do seed train."
                    ),
                ),
            ],
        )
    )
    for split, (title, condition) in SPLITS.items():
        rows = []
        for policy in POLICIES:
            row = (
                summary[
                    (summary.policy == policy)
                    & (summary.split == split)
                    & (summary.method == "ppo")
                ]
                .iloc[0]
                .to_dict()
            )
            rows.append({**row, "label": LABELS[policy]})
        for method, label in BASELINES.items():
            row = (
                summary[
                    (summary.policy == "mlp10")
                    & (summary.split == split)
                    & (summary.method == method)
                ]
                .iloc[0]
                .to_dict()
            )
            rows.append({**row, "label": label})
        results = [
            [
                r["label"],
                fmt(r["total_cost_core"]),
                fmt(r["mean_wait"], 2),
                fmt(r["p95_wait"], 2),
                fmt(100 * r["completed_share"], 2),
                fmt(100 * r["abandoned_share"], 2),
                fmt(100 * r["unfinished_share"], 2),
            ]
            for r in rows
        ]
        contrasts = []
        for policy in POLICIES:
            line = [LABELS[policy]]
            for baseline in BASELINES:
                r = comparisons[
                    (comparisons.policy == policy)
                    & (comparisons.split == split)
                    & (comparisons.baseline == baseline)
                ].iloc[0]
                line.append(
                    f"{r.cost_saved_pct:.1f}% [{r.cost_saved_pct_ci95_low:.1f}; {r.cost_saved_pct_ci95_high:.1f}]"
                )
            contrasts.append(line)
        note = (
            "Đánh đổi dịch vụ: cả ba cấu hình RL có P95 chờ cao hơn Proportional và Threshold, dù cost tổng thấp hơn. "
            "Threshold hoàn thành 98,59% khách và có 0,90% bỏ cuộc, tốt hơn ba cấu hình RL theo hai chỉ số này."
            if split == "test_ood_burst"
            else "Cả ba cấu hình RL giảm cost và P95 chờ so với các baseline; tỷ lệ hoàn thành là 100% trên điều kiện này."
        )
        pages.append(
            (
                title,
                [
                    (
                        "p",
                        f"{split}: {condition} Mỗi controller được chấm trên {spec['counts'][split]} ngày kịch bản giống nhau; RL có {len(seeds)} seed train.",
                    ),
                    table(
                        [
                            "Controller",
                            "Cost",
                            "Chờ TB (phút)",
                            "P95 chờ (phút)",
                            "Hoàn thành (%)",
                            "Bỏ cuộc (%)",
                            "Chưa xong (%)",
                        ],
                        results,
                        [0.19, 0.16, 0.13, 0.13, 0.13, 0.13, 0.13],
                    ),
                    condition_plot(rows),
                    ("h3", "Giảm cost so với từng baseline"),
                    table(
                        ["Cấu hình", "Fixed", "Proportional", "Threshold"],
                        contrasts,
                        [0.20, 0.26, 0.27, 0.27],
                    ),
                    ("p", "Ô trong bảng: % giảm cost [CI 95%]. Giá trị dương có lợi cho RL."),
                    ("p", note),
                ],
            )
        )
    training = []
    for policy in POLICIES:
        for seed in seeds:
            meta = metadata[policy, seed]
            training.append(
                [
                    LABELS[policy],
                    str(seed),
                    fmt(meta["initial_val_cost"]),
                    fmt(meta["best_val_cost"]),
                    f"{meta['best_timesteps']:,}",
                    f"{meta['total_timesteps_actual']:,}",
                ]
            )
    pages.append(
        (
            "Huấn luyện, cách tính và giới hạn",
            [
                table(
                    [
                        "Cấu hình",
                        "Seed",
                        "Val ban đầu",
                        "Val tốt nhất",
                        "Bước được chọn",
                        "Tổng bước",
                    ],
                    training,
                    [0.21, 0.08, 0.17, 0.17, 0.19, 0.18],
                ),
                ("h3", "Cách tính"),
                (
                    "p",
                    (
                        "Cost là mục tiêu tổng hợp theo trọng số, đơn vị tương đương passenger-minute, không phải chi phí tiền tệ. "
                        "Cost gồm thời gian chờ, thời gian trên xe, quá tải tiện nghi, vận hành/deadhead, từ chối lên xe lần đầu, "
                        "bỏ cuộc, thay đổi nhiệm vụ và khách chưa xong cuối ca."
                    ),
                ),
                (
                    "p",
                    (
                        "Đầu tiên lấy trung bình các seed model trên từng ngày kịch bản, sau đó lấy trung bình các ngày. "
                        "P95 chờ trong bảng là trung bình P95 của từng ngày, không phải percentile gộp toàn bộ hành khách. "
                        "Giảm cost = 100 x (baseline - RL) / baseline. CI 95% dùng 2.000 lần paired bootstrap theo ngày, seed 6001; "
                        "giữ ghép cặp theo seed và hash kịch bản. Không hiệu chỉnh nhiều phép so sánh và không đưa biến thiên seed train vào CI."
                    ),
                ),
                ("h3", "Kết luận và phạm vi"),
                (
                    "p",
                    (
                        "Trên snapshot này, ba cấu hình RL đều cải thiện cost so với baseline. GRU10 và Belief20-fade "
                        "chưa cho thấy cải thiện nhất quán về cost trung bình so với MLP10. Khi nhu cầu tăng đột biến, "
                        "cost thấp hơn không đồng nghĩa với mọi chỉ số dịch vụ tốt hơn."
                    ),
                ),
                (
                    "p",
                    (
                        "Kết quả chỉ áp dụng cho bộ mô phỏng tổng hợp trên mạng cố định ba tuyến. Không suy rộng thành "
                        "hiệu quả triển khai thực tế hay tính tối ưu toàn cục. Test dataset đã được dùng trước đây, và số seed train là ba. "
                        "Báo cáo này không thay thế các kết luận lịch sử đã chốt trong thư mục reports/ của dự án."
                    ),
                ),
                ("h3", "Nguồn và tái lập"),
                (
                    "p",
                    (
                        "Tất cả số trong báo cáo được đối chiếu lại với kết quả từng ngày. Dữ liệu gốc nằm trong evidence/: "
                        "configs/ và protocol.json lưu thiết lập; data/manifest.json lưu định danh kịch bản; <policy>/<seed>/train/ "
                        "lưu metadata và lịch sử; <policy>/<seed>/eval/ và baselines/shared/eval/ lưu kết quả. "
                        "summary.csv và comparisons.csv lưu bảng tổng hợp. artifacts.json kiểm tra inventory, kích thước và SHA256 toàn bộ bundle."
                    ),
                ),
                (
                    "p",
                    (
                        "Kiểm tra: uv run bus-rl artifact verify --path runs/standard-v1. "
                        "Weights, optimizer state và log chẩn đoán vẫn ở experiments/; bundle không chứa chúng."
                    ),
                ),
            ],
        )
    )
    return pages


def write_markdown(pages, path):
    lines = ["# Báo cáo thực nghiệm standard-v1", ""]
    for title, blocks in pages:
        lines += [f"## {title}", ""]
        for kind, value in blocks:
            if kind == "image":
                continue
            if kind == "table":
                headers, rows, _ = value
                clean = lambda x: str(x).replace("|", "\\|").replace("\n", " ")
                lines += [
                    "| " + " | ".join(map(clean, headers)) + " |",
                    "| " + " | ".join(["---"] * len(headers)) + " |",
                ]
                lines += ["| " + " | ".join(map(clean, row)) + " |" for row in rows]
            else:
                lines += [("### " if kind == "h3" else "") + value]
            lines += [""]
    path.write_text("\n".join(lines))


def write_html(pages, path):
    parts = [
        '<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Báo cáo standard-v1</title>',
        '<style>body{margin:0;background:#f2f5f7;color:#20303c;font:16px/1.6 system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:32px 24px}header{margin-bottom:28px}h1{font-size:36px;line-height:1.2}h2{font-size:26px;color:#087f8c}h3{font-size:19px}section{background:white;border:1px solid #dce4e8;padding:28px;margin:24px 0}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;vertical-align:top;padding:9px;border-bottom:1px solid #dde5e9}th{background:#edf4f5}.table{overflow-x:auto}img{width:100%;height:auto}nav a{margin-right:16px}footer{font-size:13px;color:#53636d}@media print{body{background:white}main{padding:0}section{border:0;break-before:page}nav{display:none}}</style><main><header><p>BUS FLEET CONTROL / STANDARD-V1</p><h1>Báo cáo thực nghiệm</h1><p>MLP10, GRU10 và Belief20-fade - cấu hình, điều kiện test và kết quả cuối.</p><nav><a href="report.md">Markdown</a><a href="report.pdf">PDF</a><a href="README.md">Hướng dẫn</a></nav></header>',
    ]
    for title, blocks in pages:
        parts.append(f"<section><h2>{escape(title)}</h2>")
        for kind, value in blocks:
            if kind == "image":
                parts.append(
                    f'<img alt="Biểu đồ so sánh kết quả" src="data:image/png;base64,{base64.b64encode(value).decode()}">'
                )
            elif kind == "table":
                headers, rows, _ = value
                parts.append(
                    '<div class="table"><table><thead><tr>'
                    + "".join(f"<th>{escape(str(x))}</th>" for x in headers)
                    + "</tr></thead><tbody>"
                )
                parts += [
                    "<tr>" + "".join(f"<td>{escape(str(x))}</td>" for x in row) + "</tr>"
                    for row in rows
                ]
                parts.append("</tbody></table></div>")
            else:
                tag = "h3" if kind == "h3" else "p"
                parts.append(f"<{tag}>{escape(value)}</{tag}>")
        parts.append("</section>")
    parts.append(
        "<footer>Nguồn: evidence/summary.csv, evidence/comparisons.csv và metadata của từng run. Dữ liệu mô phỏng tổng hợp; không phải kết quả triển khai thực tế.</footer></main></html>"
    )
    path.write_text("\n".join(parts))


def write_pdf(pages, path):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    font_dir = Path("/usr/share/fonts/truetype/dejavu")
    for face, filename in [("DejaVu", "DejaVuSans.ttf"), ("DejaVu-Bold", "DejaVuSans-Bold.ttf")]:
        pdfmetrics.registerFont(TTFont(face, str(font_dir / filename)))
    width = A4[0] - 34 * mm
    styles = {
        "p": ParagraphStyle("body", fontName="DejaVu", fontSize=9, leading=13, spaceAfter=8),
        "h2": ParagraphStyle(
            "heading",
            fontName="DejaVu-Bold",
            fontSize=19,
            leading=24,
            spaceAfter=14,
            textColor=colors.HexColor("#087f8c"),
        ),
        "h3": ParagraphStyle(
            "subheading",
            fontName="DejaVu-Bold",
            fontSize=11,
            leading=15,
            spaceBefore=6,
            spaceAfter=7,
            keepWithNext=True,
        ),
        "cell": ParagraphStyle(
            "cell", fontName="DejaVu", fontSize=7.4, leading=10, alignment=TA_LEFT
        ),
        "head": ParagraphStyle(
            "head",
            fontName="DejaVu-Bold",
            fontSize=7.4,
            leading=10,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#123a45"),
        ),
    }
    flow = []
    for index, (title, blocks) in enumerate(pages):
        if index:
            flow.append(PageBreak())
        flow.append(Paragraph(escape(title), styles["h2"]))
        for kind, value in blocks:
            if kind == "image":
                from PIL import Image as PILImage

                w, h = PILImage.open(io.BytesIO(value)).size
                flow += [Image(io.BytesIO(value), width=width, height=width * h / w), Spacer(1, 8)]
            elif kind == "table":
                headers, rows, fractions = value
                data = [[Paragraph(escape(str(x)), styles["head"]) for x in headers]]
                data += [[Paragraph(escape(str(x)), styles["cell"]) for x in row] for row in rows]
                item = Table(
                    data,
                    colWidths=[width * x / sum(fractions) for x in fractions],
                    repeatRows=1,
                    hAlign="LEFT",
                )
                item.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9f2f4")),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#d7e1e6")),
                            ("LEFTPADDING", (0, 0), (-1, -1), 5),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                            ("TOPPADDING", (0, 0), (-1, -1), 6),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                        ]
                    )
                )
                flow += [item, Spacer(1, 10)]
            else:
                flow.append(Paragraph(escape(value), styles[kind]))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d7e1e6"))
        canvas.line(17 * mm, 16 * mm, A4[0] - 17 * mm, 16 * mm)
        canvas.setFont("DejaVu", 7)
        canvas.setFillColor(colors.HexColor("#53636d"))
        canvas.drawString(17 * mm, 11 * mm, "BUS FLEET CONTROL / STANDARD-V1")
        canvas.drawRightString(A4[0] - 17 * mm, 11 * mm, f"{doc.page}")
        canvas.restoreState()

    SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=17 * mm,
        leftMargin=17 * mm,
        topMargin=18 * mm,
        bottomMargin=22 * mm,
        title="Báo cáo thực nghiệm standard-v1",
        author="Bus-RL study",
    ).build(flow, onFirstPage=footer, onLaterPages=footer)


def build_bundle(study):
    manifest = verify_artifacts(study)
    source = study / "evidence" if (study / "evidence").is_dir() else study
    with tempfile.TemporaryDirectory(prefix="bus-study-bundle-") as temporary:
        stage = Path(temporary) / study.name
        evidence = stage / "evidence"
        for name in manifest["files"]:
            relative = Path(name)
            if source != study:
                if relative.parts[0] != "evidence":
                    continue
                relative = Path(*relative.parts[1:])
            if relative.name == "README.md" or "report" in relative.parts:
                continue
            target = evidence / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, target)
        pages = document(evidence)
        write_markdown(pages, stage / "report.md")
        write_html(pages, stage / "report.html")
        write_pdf(pages, stage / "report.pdf")
        (stage / "README.md").write_text("""# standard-v1

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
""")
        manifest.update(
            {
                "layout": "study-report-v1",
                "updated_at": datetime.now(UTC).isoformat(),
                "files": {
                    p.relative_to(stage).as_posix(): {
                        "sha256": file_hash(p),
                        "bytes": p.stat().st_size,
                    }
                    for p in sorted(stage.rglob("*"))
                    if p.is_file()
                },
            }
        )
        manifest["total_bytes"] = sum(item["bytes"] for item in manifest["files"].values())
        manifest["inventory_sha256"] = canonical_hash(manifest["files"])
        (evidence / "artifacts.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        verify_artifacts(stage)
        backup = Path(tempfile.mkdtemp(prefix="bus-study-previous-")) / study.name
        shutil.move(str(study), backup)
        try:
            shutil.copytree(stage, study)
            verify_artifacts(study)
        except BaseException:
            shutil.rmtree(study, ignore_errors=True)
            shutil.move(str(backup), study)
            raise
        print(f"Built {study}: README.md, report.md, report.html, report.pdf, evidence/")
        print(f"Previous bundle: {backup}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("runs/standard-v1"))
    build_bundle(parser.parse_args().study)
