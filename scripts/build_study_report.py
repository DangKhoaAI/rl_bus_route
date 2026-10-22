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
import math
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
LABELS = {"mlp10": "RL-MLP10", "gru10": "RL-GRU10", "belief20-fade": "RL-Belief20-fade"}
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
BASELINES = {
    "fixed": "RULE-Fixed",
    "proportional": "RULE-Proportional",
    "threshold": "RULE-Threshold",
}
COLORS = {"RL-MLP10": "#087f8c", "RL-GRU10": "#4567a8", "RL-Belief20-fade": "#ad6843"}


def fmt(value, digits=1):
    return f"{value:,.{digits}f}"


def table(headers, rows, widths, *, best=(), separators=()):
    return ("table", (headers, rows, widths, set(best), set(separators)))


def best_cells(rows, directions, groups=None):
    """Rank unrounded metrics within a comparable group; mark every numerical tie."""
    groups = groups if groups is not None else [None] * len(rows)
    selected = set()
    for group in dict.fromkeys(groups):
        for column, direction in directions.items():
            candidates = [
                (index, row[column])
                for index, row in enumerate(rows)
                if groups[index] == group and row[column] is not None and math.isfinite(row[column])
            ]
            if not candidates:
                continue
            optimum = (min if direction == "min" else max)(value for _, value in candidates)
            selected.update(
                (index, column)
                for index, value in candidates
                if math.isclose(value, optimum, rel_tol=1e-12, abs_tol=1e-12)
            )
    return selected


def image(plot):
    buffer = io.BytesIO()
    plot.savefig(buffer, format="png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(plot)
    return ("image", buffer.getvalue())


def savings_plot(comparisons, selected):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    tick_labels = []
    for i, split in enumerate(SPLITS):
        policy = selected[split]
        for j, baseline in enumerate(BASELINES):
            row = comparisons[
                (comparisons.policy == policy)
                & (comparisons.split == split)
                & (comparisons.baseline == baseline)
            ].iloc[0]
            y = i * 4 + j
            color = COLORS[LABELS[policy]]
            ax.plot(
                [row.cost_saved_pct_ci95_low, row.cost_saved_pct_ci95_high],
                [y, y],
                color=color,
                lw=2,
            )
            ax.scatter(row.cost_saved_pct, y, color=color, s=40, zorder=3)
            tick_labels.append((y, f"{split.replace('test_', '')} / {BASELINES[baseline]}"))
    ax.set_yticks([x[0] for x in tick_labels], [x[1] for x in tick_labels], fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="#555555", ls="--", lw=1)
    ax.set_xlabel(
        "Selected RL vs each RULE baseline: cost reduction (%) - paired 95% interval", fontsize=10
    )
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
    selected, rule_costs, rule_savings, rule_best = {}, [], [], set()
    overview, overview_metrics, overview_groups = [], [], []
    for split in SPLITS:
        policies = summary[(summary.split == split) & (summary.method == "ppo")]
        winner = policies.loc[policies.total_cost_core.idxmin()]
        selected[split] = winner.policy
        baseline_rows = [
            summary[
                (summary.split == split)
                & (summary.policy == "mlp10")
                & (summary.method == baseline)
            ].iloc[0]
            for baseline in BASELINES
        ]
        numeric_costs = [winner.total_cost_core, *[row.total_cost_core for row in baseline_rows]]
        row_index = len(rule_costs)
        rule_best.update(
            (row_index, column + 2)
            for column, cost in enumerate(numeric_costs)
            if math.isclose(cost, min(numeric_costs), rel_tol=1e-12, abs_tol=1e-12)
        )
        rule_costs.append(
            [SPLITS[split][0], LABELS[winner.policy], *[fmt(cost) for cost in numeric_costs]]
        )
        reductions = []
        for baseline in BASELINES:
            contrast = comparisons[
                (comparisons.policy == winner.policy)
                & (comparisons.split == split)
                & (comparisons.baseline == baseline)
            ].iloc[0]
            reductions.append(
                f"{contrast.cost_saved_pct:.1f}% [{contrast.cost_saved_pct_ci95_low:.1f}; {contrast.cost_saved_pct_ci95_high:.1f}]"
            )
        rule_savings.append([SPLITS[split][0], *reductions])
        for policy in POLICIES:
            row = policies[policies.policy == policy].iloc[0]
            candidates = comparisons[(comparisons.policy == policy) & (comparisons.split == split)]
            reference = candidates.loc[candidates.baseline_cost.idxmin()]
            overview.append(
                [
                    SPLITS[split][0],
                    LABELS[policy],
                    f"{fmt(row.total_cost_core)} +/- {fmt(row.seed_std_cost)}",
                    f"{reference.cost_saved_pct:+.2f}% ({BASELINES[reference.baseline]})",
                ]
            )
            overview_metrics.append([None, None, row.total_cost_core, reference.cost_saved_pct])
            overview_groups.append(split)
    pages = []
    pages.append(
        (
            "I. EXPERIMENT SETUP",
            [
                (
                    "p",
                    (
                        "Báo cáo tổng hợp standard-v1: so sánh ba cấu hình bộ mã hóa quan sát của MaskablePPO "
                        "với ba bộ điều khiển RULE-Fixed, RULE-Proportional và RULE-Threshold trong cùng bộ mô phỏng xe buýt."
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
                        ["RL-MLP10", "MLP / 10 phút", "Mạng feed-forward"],
                        ["RL-GRU10", "GRU / 10 phút", "Mã hóa cửa sổ quan sát bằng GRU"],
                        [
                            "RL-Belief20-fade",
                            "Belief / 20 phút",
                            "Đặc trưng rate Gamma-Poisson; prior shape/rate = 0.1/0.1; decay = 0.8",
                        ],
                    ],
                    [0.23, 0.25, 0.52],
                ),
                (
                    "p",
                    (
                        "RL-Belief20-fade là xấp xỉ cục bộ cường độ nhu cầu, không phải belief state đầy đủ của POMDP. "
                        "Đây là snapshot tái lập trên bộ dữ liệu tham chiếu đã được dùng trong nghiên cứu trước; không phải một test set xác nhận hoàn toàn mới."
                    ),
                ),
            ],
        )
    )
    pages.append(
        (
            "1. Summary",
            [
                ("h3", "1.1. Best RL so với các RULE baseline"),
                (
                    "p",
                    "Chọn RL có mean cost thấp nhất trong từng điều kiện test: RL-MLP10 ở điều kiện bình thường và gián đoạn giao thông; RL-Belief20-fade ở điều kiện tăng nhu cầu. Đây là lựa chọn mô tả trên kết quả đã đánh giá, không phải một cấu hình duy nhất thắng mọi điều kiện.",
                ),
                table(
                    [
                        "Điều kiện",
                        "Best RL",
                        "Cost RL",
                        *BASELINES.values(),
                    ],
                    rule_costs,
                    [0.19, 0.20, 0.13, 0.14, 0.18, 0.16],
                    best=rule_best,
                ),
                ("h3", "Mức giảm cost của Best RL (%) [CI 95%]"),
                table(
                    ["Điều kiện", *BASELINES.values()],
                    rule_savings,
                    [0.25, 0.25, 0.25, 0.25],
                ),
                (
                    "legend",
                    "Ô xanh, chữ đậm: cost thấp nhất trong cùng điều kiện test. Mức giảm cost dương có lợi cho RL; CI được ghép cặp theo ngày, không bao gồm bất định do seed train.",
                ),
                savings_plot(comparisons, selected),
                ("pagebreak", None),
                ("h3", "1.2. So sánh các cấu hình RL với nhau"),
                (
                    "p",
                    "Cột +/- là độ lệch chuẩn cost trung bình giữa ba seed train. Mỗi RL được so với RULE có mean cost thấp nhất trong cùng điều kiện: RULE-Fixed ở điều kiện bình thường và gián đoạn giao thông; RULE-Threshold khi nhu cầu tăng đột biến. % giảm cost = 100 x (cost RULE - cost RL) / cost RULE; số dương nghĩa là RL có cost thấp hơn RULE.",
                ),
                table(
                    [
                        "Điều kiện",
                        "Cấu hình RL",
                        "Cost +/- seed SD",
                        "Giảm cost so với RULE (%)",
                    ],
                    overview,
                    [0.24, 0.22, 0.26, 0.28],
                    best=best_cells(overview_metrics, {2: "min", 3: "max"}, overview_groups),
                    separators=[
                        i
                        for i in range(1, len(overview_groups))
                        if overview_groups[i] != overview_groups[i - 1]
                    ],
                ),
                (
                    "legend",
                    "Ô xanh: cost thấp nhất và % giảm cost so với RULE cao nhất giữa các RL trong cùng điều kiện. Đồng hạng được xét bằng số liệu chưa làm tròn.",
                ),
                (
                    "p",
                    "So sánh giữa các RL ở đây chỉ mang tính mô tả. Báo cáo không ước lượng khoảng tin cậy cho chênh lệch giữa hai cấu hình RL và không khẳng định ưu thế thống kê giữa chúng.",
                ),
            ],
        )
    )
    for case_index, (split, (title, condition)) in enumerate(SPLITS.items(), 1):
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
        contrast_metrics = []
        for policy in POLICIES:
            line = [LABELS[policy]]
            values = [None]
            for baseline in BASELINES:
                r = comparisons[
                    (comparisons.policy == policy)
                    & (comparisons.split == split)
                    & (comparisons.baseline == baseline)
                ].iloc[0]
                line.append(
                    f"{r.cost_saved_pct:.1f}% [{r.cost_saved_pct_ci95_low:.1f}; {r.cost_saved_pct_ci95_high:.1f}]"
                )
                values.append(r.cost_saved_pct)
            contrasts.append(line)
            contrast_metrics.append(values)
        note = (
            "Đánh đổi dịch vụ: cả ba cấu hình RL có P95 chờ cao hơn RULE-Proportional và RULE-Threshold, dù cost tổng thấp hơn. "
            "RULE-Threshold hoàn thành 98,59% khách và có 0,90% bỏ cuộc, tốt hơn ba cấu hình RL theo hai chỉ số này."
            if split == "test_ood_burst"
            else "Cả ba cấu hình RL giảm cost và P95 chờ so với các baseline; tỷ lệ hoàn thành là 100% trên điều kiện này."
        )
        pages.append(
            (
                f"{case_index + 1}. DETAIL CASE {case_index}: {title}",
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
                        best=best_cells(
                            [
                                [
                                    None,
                                    r["total_cost_core"],
                                    r["mean_wait"],
                                    r["p95_wait"],
                                    r["completed_share"],
                                    r["abandoned_share"],
                                    r["unfinished_share"],
                                ]
                                for r in rows
                            ],
                            {1: "min", 2: "min", 3: "min", 4: "max", 5: "min", 6: "min"},
                        ),
                    ),
                    condition_plot(rows),
                    ("h3", "Giảm cost so với từng baseline"),
                    table(
                        ["Cấu hình", "RULE-Fixed", "RULE-Proportional", "RULE-Threshold"],
                        contrasts,
                        [0.20, 0.26, 0.27, 0.27],
                        best=best_cells(contrast_metrics, {1: "max", 2: "max", 3: "max"}),
                    ),
                    ("p", "Ô trong bảng: % giảm cost [CI 95%]. Giá trị dương có lợi cho RL."),
                    ("p", note),
                ],
            )
        )
    training = []
    training_metrics = []
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
            training_metrics.append([None, None, None, meta["best_val_cost"]])
    pages.append(
        (
            "APPENDIX B. Huấn luyện, cách tính",
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
                    best=best_cells(training_metrics, {3: "min"}),
                ),
                (
                    "legend",
                    "Ô xanh trong cột Val tốt nhất là checkpoint có cost thấp nhất trên cùng tập validation; số bước và cost ban đầu không được xếp hạng.",
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
    detailed_setup = pages[0][1]
    pages[0] = (
        "I. EXPERIMENT SETUP",
        [
            table(
                ["Hạng mục", "Thông tin chính"],
                [
                    [
                        "Bài toán",
                        "Điều khiển đội xe buýt bằng MaskablePPO, so với các luật điều khiển",
                    ],
                    [
                        "Mô phỏng",
                        f"{physical['route_count']} tuyến, {physical['stops_per_route']} điểm dừng/tuyến; {physical['fleet_size']} xe; {physical['horizon_s'] // 60} phút/ngày",
                    ],
                    ["Controllers", ", ".join([*LABELS.values(), *BASELINES.values()])],
                    [
                        "Dữ liệu",
                        f"{spec['counts']['train']} ngày train, {spec['counts']['validation']} ngày validation; {spec['counts']['test_id']} ngày/điều kiện test",
                    ],
                    [
                        "Huấn luyện",
                        f"{algo['total_timesteps']:,} transitions/seed; seeds {', '.join(map(str, seeds))}",
                    ],
                    ["Điều kiện test", "; ".join(title for title, _ in SPLITS.values())],
                    [
                        "Đánh giá",
                        "Cost thấp hơn là tốt hơn; kèm thời gian chờ, tỷ lệ hoàn thành và bỏ cuộc. Chọn checkpoint theo cost validation thấp nhất.",
                    ],
                ],
                [0.23, 0.77],
            ),
        ],
    )
    training_title, training_blocks = pages.pop()
    source_index = training_blocks.index(("h3", "Nguồn và tái lập"))
    pages.extend(
        [
            (
                "APPENDIX A. Chi tiết thực nghiệm",
                detailed_setup
                + [
                    ("h3", "Điều kiện test chi tiết"),
                    table(
                        ["Split", "Điều kiện"],
                        [[split, condition] for split, (_, condition) in SPLITS.items()],
                        [0.23, 0.77],
                    ),
                ],
            ),
            (training_title, training_blocks[:source_index]),
            ("APPENDIX C. Nguồn", training_blocks[source_index + 1 :]),
        ]
    )
    additions = algorithm_pages(evidence)
    if additions:
        position = next(i for i, (title, _) in enumerate(pages) if title.startswith("APPENDIX"))
        pages[position:position] = additions
    return pages


def load_algorithm_extension(evidence):
    """Verify the full matrix from per-day results, rather than trusting summary cells."""
    extension = evidence / "extensions/masking"
    protocol_path = extension / "protocol.json"
    if not protocol_path.exists():
        return None
    protocol = json.loads(protocol_path.read_text())
    spec = protocol["plan"]["spec"]
    expected_arms = {
        f"{algo}_{mode}" for algo in ("reinforce", "a2c", "ppo") for mode in ("apply", "free")
    }
    if set(spec["policies"]) != expected_arms or spec["seeds"] != [11, 22, 33]:
        raise ValueError("incomplete algorithm/masking matrix")
    if spec["counts"] != {"train": 500, "validation": 100, **{s: 200 for s in SPLITS}}:
        raise ValueError("algorithm extension must use the standard-v1 dataset counts")
    reference = json.loads((evidence / "protocol.json").read_text())["resolved_configs"]["gru10"]
    raw, metadata = {}, {}
    common_algorithm = (
        "encoder",
        "history_minutes",
        "gamma",
        "learning_rate",
        "n_steps",
        "n_envs",
        "batch_size",
        "n_epochs",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
        "gae_lambda",
        "clip_range",
        "total_timesteps",
        "eval_freq",
        "device",
        "torch_threads",
    )
    for algo in ("reinforce", "a2c", "ppo"):
        for mode in ("apply", "free"):
            arm = f"{algo}_{mode}"
            config = protocol["resolved_configs"][arm]
            for field in common_algorithm:
                if config["algorithm"][field] != reference["algorithm"][field]:
                    raise ValueError(f"standard setup mismatch: {arm}/{field}")
            for field in ("physical", "control", "reward", "runtime", "forecast", "shaping"):
                if config[field] != reference[field]:
                    raise ValueError(f"standard setup mismatch: {arm}/{field}")
            for seed in spec["seeds"]:
                base = evidence / algo / mode / str(seed)
                meta = json.loads((base / "train/metadata.json").read_text())
                history = pd.read_csv(base / "train/evaluations.csv")
                if (
                    meta["algorithm"]["name"] != algo
                    or meta["mask_mode"] != mode
                    or meta["total_timesteps_actual"] != 245760
                    or meta["episodes_completed"] != 2048
                    or meta["action_execution"]["invalid_action"] != "noop"
                    or meta["action_execution"]["invalid_penalty"] != 0
                    or meta["evaluations_sha256"] != file_hash(base / "train/evaluations.csv")
                ):
                    raise ValueError(f"invalid training evidence: {arm}/{seed}")
                eligible = history[history.checkpoint_eligible]
                if not math.isclose(float(eligible.val_cost.min()), meta["best_val_cost"]):
                    raise ValueError("best checkpoint does not match eligible validation history")
                if history.iloc[-1].optimizer_steps != meta["optimizer_steps"]:
                    raise ValueError("final post-update validation is missing")
                metadata[algo, mode, seed] = meta
                for split in SPLITS:
                    path = base / "eval" / split / "results.csv"
                    sidecar = json.loads(path.with_name("evaluation.json").read_text())
                    frame = pd.read_csv(
                        path,
                        dtype={
                            "scenario_seed": "string",
                            "model_seed": "string",
                            "scenario_hash": "string",
                        },
                    )
                    method = algo if mode == "apply" else f"{algo}_free"
                    if (
                        sidecar["results_sha256"] != file_hash(path)
                        or len(frame) != 200
                        or set(frame.method) != {method}
                        or set(frame.model_seed) != {str(seed)}
                    ):
                        raise ValueError(f"invalid evaluation evidence: {path}")
                    if frame.decision_steps.ne(120).any():
                        raise ValueError("mask mode changed episode duration")
                    if mode == "apply" and frame.invalid_request_count.ne(0).any():
                        raise ValueError("apply selected invalid actions")
                    raw.setdefault((algo, mode), []).append(frame)
    raw = {
        key: validate_results(pd.concat(frames, ignore_index=True)) for key, frames in raw.items()
    }
    # Force exact day pairing with the existing shared baseline tapes, for every seed.
    baselines = pd.concat(
        [
            pd.read_csv(
                evidence / "baselines/shared/eval" / split / "results.csv",
                dtype={
                    "scenario_seed": "string",
                    "model_seed": "string",
                    "scenario_hash": "string",
                },
            )
            for split in SPLITS
        ],
        ignore_index=True,
    )
    for (algo, mode), frame in raw.items():
        compare_baselines(
            validate_results(pd.concat([frame, baselines], ignore_index=True)),
            algo if mode == "apply" else f"{algo}_free",
        )
    for algo in ("reinforce", "a2c", "ppo"):
        for seed in spec["seeds"]:
            if (
                metadata[algo, "apply", seed]["action_execution_hash"]
                != metadata[algo, "free", seed]["action_execution_hash"]
            ):
                raise ValueError("mask A/B execution contracts differ")
    return raw, metadata, protocol


def algorithm_pages(evidence):
    loaded = load_algorithm_extension(evidence)
    if loaded is None:
        return []
    raw, metadata, _protocol = loaded
    pages = [
        (
            "III. ALGORITHM COMPARISON & MASK A/B",
            [
                (
                    "p",
                    (
                        "Bổ sung REINFORCE, A2C và PPO với cùng encoder GRU10, trên đúng dữ liệu và "
                        "ngân sách của standard-v1. Mỗi thuật toán có hai nhánh được train riêng: apply và free."
                    ),
                ),
                table(
                    ["Thiết lập", "Giá trị"],
                    [
                        [
                            "Dữ liệu",
                            "500 ngày train, 100 validation, 200 ngày cho mỗi split test; cùng tapes",
                        ],
                        ["Số lần train", "3 thuật toán × 2 mask mode × 3 seeds (11/22/33) = 18"],
                        [
                            "Ngân sách",
                            "245.760 transitions/seed, 2.048 episode; CPU, 4 env, Torch 2 threads",
                        ],
                        [
                            "Setup kế thừa",
                            (
                                "GRU10; gamma 1; lr 3e-4; rollout yêu cầu 256; batch 256; "
                                "PPO 4 epoch; cùng reward và guards"
                            ),
                        ],
                        [
                            "REINFORCE",
                            (
                                "Return Monte Carlo, không baseline/critic; thu 240 bước/env để đủ "
                                "2 episode; một cập nhật toàn rollout"
                            ),
                        ],
                        ["A2C", "GAE 0,95, Adam; một cập nhật toàn rollout 256 bước/env"],
                        ["PPO", "PPO-Clip 0,2; GAE 0,95; 4 epoch, 16 optimizer steps/rollout"],
                        [
                            "Validation",
                            (
                                "Mốc yêu cầu 12.288 transitions; đo sau update ở rollout boundary "
                                "đầu tiên đạt mốc; đo cả trọng số sau update cuối"
                            ),
                        ],
                        [
                            "Mask-free",
                            (
                                "Policy chọn trong 221 action; request sai thực thi NOOP, vẫn chạy "
                                "2 phút, nhận cost thực; không penalty bổ sung"
                            ),
                        ],
                        [
                            "Execution chung",
                            "Cả apply/free dùng cùng noop adapter; native kernel giữ guard strict",
                        ],
                    ],
                    [0.24, 0.76],
                ),
                (
                    "p",
                    (
                        "Ba policy MLP10/GRU10/Belief20-fade trong các phần trước là snapshot gốc. "
                        "Phần bổ sung train lại PPO control và dùng validation sau update; so thuật toán "
                        "và hiệu quả masking bằng chính các control mới trong phần này."
                    ),
                ),
            ],
        )
    ]
    mask_contrasts = {}
    for algo in ("reinforce", "a2c", "ppo"):
        mask_contrasts[algo] = compare_baselines(
            pd.concat([raw[algo, "apply"], raw[algo, "free"]], ignore_index=True), algo
        )
    for split, (label, _) in SPLITS.items():
        rows, numeric, plotting = [], [], []
        for algo in ("reinforce", "a2c", "ppo"):
            for mode in ("apply", "free"):
                frame = raw[algo, mode]
                group = frame[frame.split == split]
                stats = summarize_results(group).iloc[0]
                rows.append(
                    [
                        algo.upper(),
                        mode,
                        f"{fmt(stats.total_cost_core)} +/- {fmt(stats.seed_std_cost)}",
                        fmt(stats.mean_wait),
                        fmt(stats.p95_wait),
                        f"{100 * stats.completed_share:.2f}%",
                        f"{100 * group.invalid_request_rate.mean():.2f}%",
                    ]
                )
                numeric.append(
                    [
                        None,
                        None,
                        stats.total_cost_core,
                        stats.mean_wait,
                        stats.p95_wait,
                        stats.completed_share,
                        None,
                    ]
                )
                plotting.append(
                    {
                        "label": f"{algo.upper()} / {mode}",
                        "total_cost_core": stats.total_cost_core,
                        "p95_wait": stats.p95_wait,
                    }
                )
        contrasts = []
        for algo in ("reinforce", "a2c", "ppo"):
            contrast = mask_contrasts[algo]
            row = contrast[contrast.split == split].iloc[0]
            contrasts.append(
                [
                    algo.upper(),
                    f"{row.cost_saved_pct:+.2f}%",
                    f"[{row.cost_saved_pct_ci95_low:+.2f}; {row.cost_saved_pct_ci95_high:+.2f}]",
                ]
            )
        pages.append(
            (
                f"MASK A/B - {label}",
                [
                    table(
                        [
                            "Thuật toán",
                            "Mode",
                            "Cost +/- seed std",
                            "Mean wait",
                            "P95 wait",
                            "Hoàn thành",
                            "Request sai",
                        ],
                        rows,
                        [0.15, 0.08, 0.24, 0.12, 0.12, 0.15, 0.14],
                        best=best_cells(numeric, {2: "min", 3: "min", 4: "min", 5: "max"}),
                    ),
                    (
                        "p",
                        (
                            "Wait tính bằng phút. Request sai là tỷ lệ quyết định được chuyển thành NOOP. "
                            "Mean cost lấy trung bình model seeds trong từng ngày rồi trung bình các ngày."
                        ),
                    ),
                    condition_plot(plotting),
                    ("h3", "Lợi ích của mask trong từng thuật toán"),
                    table(
                        ["Thuật toán", "Giảm cost apply so với free", "CI 95% paired theo ngày"],
                        contrasts,
                        [0.22, 0.38, 0.40],
                    ),
                    (
                        "p",
                        (
                            "Giá trị dương nghĩa là apply giảm cost so với free. CI dùng 2.000 bootstrap "
                            "ghép cặp theo ngày, seed 6001; không bao gồm bất định giữa các seed train."
                        ),
                    ),
                ],
            )
        )
    algorithm_rows = []
    for algo in ("reinforce", "a2c"):
        paired = compare_baselines(
            pd.concat([raw[algo, "apply"], raw["ppo", "apply"]], ignore_index=True), "ppo"
        )
        for split, (label, _) in SPLITS.items():
            row = paired[paired.split == split].iloc[0]
            algorithm_rows.append(
                [
                    label,
                    algo.upper(),
                    f"{row.cost_saved_pct:+.2f}%",
                    f"[{row.cost_saved_pct_ci95_low:+.2f}; {row.cost_saved_pct_ci95_high:+.2f}]",
                ]
            )
    training = []
    for (algo, mode, seed), meta in metadata.items():
        training.append(
            [
                algo.upper(),
                mode,
                str(seed),
                fmt(meta["best_val_cost"]),
                str(meta["effective_n_steps"]),
                f"{meta['optimizer_steps']:,}",
                fmt(meta["wall_time_s"]),
            ]
        )
    pages.append(
        (
            "ALGORITHM COMPARISON - cùng mask apply",
            [
                table(
                    ["Điều kiện", "Đối chứng", "PPO giảm cost", "CI 95% paired theo ngày"],
                    algorithm_rows,
                    [0.29, 0.18, 0.21, 0.32],
                ),
                (
                    "p",
                    (
                        "PPO được so trực tiếp với REINFORCE/A2C có mask trên cùng ngày test. "
                        "Dấu dương có lợi cho PPO; mọi arm nhận cùng số transitions, nhưng số lượt cập "
                        "nhật và độ dài rollout thực tế khác theo định nghĩa thuật toán."
                    ),
                ),
                ("h3", "Huấn luyện từng arm và seed"),
                table(
                    [
                        "Thuật toán",
                        "Mode",
                        "Seed",
                        "Best val cost",
                        "Steps/env",
                        "Optimizer steps",
                        "Train + val (s)",
                    ],
                    training,
                    [0.16, 0.08, 0.07, 0.20, 0.13, 0.18, 0.18],
                ),
                (
                    "p",
                    (
                        "Weights/logs nằm ở experiments/standard-v1-masking. Evidence tại "
                        "evidence/<algorithm>/<mode>/<seed>/{train,eval}. Protocol và config của phần "
                        "bổ sung nằm ở evidence/extensions/masking. Các số được tính lại từ results.csv "
                        "từng ngày và đối chiếu checksum trước khi dựng report."
                    ),
                ),
            ],
        )
    )
    return pages


def write_markdown(pages, path):
    lines = ["# Báo cáo thực nghiệm standard-v1", ""]
    for index, (title, blocks) in enumerate(pages):
        is_result = 1 <= index <= len(SPLITS) + 1
        if index == 1:
            lines += ["## II. RESULT", ""]
        lines += [f"{'###' if is_result else '##'} {title}", ""]
        for kind, value in blocks:
            if kind in {"image", "legend", "pagebreak"}:
                continue
            if kind == "table":
                headers, rows, _, _best, _separators = value
                clean = lambda x: str(x).replace("|", "\\|").replace("\n", " ")
                lines += [
                    "| " + " | ".join(map(clean, headers)) + " |",
                    "| " + " | ".join(["---"] * len(headers)) + " |",
                ]
                lines += ["| " + " | ".join(map(clean, row)) + " |" for row in rows]
            else:
                prefix = ("#### " if is_result else "### ") if kind == "h3" else ""
                lines += [prefix + value]
            lines += [""]
    path.write_text("\n".join(lines))


def write_html(pages, path):
    parts = [
        '<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Báo cáo standard-v1</title>',
        '<style>body{margin:0;background:#f2f5f7;color:#20303c;font:16px/1.6 system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:32px 24px}header{margin-bottom:28px}h1{font-size:36px;line-height:1.2}h2{font-size:26px;color:#087f8c}h3{font-size:19px}h3.numbered{font-size:24px;color:#087f8c}h4{font-size:19px}section{background:white;border:1px solid #dce4e8;padding:28px;margin:24px 0}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;vertical-align:top;padding:9px;border-bottom:1px solid #dde5e9}th{background:#edf4f5}tr.condition-start td{border-top:3px solid #748895}td.best{background:#e8f5e9;color:#166534;font-weight:700}.legend{font-size:13px;color:#42594c}.table{overflow-x:auto}img{width:100%;height:auto}nav a{margin-right:16px}footer{font-size:13px;color:#53636d}@media print{.pagebreak{break-before:page}body{background:white}main{padding:0}section{border:0;break-before:page}nav{display:none}}</style><main><header><p>BUS FLEET CONTROL / STANDARD-V1</p><h1>Báo cáo thực nghiệm</h1><p>RL-MLP10, RL-GRU10 và RL-Belief20-fade - cấu hình, điều kiện test và kết quả cuối.</p><nav><a href="report.md">Markdown</a><a href="report.pdf">PDF</a><a href="README.md">Hướng dẫn</a></nav></header>',
    ]
    if any(title.startswith("III. ALGORITHM") for title, _ in pages):
        parts[1] = parts[1].replace(
            "RL-MLP10, RL-GRU10 và RL-Belief20-fade - cấu hình, điều kiện test và kết quả cuối.",
            "Study gốc MLP10/GRU10/Belief20-fade và mở rộng REINFORCE/A2C/PPO với A/B masking.",
        )
    for index, (title, blocks) in enumerate(pages):
        is_result = 1 <= index <= len(SPLITS) + 1
        parts.append("<section>")
        if index == 1:
            parts.append("<h2>II. RESULT</h2>")
        heading = "h3" if is_result else "h2"
        attribute = ' class="numbered"' if is_result else ""
        parts.append(f"<{heading}{attribute}>{escape(title)}</{heading}>")
        for kind, value in blocks:
            if kind == "image":
                parts.append(
                    f'<img alt="Biểu đồ so sánh kết quả" src="data:image/png;base64,{base64.b64encode(value).decode()}">'
                )
            elif kind == "pagebreak":
                parts.append('<div class="pagebreak"></div>')
            elif kind == "table":
                headers, rows, _, best, separators = value
                parts.append(
                    '<div class="table"><table><thead><tr>'
                    + "".join(f"<th>{escape(str(x))}</th>" for x in headers)
                    + "</tr></thead><tbody>"
                )
                parts += [
                    ('<tr class="condition-start">' if row_index in separators else "<tr>")
                    + "".join(
                        f'<td class="best"><strong>{escape(str(x))}</strong></td>'
                        if (row_index, column) in best
                        else f"<td>{escape(str(x))}</td>"
                        for column, x in enumerate(row)
                    )
                    + "</tr>"
                    for row_index, row in enumerate(rows)
                ]
                parts.append("</tbody></table></div>")
            else:
                tag = ("h4" if is_result else "h3") if kind == "h3" else "p"
                attribute = ' class="legend"' if kind == "legend" else ""
                parts.append(f"<{tag}{attribute}>{escape(value)}</{tag}>")
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
    styles["best"] = ParagraphStyle(
        "best", parent=styles["cell"], fontName="DejaVu-Bold", textColor=colors.HexColor("#166534")
    )
    styles["result"] = ParagraphStyle("result", parent=styles["h2"], fontSize=16, leading=21)
    flow = []
    for index, (title, blocks) in enumerate(pages):
        if index:
            flow.append(Spacer(1, 12) if title.startswith("APPENDIX C.") else PageBreak())
        if index == 1:
            flow.append(Paragraph("II. RESULT", styles["h2"]))
        style = "result" if 1 <= index <= len(SPLITS) + 1 else "h2"
        flow.append(Paragraph(escape(title), styles[style]))
        for kind, value in blocks:
            if kind == "image":
                from PIL import Image as PILImage

                w, h = PILImage.open(io.BytesIO(value)).size
                flow += [Image(io.BytesIO(value), width=width, height=width * h / w), Spacer(1, 8)]
            elif kind == "pagebreak":
                flow.append(PageBreak())
            elif kind == "table":
                headers, rows, fractions, best, separators = value
                data = [[Paragraph(escape(str(x)), styles["head"]) for x in headers]]
                data += [
                    [
                        Paragraph(
                            escape(str(x)),
                            styles["best" if (row_index, column) in best else "cell"],
                        )
                        for column, x in enumerate(row)
                    ]
                    for row_index, row in enumerate(rows)
                ]
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
                item.setStyle(
                    TableStyle(
                        [
                            (
                                "BACKGROUND",
                                (column, row + 1),
                                (column, row + 1),
                                colors.HexColor("#e8f5e9"),
                            )
                            for row, column in best
                        ]
                    )
                )
                item.setStyle(
                    TableStyle(
                        [
                            (
                                "LINEABOVE",
                                (0, row + 1),
                                (-1, row + 1),
                                1.7,
                                colors.HexColor("#748895"),
                            )
                            for row in separators
                        ]
                    )
                )
                flow += [item, Spacer(1, 10)]
            else:
                flow.append(Paragraph(escape(value), styles["p" if kind == "legend" else kind]))

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

Một báo cáo tổng hợp cho RL-MLP10, RL-GRU10 và RL-Belief20-fade.

- [Báo cáo Markdown](report.md): đọc ngay trong editor.
- [Báo cáo HTML](report.html): bảng và biểu đồ, tự chứa hình ảnh.
- [Báo cáo PDF](report.pdf): bản để đọc, in hoặc chia sẻ.

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
        if (evidence / "extensions/masking/protocol.json").exists():
            readme = stage / "README.md"
            text = readme.read_text().replace(
                "Một báo cáo tổng hợp cho RL-MLP10, RL-GRU10 và RL-Belief20-fade.",
                "Study gốc MLP10/GRU10/Belief20-fade, bổ sung REINFORCE/A2C/PPO với A/B masking.",
            )
            text += "\nPhần mở rộng: evidence/<algorithm>/<apply|free>/<seed>/{train,eval}. "
            text += "Protocol/config/screening được giữ tại evidence/extensions/masking/. "
            text += "Evidence gốc giữ nguyên; các file report gồm cả kết quả mới.\n"
            readme.write_text(text)
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
