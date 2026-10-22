"""Human-readable report tables and a self-contained presentation document."""

from __future__ import annotations

import base64
import json
from html import escape
from pathlib import Path

import pandas as pd

from bus_rl.evaluation.report_data import METHOD_NAMES, SPLIT_NAMES


def _number(value, digits=1):
    return f"{value:,.{digits}f}" if pd.notna(value) else "Unavailable"


def human_tables(summary, comparisons, training):
    benchmark = pd.DataFrame(
        {
            "Condition": summary.split.map(lambda x: SPLIT_NAMES.get(x, x)),
            "Controller": summary.method.map(lambda x: METHOD_NAMES.get(x, x)),
            "Mean cost ↓": summary.total_cost_core.map(lambda x: _number(x)),
            "Days": summary.scenario_days,
            "Training seeds": summary.model_seeds.map(lambda x: str(x) if x else "—"),
            "Seed SD": summary.seed_std_cost.map(lambda x: _number(x) if pd.notna(x) else "—"),
        }
    )
    comparison_rows = []
    for _, row in comparisons.iterrows():
        low, high = row.get("cost_saved_pct_ci95_low"), row.get("cost_saved_pct_ci95_high")
        interval = (
            f"[{low:+.1f}%, {high:+.1f}%]" if pd.notna(low) and pd.notna(high) else "Unavailable"
        )
        comparison_rows.append(
            {
                "Condition": SPLIT_NAMES.get(row.split, row.split),
                "Baseline": METHOD_NAMES.get(row.baseline, row.baseline),
                "RL cost reduction": f"{row.cost_saved_pct:+.1f}%"
                if pd.notna(row.get("cost_saved_pct"))
                else "Unavailable",
                "95% interval": interval,
                "Paired days": int(row.paired_days)
                if pd.notna(row.get("paired_days"))
                else "Unavailable",
                "Evidence": {
                    "rl_lower_cost": "RL lower cost",
                    "rl_higher_cost": "RL higher cost",
                    "inconclusive": "Inconclusive",
                    "insufficient_days": "Insufficient days",
                    "missing_scenario_identity": "Pairing unavailable",
                }[row.status],
            }
        )
    train_rows = []
    for _, row in training.iterrows():
        train_rows.append(
            {
                "Model": f"{str(row.encoder).upper() if row.encoder != 'unknown' else 'Policy'} · seed {row.seed}",
                "Reference": "Untrained policy"
                if row.reference_kind == "untrained"
                else f"First logged ({row.reference_timestep:,.0f} steps)",
                "Reference cost": _number(row.reference_val_cost),
                "Best cost": _number(row.best_val_cost),
                "Reduction": f"{row.validation_reduction_pct:.1f}%"
                if pd.notna(row.validation_reduction_pct)
                else "Unavailable",
                "Selected step": f"{row.best_timesteps:,.0f}",
                "Total steps": f"{row.last_timesteps:,.0f}",
                "Validation days": _number(row.validation_days, 0),
            }
        )
    service = benchmark[["Condition", "Controller"]].copy()
    for metric, label, scale in [
        ("mean_wait", "Mean wait (min) ↓", 1),
        ("p95_wait", "P95 wait (min) ↓", 1),
        ("completed_share", "Completed (%) ↑", 100),
        ("abandoned_share", "Abandoned (%) ↓", 100),
        ("unfinished_share", "Unfinished (%) ↓", 100),
    ]:
        if metric in summary:
            service[label] = summary[metric].map(lambda x, scale=scale: _number(x * scale, 2))
    return {
        "benchmark": benchmark,
        "comparison": pd.DataFrame(comparison_rows),
        "training": pd.DataFrame(train_rows),
        "service": service,
    }


def report_findings(comparisons, training, frame):
    findings = []
    if comparisons.empty:
        findings.append(
            "Baseline comparison is unavailable. Supply RL and baseline evaluations on the same scenario days to quantify performance differences."
        )
    elif not comparisons.cost_saved_pct.notna().any():
        findings.append(
            "Relative reductions require matched scenario identities and a positive baseline mean cost. The supplied evaluations do not establish a relative reduction; absolute costs remain available."
        )
    else:
        valid = comparisons[comparisons.cost_saved_pct.notna()]
        for split, group in valid.groupby("split", sort=False):
            best = group.loc[group.baseline_cost.idxmin()]
            direction = "lower" if best.cost_saved_pct >= 0 else "higher"
            certainty = (
                "The paired cost interval excludes zero."
                if best.status in {"rl_lower_cost", "rl_higher_cost"}
                else "The available interval does not establish a reliable difference."
            )
            findings.append(
                f"{SPLIT_NAMES.get(split, split)}: RL cost is {abs(best.cost_saved_pct):.1f}% {direction} than the lowest-cost baseline ({METHOD_NAMES.get(best.baseline, best.baseline)}), across {int(best.paired_days)} matched days. {certainty}"
            )
        for split, group in valid.groupby("split", sort=False):
            if "rl_minus_baseline_p95_wait" in group:
                worse = group[group.rl_minus_baseline_p95_wait > 0]
                if not worse.empty:
                    names = ", ".join(METHOD_NAMES.get(x, x) for x in worse.baseline)
                    findings.append(
                        f"Service trade-off in {SPLIT_NAMES.get(split, split)}: RL has higher mean daily P95 wait than {names}; lower total cost does not imply improvement on every service metric."
                    )
    if not training.empty:
        zero = training[training.reference_kind == "untrained"]
        if not zero.empty:
            findings.append(
                f"Best-checkpoint validation reduction ranges from {zero.validation_reduction_pct.min():.1f}% to {zero.validation_reduction_pct.max():.1f}% relative to the measured untrained policy across {len(zero)} runs. Negative reductions indicate higher cost. Checkpoints were selected on validation, independently of evaluation scores."
            )
        if len(zero) < len(training):
            findings.append(
                "Historical training runs without a step-zero evaluation use the first recorded validation measurement as reference; no untrained performance is inferred."
            )
    if "validation" in set(frame.split):
        findings.append(
            "Validation results are used for model selection and do not constitute an independent held-out test."
        )
    return findings


def write_report_views(
    output: Path,
    frame,
    summary,
    comparisons,
    training,
    metadata_status: str,
    training_metadata: list[dict],
    *,
    figures: Path,
    payload: dict,
):
    tables = human_tables(summary, comparisons, training)
    findings = report_findings(comparisons, training, frame)
    methodology = (
        "Operational cost is a synthetic, core-weighted objective in passenger-minute equivalents. "
        "Controllers are compared within each operating condition. Model seeds are averaged per "
        "scenario day before averaging days. Cost reduction is 100 × (baseline − RL) / baseline. "
        "Intervals use 2,000 paired day-bootstrap resamples (seed 6001), recomputing the percentage "
        "ratio on each resample. They exclude training-seed uncertainty and are not adjusted for "
        "multiple comparisons. At least two matched days are required for an interval; small "
        "samples remain exploratory. Point charts use explicitly labelled data-scaled axes."
    )
    controller_description = (
        "RL uses a learned MaskablePPO controller. Fixed follows a fixed schedule; Proportional "
        "responds proportionally to observed demand; Threshold activates control at configured "
        "thresholds. Checkpoints are selected by the lowest post-training validation cost "
        "(earliest checkpoint on ties); test scores are not used for selection."
    )
    setup_rows = []
    seen = set()
    for meta in training_metadata:
        algo = meta.get("algorithm", {})
        identity = (meta.get("seed"), algo.get("encoder"), meta.get("total_timesteps_actual"))
        if identity in seen:
            continue
        seen.add(identity)
        physical = meta.get("physical_config", {})
        setup_rows.append(
            {
                "Seed": str(meta.get("seed", "Unavailable")),
                "Encoder": str(algo.get("encoder", "Not recorded")).upper(),
                "History (min)": str(algo.get("history_minutes", "Not recorded")),
                "Train steps": _number(meta.get("total_timesteps_actual"), 0),
                "Train days": _number(meta.get("train_days"), 0),
                "Validation days": _number(meta.get("validation_days_used"), 0),
                "Routes": _number(physical.get("route_count"), 0),
                "Buses": _number(physical.get("fleet_size"), 0),
                "Parameters": _number(meta.get("policy_parameters"), 0),
            }
        )
    setup = pd.DataFrame(
        setup_rows,
        columns=[
            "Seed",
            "Encoder",
            "History (min)",
            "Train steps",
            "Train days",
            "Validation days",
            "Routes",
            "Buses",
            "Parameters",
        ],
    )
    sections = [
        (
            "Operational Cost",
            "01_main_result.png",
            "Mean daily cost by controller and operating condition. Lower values indicate better performance on the operational objective.",
            tables["benchmark"],
        ),
        (
            "Cost Reduction",
            "02_baseline_savings.png",
            "Relative cost reduction against each baseline on identical scenario days. Positive values favor RL; intervals crossing zero are inconclusive.",
            tables["comparison"],
        ),
        (
            "Training Progress",
            "03_training_progress.png",
            "Recorded validation cost across training. Stars mark the lowest-cost checkpoints; historical runs retain their original measurement coverage.",
            tables["training"],
        ),
        (
            "Service Quality",
            "04_service_quality.png",
            "Waiting-time outcomes complement the cost objective. Completion, abandonment and unfinished shares expose potential service trade-offs.",
            tables["service"],
        ),
    ]
    parts = []
    for title, image, caption, table in sections:
        encoded = base64.b64encode((figures / image).read_bytes()).decode()
        parts.append(
            f'<section><h2>{escape(title)}</h2><p>{escape(caption)}</p><img alt="{escape(title)}" src="data:image/png;base64,{encoded}"><div class="table">{table.to_html(index=False, border=0, escape=True) if not table.empty else "<p>Required data were not supplied.</p>"}</div></section>'
        )
    wins = int((comparisons.status == "rl_lower_cost").sum()) if not comparisons.empty else 0
    cards = [
        ("Operating conditions", str(frame.split.nunique())),
        ("Controllers", str(frame.method.nunique())),
        ("Lower-cost RL contrasts", f"{wins} / {len(comparisons)}"),
    ]
    html = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RL Performance Report</title>
<style>body{margin:0;background:#f3f5f7;color:#1c2b38;font:16px/1.6 system-ui,sans-serif}main{max-width:1280px;margin:auto;padding:48px 32px}h1{font-size:44px;line-height:1.1;margin:8px 0 16px}h2{font-size:28px;margin:0 0 12px}p{color:#475766}.eyebrow{color:#087f8c;font-weight:700;letter-spacing:.12em;font-size:12px}.cards{display:flex;gap:20px;margin:28px 0}.card{flex:1;background:white;border-top:4px solid #087f8c;padding:20px}.card strong{display:block;font-size:32px;color:#087f8c}.card span{font-size:13px;color:#475766}section{background:white;padding:32px;margin:24px 0;border:1px solid #e1e6eb}img{width:100%;height:auto;display:block}.table{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}th{text-align:left;background:#edf4f5;color:#243b45}th,td{padding:12px;border-bottom:1px solid #e5eaee}li{margin:12px 0}details{margin:24px 0;padding:16px;background:white;border:1px solid #e1e6eb}summary{cursor:pointer;font-weight:600}details section{border:0;margin:0}footer{font-size:14px;color:#56616c}@media(max-width:700px){main{padding:24px 12px}.cards{flex-direction:column}section{padding:16px}h1{font-size:32px}}@media print{body{background:white}main{padding:0}section{break-inside:avoid;border:0;padding:16px 0}}</style><main><header><div class="eyebrow">BUS FLEET CONTROL · EVALUATION</div><h1>RL Performance Report</h1><p>Training evidence, baseline comparisons and service outcomes in a controlled simulator.</p></header>"""
    html += (
        '<div class="cards">'
        + "".join(
            f'<div class="card"><span>{escape(label)}</span><strong>{escape(value)}</strong></div>'
            for label, value in cards
        )
        + "</div>"
    )
    html += _experimental_setup(setup, summary, payload, controller_description, metadata_status)
    html += (
        "<section><h2>Final Results</h2><h3>Key Findings</h3><ul>"
        + "".join(f"<li>{escape(finding)}</li>" for finding in findings)
        + "</ul></section>"
    )
    # Keep the learning curve available without interrupting the final evaluation results.
    html += "".join(part for i, part in enumerate(parts) if i != 2)
    html += f"<details><summary>Training Progress</summary>{parts[2]}</details>"
    html += f"<section><h2>How to Read the Results</h2><p>{escape(methodology)}</p></section>"
    html += _reproducibility(payload)
    encoded = json.dumps(payload, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    html += f'<script id="report-data" type="application/json">{encoded}</script>'
    html += "<footer>Self-contained report. Original training and evaluation files remain at their source locations.</footer></main></html>"
    (output / "report.html").write_text(html)


def read_report_data(path: Path) -> dict:
    """Read the embedded report data used for study aggregation and resume."""
    marker = '<script id="report-data" type="application/json">'
    html = path.read_text()
    if marker not in html:
        raise ValueError(f"report has no embedded data: {path}; regenerate the report")
    return json.loads(html.split(marker, 1)[1].split("</script>", 1)[0])


# These names refer to the standard simulator splits, not inferred custom conditions.
_CONDITIONS = {
    "train": "Standard demand and traffic; used for learning.",
    "validation": "Standard demand and traffic; used for checkpoint selection, not an independent test.",
    "test_id": "Standard demand and traffic, with held-out scenario seeds.",
    "test_ood_burst": "Extra demand at one stop: 8–20 arrivals per simulation tick for 5–10 minutes, starting between minutes 30 and 150.",
    "test_ood_traffic": "One route corridor has travel times multiplied by 1.5–2.0 for 30 minutes.",
}


def _experimental_setup(setup, summary, payload, description, status):
    conditions = []
    for split, group in summary.groupby("split", sort=False):
        counts = "; ".join(
            f"{METHOD_NAMES.get(row.method, row.method)}: {int(row.scenario_days)}"
            for row in group.itertuples()
        )
        conditions.append(
            {
                "Condition": SPLIT_NAMES.get(split, split),
                "Split": split,
                "Standard simulator definition": _CONDITIONS.get(
                    split, "Custom condition; see source configuration."
                ),
                "Days per controller": counts,
            }
        )
    configurations = []
    seen = set()
    for item in payload["config"]["evaluations"]:
        if item is None:
            continue
        config = item["config"]
        # Seed-specific run identities are shown in setup; show each shared configuration once.
        for group in (
            "physical",
            "algorithm",
            "control",
            "reward",
            "forecast",
            "shaping",
            "runtime",
        ):
            for key, value in config.get(group, {}).items():
                if group == "algorithm" and key == "seed":
                    continue
                identity = (config.get("source"), group, key, json.dumps(value, sort_keys=True))
                if identity in seen:
                    continue
                seen.add(identity)
                configurations.append(
                    {
                        "Configuration": config.get("source", "Not recorded"),
                        "Group": group,
                        "Setting": key,
                        "Value": str(value),
                    }
                )
    table = lambda data: data.to_html(index=False, border=0, escape=True)
    setup_html = (
        table(setup) if not setup.empty else "<p>Training configuration was not recorded.</p>"
    )
    config_html = (
        table(pd.DataFrame(configurations))
        if configurations
        else "<p>Resolved configuration was not recorded.</p>"
    )
    return (
        "<section><h2>Run Configuration and Test Conditions</h2>"
        f'<p>{escape(description)}</p><div class="table">{setup_html}</div>'
        "<h3>Test Conditions</h3><p>Valid paired comparisons require identical scenario days within each condition. "
        "The definitions below describe the standard simulator splits; original source metadata is embedded for verification.</p>"
        f'<div class="table">{table(pd.DataFrame(conditions))}</div>'
        f'<details><summary>All recorded run settings</summary><div class="table">{config_html}</div></details>'
        f"<p>Evaluation metadata coverage: {escape(status)}.</p></section>"
    )


def _reproducibility(payload):
    sources = pd.DataFrame(
        [
            {"Input": key, "Original file": item["path"], "SHA256": item["sha256"]}
            for key, item in payload["verification"]["sources"].items()
        ]
    )
    return (
        "<details><summary>Source files and reproducibility</summary>"
        "<p>Inputs are kept in their original train/eval directories. This HTML embeds "
        "the resolved configuration, aggregate tables and source hashes; it does not copy raw inputs.</p>"
        f'<div class="table">{sources.to_html(index=False, border=0, escape=True)}</div></details>'
    )
