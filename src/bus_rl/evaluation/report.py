"""Generate a report bundle and preserve the evidence behind its tables."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from bus_rl.config import RunConfig
from bus_rl.evaluation.plots import render_plots
from bus_rl.provenance import file_hash, git_status, require_fresh_output

METRICS = {
    "total_cost_core": "Core-weighted synthetic operational cost; lower is better.",
    "mean_wait": "Mean observed passenger wait, minutes (includes censored waits).",
    "p95_wait": "Episode P95 passenger wait, minutes; summary is the mean of episode P95s.",
    "worst_route_mean_wait": "Worst route's mean wait in each episode, minutes.",
    "completed_share": "Completed passengers / generated passengers, fraction [0, 1].",
    "unfinished_share": "Waiting or onboard passengers / generated passengers, fraction [0, 1].",
    "abandoned_share": "Abandoned passengers / generated passengers, fraction [0, 1].",
    "unique_denied_share": "Passengers denied boarding at least once / generated passengers.",
}


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _source(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": file_hash(path)}


def write_evaluation_metadata(
    output: Path,
    frame: pd.DataFrame,
    run: RunConfig,
    manifest: Path,
    checkpoint: Path | None = None,
    checkpoint_metadata: dict | None = None,
) -> None:
    """Link resolved CLI config and checkpoint metadata to this exact results CSV."""
    sha, dirty = git_status()
    _write_json(
        output / "evaluation.json",
        {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "git_sha": sha,
            "git_dirty": dirty,
            "config": asdict(run),
            "manifest": _source(manifest),
            "checkpoint": _source(checkpoint) if checkpoint else None,
            "checkpoint_metadata": checkpoint_metadata,
            "results_sha256": file_hash(output / "results.csv"),
            "rows": len(frame),
            "splits": sorted(frame["split"].dropna().unique().tolist()),
            "methods": sorted(frame["method"].dropna().unique().tolist()),
        },
    )


def generate_report(results: Path, output: Path, history: Path | None = None) -> None:
    """Write tables/, figures/ and provenance/ from one evaluation CSV."""
    require_fresh_output(output)
    frame = pd.read_csv(results)
    required = {"method", "total_cost_core", "mean_wait"}
    if missing := required - set(frame.columns):
        raise ValueError(f"report results missing columns: {', '.join(sorted(missing))}")
    if frame.empty or frame["method"].isna().any():
        raise ValueError("report requires non-empty results with a method on every row")
    metrics = [name for name in METRICS if name in frame]
    for name in metrics:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        if np.isinf(frame[name].to_numpy(dtype=float)).any():
            raise ValueError(f"report metric {name} contains infinite values")
    if frame["total_cost_core"].isna().any():
        raise ValueError("report total_cost_core must be present on every row")

    sources = {"results": _source(results)}
    evaluation_path = results.parent / "evaluation.json"
    evaluation = None
    if evaluation_path.exists():
        evaluation = json.loads(evaluation_path.read_text())
        if evaluation.get("results_sha256") != sources["results"]["sha256"]:
            raise ValueError("evaluation.json does not match results.csv (SHA256 mismatch)")
        sources["evaluation"] = _source(evaluation_path)
    events = results.parent / "events.jsonl"
    traces = []
    if events.exists():
        sources["events"] = _source(events)
        traces = [json.loads(line) for line in events.read_text().splitlines() if line.strip()]
    learning = None
    if history is not None:
        sources["history"] = _source(history)
        learning = pd.read_csv(history)
        if not {"timesteps", "val_cost"}.issubset(learning.columns):
            raise ValueError("report history requires timesteps and val_cost columns")
        if "seed" not in learning:
            learning["seed"] = 0

    groups = [name for name in ("split", "method") if name in frame]
    grouped = frame.groupby(groups, dropna=False)
    summary = grouped[metrics].mean()
    summary.insert(0, "episodes", grouped.size())
    if "scenario_seed" in frame:
        summary.insert(1, "scenario_days", grouped["scenario_seed"].nunique())
    if "model_seed" in frame:
        summary.insert(
            2 if "scenario_seed" in frame else 1, "model_seeds", grouped["model_seed"].nunique()
        )

    tables, figures, provenance = (output / name for name in ("tables", "figures", "provenance"))
    for directory in (tables, figures, provenance):
        directory.mkdir(parents=True)
    shutil.copyfile(results, tables / "results.csv")
    summary.to_csv(tables / "summary.csv")
    render_plots(frame, figures, traces=traces, history=learning)
    if history is not None:
        shutil.copyfile(history, provenance / "history.csv")

    # Retain the original pooled JSON interface for consumers of the old report command.
    legacy_metrics = [
        name for name in ("total_cost_core", "mean_wait", "abandoned_share") if name in frame
    ]
    (output / "summary.json").write_text(
        frame.groupby("method", dropna=False)[legacy_metrics].mean().to_json(indent=2) + "\n"
    )
    (tables / "README.md").write_text(
        "# Tables\n\n"
        "- `results.csv`: exact copy of input; one row per evaluated episode/model.\n"
        "- `summary.csv`: arithmetic means grouped by " + ", ".join(groups) + ".\n\n"
        "Each episode row has equal weight; missing metric values are excluded per column.\n"
        "No confidence intervals or paired comparisons are inferred.\n\n"
        "`episodes` counts rows; `scenario_days` counts distinct scenario seeds when available;\n"
        "`model_seeds` counts non-null model seeds (0 for unseeded baselines).\n"
        "Counts do not establish that methods used identical scenario tapes.\n\n"
        "| Metric | Meaning |\n|---|---|\n"
        + "".join(f"| `{name}` | {METRICS[name]} |\n" for name in metrics)
        + "\nSee the original result columns for scenario hashes, seeds and cost components.\n"
    )
    _write_json(
        provenance / "config.json",
        {
            "status": "recorded" if evaluation else "unavailable",
            "evaluation_config": evaluation["config"] if evaluation else None,
            "checkpoint_metadata": evaluation.get("checkpoint_metadata") if evaluation else None,
        },
    )
    _write_json(
        provenance / "protocol.json",
        {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "evaluation": evaluation,
            "group_by": groups,
            "aggregation": "arithmetic mean over episode rows; skip missing values per metric",
            "metrics": {name: METRICS[name] for name in metrics},
            "confidence_intervals": "not computed",
            "figures": "pooled across input splits; trace plots use supplied events.jsonl",
            "history_available": history is not None,
            "trace_available": bool(traces),
        },
    )
    (provenance / "README.md").write_text(
        "# Provenance\n\n"
        "- `protocol.json`: aggregation rules and recorded evaluation context, when available.\n"
        "- `config.json`: resolved evaluation config and original checkpoint metadata.\n"
        "- `verification.json`: source paths/SHA256, row counts and report artifact hashes.\n"
        "- `history.csv`: exact learning-history input, if supplied via `--history`.\n\n"
        "The evaluation sidecar is checked against the results SHA256. Old CSVs without\n"
        "`evaluation.json` are supported, but their config/protocol is unknown.\n"
        "Source hashes identify inputs at report time; they do not certify policy quality\n"
        "or revalidate the original simulator, manifest tapes or checkpoint.\n"
    )
    (output / "README.md").write_text(
        "# Evaluation report\n\n"
        "- [Tables](tables/README.md): episode results and summary by split/method.\n"
        "- [Figures](figures/): costs, waiting CDF, learning curve and trace plots.\n"
        "- [Provenance](provenance/README.md): configuration, protocol and source hashes.\n\n"
        "`summary.json` retains the legacy means by method, pooled across splits.\n"
        "Use `tables/summary.csv` for split-specific results. Figure cost/CDF summaries\n"
        "are also pooled across splits. Missing history/trace inputs produce placeholder plots.\n\n"
        + (
            "Evaluation configuration was recorded by the CLI.\n"
            if evaluation
            else "Evaluation configuration is unavailable: no evaluation.json accompanied the CSV.\n"
        )
    )
    _write_json(
        provenance / "verification.json",
        {
            "schema_version": 1,
            "sources": sources,
            "rows": len(frame),
            "summary_rows": len(summary),
            "missing_metric_values": {name: int(frame[name].isna().sum()) for name in metrics},
            "evaluation_metadata_matches_results": True if evaluation else None,
            "artifacts": {
                str(path.relative_to(output)): file_hash(path)
                for path in sorted(output.rglob("*"))
                if path.is_file()
            },
        },
    )
