"""Generate an auditable presentation report from training and evaluation artifacts."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from bus_rl.config import RunConfig
from bus_rl.evaluation.report_data import (
    METRICS,
    compare_baselines,
    summarize_results,
    summarize_training,
    validate_results,
)
from bus_rl.evaluation.report_plots import render_presentation
from bus_rl.evaluation.report_views import write_report_views
from bus_rl.provenance import file_hash, git_status, require_fresh_output


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
    """Bind configuration, model identity and portable training history to evaluation."""
    sha, dirty = git_status()
    history = None
    if checkpoint is not None:
        original = checkpoint.parent / "evaluations.csv"
        if original.exists():
            recorded_hash = (checkpoint_metadata or {}).get("evaluations_sha256")
            if recorded_hash and recorded_hash != file_hash(original):
                raise ValueError("training history does not match checkpoint metadata")
            shutil.copyfile(original, output / "training_history.csv")
            history = {**_source(original), "artifact": "training_history.csv"}
    _write_json(
        output / "evaluation.json",
        {
            "schema_version": 2,
            "created_at": datetime.now(UTC).isoformat(),
            "git_sha": sha,
            "git_dirty": dirty,
            "config": asdict(run),
            "manifest": _source(manifest),
            "checkpoint": _source(checkpoint) if checkpoint else None,
            "checkpoint_metadata": checkpoint_metadata,
            "training_history": history,
            "results_sha256": file_hash(output / "results.csv"),
            "events_sha256": file_hash(output / "events.jsonl")
            if (output / "events.jsonl").exists()
            else None,
            "rows": len(frame),
            "splits": sorted(frame["split"].dropna().unique().tolist()),
            "methods": sorted(frame["method"].dropna().unique().tolist()),
        },
    )


def _paths(value) -> list[Path]:
    if value is None:
        return []
    return [Path(value)] if isinstance(value, (str, Path)) else [Path(path) for path in value]


def _history(path: Path, metadata: dict | None, index: int) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"seed": "string"})
    if frame.empty or not {"timesteps", "val_cost"}.issubset(frame):
        raise ValueError("report history requires non-empty timesteps and val_cost columns")
    for name in ("timesteps", "val_cost"):
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        if not np.isfinite(frame[name]).all() or (frame[name] < 0).any():
            raise ValueError(f"invalid training history {name}")
    metadata = metadata or {}
    expected = metadata.get("evaluations_sha256")
    if expected and file_hash(path) != expected:
        raise ValueError("training history SHA256 mismatch")
    algorithm = metadata.get("algorithm", {})
    if "seed" not in frame:
        frame["seed"] = str(metadata.get("seed", f"unknown-{index + 1}"))
    if "encoder" not in frame:
        frame["encoder"] = algorithm.get("encoder", "unknown")
    if "history_minutes" not in frame:
        frame["history_minutes"] = algorithm.get("history_minutes", np.nan)
    frame["run_id"] = (
        frame.encoder.astype("string").fillna("unknown")
        + "/"
        + frame.history_minutes.astype("string").fillna("unknown")
        + "/seed-"
        + frame.seed.astype("string").fillna("unknown")
    )
    if frame.duplicated(["run_id", "timesteps"]).any():
        raise ValueError("duplicate training timestep for the same run")
    return frame


def generate_report(results, output: Path, history=None, rl_method: str | None = None) -> dict:
    """Read one or more CSVs; pair scenarios before presenting baseline comparisons."""
    require_fresh_output(output)
    result_paths, history_paths = _paths(results), _paths(history)
    frames, evaluations, sources = [], [], {}
    inferred_histories = []
    for index, path in enumerate(result_paths):
        source_key = "results" if len(result_paths) == 1 else f"results_{index}"
        sources[source_key] = _source(path)
        frames.append(
            pd.read_csv(
                path,
                dtype={
                    "scenario_seed": "string",
                    "scenario_hash": "string",
                    "model_seed": "string",
                },
            )
        )
        evaluation_path = path.parent / "evaluation.json"
        evaluation = None
        if evaluation_path.exists():
            evaluation = json.loads(evaluation_path.read_text())
            if evaluation.get("results_sha256") != sources[source_key]["sha256"]:
                raise ValueError("evaluation.json does not match results.csv (SHA256 mismatch)")
            sources["evaluation" if len(result_paths) == 1 else f"evaluation_{index}"] = _source(
                evaluation_path
            )
            recorded = evaluation.get("training_history")
            if recorded:
                relative = Path(recorded["artifact"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("invalid training_history artifact path")
                candidate = path.parent / relative
                if file_hash(candidate) != recorded["sha256"]:
                    raise ValueError("evaluation training history SHA256 mismatch")
                inferred_histories.append((candidate, evaluation.get("checkpoint_metadata")))
        evaluations.append(evaluation)
        events = path.parent / "events.jsonl"
        if events.exists():
            digest = file_hash(events)
            if (
                evaluation
                and evaluation.get("events_sha256")
                and digest != evaluation["events_sha256"]
            ):
                raise ValueError("events.jsonl SHA256 mismatch")
            sources[f"events_{index}"] = _source(events)

    frame = validate_results(pd.concat(frames, ignore_index=True))
    if rl_method is None:
        rl_method = next(
            (
                name
                for name in (
                    "ppo",
                    "maskable_ppo",
                    "a2c",
                    "reinforce",
                    "ppo_free",
                    "a2c_free",
                    "reinforce_free",
                    "core",
                )
                if name in set(frame.method)
            ),
            "ppo",
        )
    elif rl_method not in set(frame.method):
        raise ValueError(f"RL method {rl_method!r} not found in results")
    summary = summarize_results(frame)
    comparisons = compare_baselines(frame, rl_method)
    histories = []
    training_metadata = []
    seen_histories = set()
    candidates = (
        [
            (
                path,
                json.loads((path.parent / "metadata.json").read_text())
                if (path.parent / "metadata.json").exists()
                else None,
            )
            for path in history_paths
        ]
        if history_paths
        else inferred_histories
    )
    for index, (path, metadata) in enumerate(candidates):
        learning = _history(path, metadata, index)
        identity = (file_hash(path), tuple(sorted(learning.run_id.unique())))
        if identity in seen_histories:
            continue
        seen_histories.add(identity)
        sources[f"history_{len(histories)}"] = _source(path)
        histories.append(learning)
        if metadata:
            training_metadata.append(metadata)
    learning = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    if not learning.empty and learning.duplicated(["run_id", "timesteps"]).any():
        raise ValueError("conflicting training histories for the same encoder/model seed")
    training = summarize_training(learning)
    coverage = sum(item is not None for item in evaluations)
    status = (
        "recorded" if coverage == len(evaluations) else "partial" if coverage else "unavailable"
    )

    metrics = [name for name in METRICS if name in frame]
    single = evaluations[0] if len(evaluations) == 1 else None
    payload = {
        "schema_version": 3,
        "config": {
            "status": status,
            "evaluation_config": single["config"] if single else None,
            "checkpoint_metadata": single.get("checkpoint_metadata") if single else None,
            "training_runs": training_metadata,
            "evaluations": [
                {"config": item["config"], "checkpoint_metadata": item.get("checkpoint_metadata")}
                if item
                else None
                for item in evaluations
            ],
        },
        "protocol": {
            "created_at": datetime.now(UTC).isoformat(),
            "evaluation": single,
            "evaluations": evaluations,
            "rl_method": rl_method,
            "aggregation": "average model seeds per scenario day, then average days",
            "metrics": {name: METRICS[name] for name in metrics},
            "confidence_intervals": {
                "method": "paired day bootstrap; seeds averaged within day",
                "resamples": 2000,
                "seed": 6001,
                "level": 0.95,
                "training_seed_uncertainty_included": False,
                "multiple_comparison_adjustment": False,
            },
            "history_available": bool(histories),
        },
        "verification": {
            "sources": sources,
            "rows": len(frame),
            "summary_rows": len(summary),
            "missing_metric_values": {name: int(frame[name].isna().sum()) for name in metrics},
            "evaluation_metadata_matches_results": True if coverage == len(evaluations) else None,
            "metadata_coverage": {"recorded": coverage, "inputs": len(evaluations)},
        },
        "tables": {
            name: json.loads(table.to_json(orient="records", double_precision=15))
            for name, table in (
                ("summary", summary),
                ("comparisons", comparisons),
                ("training", training),
            )
        },
    }
    # Figures are intermediate files only. The public report is a single portable HTML.
    with TemporaryDirectory(prefix="bus-rl-report-") as temporary:
        figures = Path(temporary)
        render_presentation(summary, comparisons, learning, training, figures, rl_method)
        output.mkdir(parents=True)
        write_report_views(
            output,
            frame,
            summary,
            comparisons,
            training,
            status,
            training_metadata,
            figures=figures,
            payload=payload,
        )
    return payload
