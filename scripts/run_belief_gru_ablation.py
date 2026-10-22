"""Run a small fixed-budget 2x2 encoder ablation without changing simulator semantics."""

from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from bus_rl.config import load_run_config
from bus_rl.evaluation.runner import evaluate_scenarios
from bus_rl.evaluation.statistics import paired_bootstrap
from bus_rl.execution.runtime import apply_runtime_settings
from bus_rl.execution.scenarios.generation import generate_scenario
from bus_rl.execution.scenarios.io import load_manifest, load_split
from bus_rl.learning.training.checkpoint import load_model
from bus_rl.learning.training.train import train_run
from bus_rl.provenance import (
    assert_physical_compatible,
    file_hash,
    git_status,
    require_fresh_output,
)

ARMS = ("mlp", "belief", "gru", "belief_gru")
PROBES = {
    "probe_id": (61001, "base"),
    "probe_burst": (62001, "burst"),
    "probe_traffic": (63001, "traffic"),
}
METRICS = (
    "total_cost_core",
    "mean_wait",
    "p95_wait",
    "unfinished_share",
    "abandoned_share",
    "worst_route_mean_wait",
)


def summarize(output: Path, rows: list[pd.DataFrame], runs: list[dict], comparisons=None) -> None:
    frame = pd.concat(rows, ignore_index=True)
    frame.to_csv(output / "probe_metrics.csv", index=False)
    pd.DataFrame(runs).to_csv(output / "runs.csv", index=False)
    summaries = []
    contrasts = []
    for split, split_frame in frame.groupby("split"):
        for arm, arm_frame in split_frame.groupby("encoder"):
            item = {"split": split, "encoder": arm}
            for metric in METRICS:
                item[metric] = arm_frame[metric].mean()
            item["seed_std_cost"] = arm_frame.groupby("model_seed")["total_cost_core"].mean().std()
            summaries.append(item)
        # Factorial contrasts isolate each addition, both with and without the other.
        for candidate, control in comparisons or [
            ("belief", "mlp"),
            ("gru", "mlp"),
            ("belief_gru", "mlp"),
            ("belief_gru", "gru"),
            ("belief_gru", "belief"),
        ]:
            left = split_frame[split_frame.encoder == candidate]
            right = split_frame[split_frame.encoder == control]
            left_days = left.groupby("scenario_seed")["total_cost_core"].mean().sort_index()
            right_days = right.groupby("scenario_seed")["total_cost_core"].mean().sort_index()
            if not left_days.index.equals(right_days.index):
                raise ValueError("unaligned probe days")
            delta, low, high = paired_bootstrap(left_days.to_numpy(), right_days.to_numpy())
            left_seeds = left.groupby("model_seed")["total_cost_core"].mean()
            right_seeds = right.groupby("model_seed")["total_cost_core"].mean()
            item = {
                "split": split,
                "candidate": candidate,
                "control": control,
                "delta_cost": delta,
                "delta_cost_pct": delta / right_days.mean() * 100,
                "ci95_low": low,
                "ci95_high": high,
                "seeds_better": int((left_seeds < right_seeds).sum()),
            }
            for metric in METRICS[1:]:
                item[f"delta_{metric}"] = left[metric].mean() - right[metric].mean()
            contrasts.append(item)
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(contrasts).to_csv(output / "contrasts.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/belief-gru.toml"))
    parser.add_argument("--manifest", type=Path, default=Path("data/generated/base/manifest.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--probe-days", type=int, default=50)
    args = parser.parse_args()
    if not 1 <= args.probe_days <= 1000:
        parser.error("--probe-days must be between 1 and 1000 (probe seed ranges are disjoint)")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("model seeds must be unique")
    require_fresh_output(args.output)
    run = load_run_config(args.config)
    study = tomllib.loads(args.config.read_text()).get("study", {})
    specs = study.get("arms", [{"name": arm, "encoder": arm} for arm in ARMS])
    arm_configs = {
        spec["name"]: replace(
            run,
            algorithm=replace(
                run.algorithm, **{key: value for key, value in spec.items() if key != "name"}
            ),
        )
        for spec in specs
    }
    if len(arm_configs) != len(specs):
        parser.error("study arm names must be unique")
    for config in arm_configs.values():
        _ = config.arrival_memory_bins
    comparisons = study.get("comparisons")
    if comparisons and any(a not in arm_configs or b not in arm_configs for a, b in comparisons):
        parser.error("comparison references an unknown study arm")
    probes_spec = PROBES
    if "probe_seed_start" in study:
        start = int(study["probe_seed_start"])
        probes_spec = {
            name: (start + i * 1000, variant)
            for i, (name, (_, variant)) in enumerate(PROBES.items())
        }
    steps = run.algorithm.total_timesteps
    if steps % (run.algorithm.n_steps * run.algorithm.n_envs):
        parser.error("budget must be an exact number of PPO rollouts")
    apply_runtime_settings(run)
    manifest = load_manifest(args.manifest)
    assert_physical_compatible(manifest, run.physical)
    train = load_split(args.manifest, "train", with_tapes=False)
    validation = load_split(args.manifest, "validation", with_tapes=False)
    existing = {entry["seed"] for entries in manifest["splits"].values() for entry in entries}
    probes = {
        name: [generate_scenario(start + i, run.physical, variant) for i in range(args.probe_days)]
        for name, (start, variant) in probes_spec.items()
    }
    for scenarios in probes.values():
        if existing.intersection(scenario.seed for scenario in scenarios):
            raise ValueError("probe seeds overlap the existing data")
    args.output.mkdir(parents=True)
    sources = [
        *sorted(Path("src/bus_rl").rglob("*.py")),
        Path(__file__),
        args.config,
        Path("configs/base.toml"),
    ]
    sha, dirty = git_status()
    protocol = {
        "study": study.get("name", "belief-gru-small-ablation"),
        "status": "frozen-before-training",
        "git_sha": sha,
        "git_dirty": dirty,
        "sources": {str(p): file_hash(p) for p in sources},
        "manifest": str(args.manifest),
        "manifest_sha256": file_hash(args.manifest),
        "run_config": asdict(run),
        "arms": {name: asdict(cfg.algorithm) for name, cfg in arm_configs.items()},
        "comparisons": comparisons,
        "seeds": args.seeds,
        "train_days": len(train),
        "validation_days": len(validation),
        "checkpoint": "minimum validation cost; tie keeps earliest",
        "probes": {
            name: [{"seed": s.seed, "hash": s.scenario_hash} for s in scenarios]
            for name, scenarios in probes.items()
        },
        "selection": "No tuning after probe scores. Report all registered arms including failures.",
        "scope": "Windowed local-rate belief and windowed GRU; neither is a full-state POMDP solver.",
        "claim": "Exploratory equal-budget ablation at 40% of original core budget, not replacement evidence.",
        "ci": "paired day bootstrap after averaging model seeds; 2000 resamples, seed 6001; no seed CI",
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True))
    (args.output / "source_snapshot.json").write_text(
        json.dumps({str(path): path.read_text() for path in sources}, indent=2, sort_keys=True)
    )
    # Complete every training run before opening any new probe score.
    runs = []
    for seed in args.seeds:
        for arm, base_config in arm_configs.items():
            config = replace(base_config, algorithm=replace(base_config.algorithm, seed=seed))
            destination = args.output / arm / str(seed)
            print(f"[ablation] train encoder={arm} seed={seed}", flush=True)
            meta = train_run(config, train, validation, destination)
            if meta["total_timesteps_actual"] != steps:
                raise ValueError("training budget mismatch")
            runs.append(
                {
                    "encoder": arm,
                    "seed": seed,
                    "policy_encoder": config.algorithm.encoder,
                    "history_minutes": config.algorithm.history_minutes,
                    "timesteps": meta["total_timesteps_actual"],
                    "wall_time_s": meta["wall_time_s"],
                    "best_val_cost": meta["best_val_cost"],
                    "policy_parameters": meta["policy_parameters"],
                    "checkpoint_sha256": file_hash(destination / "best.zip"),
                }
            )
            pd.DataFrame(runs).to_csv(args.output / "runs.csv", index=False)
    rows = []
    for entry in runs:
        arm, seed = entry["encoder"], entry["seed"]
        base_config = arm_configs[arm]
        config = replace(base_config, algorithm=replace(base_config.algorithm, seed=seed))
        model, _ = load_model(
            args.output / arm / str(seed) / "best.zip", None, run.physical, "rust"
        )
        for split, scenarios in probes.items():
            print(f"[ablation] probe encoder={arm} seed={seed} split={split}", flush=True)
            frame, _ = evaluate_scenarios(
                scenarios, config, "ppo", model=model, model_seed=seed, split=split
            )
            if not np.isfinite(frame[list(METRICS)].to_numpy()).all():
                raise ValueError("nonfinite probe metrics")
            frame["encoder"] = arm
            frame["policy_encoder"] = config.algorithm.encoder
            frame["history_minutes"] = config.algorithm.history_minutes
            rows.append(frame)
    summarize(args.output, rows, runs, comparisons)
    print(f"[ablation] complete: {args.output}", flush=True)


if __name__ == "__main__":
    main()
