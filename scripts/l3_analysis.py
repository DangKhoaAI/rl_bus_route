#!/usr/bin/env python
"""Regenerate the L3 T9 held-out tables and plots from raw runs.

Raw inputs (gitignored):
  runs/rl-improvement/l3-t9/<arm>/<seed>/train-20260913-r2/   training runs
  runs/rl-improvement/l3-eval-fixed/<arm>/<seed>/<split>/     held-out evals
Curated outputs:
  reports/rl-improvement/tables/l3_*.csv|json
  reports/rl-improvement/plots/l3_*.png

Failure-trace decision rows are produced separately by
``scripts/l3_failure_traces.py`` (they require loading checkpoints).

Run:  uv run python scripts/l3_analysis.py
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from bus_rl.evaluation.statistics import paired_bootstrap

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "rl-improvement"
TABLES = ROOT / "reports" / "rl-improvement" / "tables"
PLOTS = ROOT / "reports" / "rl-improvement" / "plots"
CONFIGS = ROOT / "configs" / "experiments" / "rl-improvement"

ARMS = ["core", "no_reassign", "no_short", "fairness_zero"]
SEEDS = [11, 22, 33]
SPLITS = ["test_id", "test_ood_burst", "test_ood_traffic"]
SERVICE_COLS = ["unfinished_share", "abandoned_share", "p95_wait", "worst_route_mean_wait"]
TRAIN_RUN = "l3-t9"
EVAL_RUN = "l3-eval-fixed"


def training_dir(arm: str, seed: int) -> Path:
    return RUNS / TRAIN_RUN / arm / str(seed) / "train-20260913-r2"


def eval_dir(arm: str, seed: int, split: str) -> Path:
    return RUNS / EVAL_RUN / arm / str(seed) / split


def load_training_curves() -> pd.DataFrame:
    frames = []
    for arm in ARMS:
        for seed in SEEDS:
            frame = pd.read_csv(training_dir(arm, seed) / "evaluations.csv")
            frame["arm"], frame["seed"] = arm, seed
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_held_out() -> pd.DataFrame:
    frames = []
    for arm in ARMS:
        for seed in SEEDS:
            for split in SPLITS:
                frame = pd.read_csv(eval_dir(arm, seed, split) / "results.csv").copy()
                frame["arm"], frame["model_seed"], frame["split"] = arm, seed, split
                frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _hashes(arm: str, seed: int) -> tuple[str, str]:
    config_sha = sha256((CONFIGS / f"{arm}.toml").read_bytes()).hexdigest()
    checkpoint_sha = sha256((training_dir(arm, seed) / "best.zip").read_bytes()).hexdigest()
    return config_sha, checkpoint_sha


def build_tables(held: pd.DataFrame) -> dict[str, pd.DataFrame]:
    summary_rows, contrast_rows, paired_rows = [], [], []
    for split in SPLITS:
        core_sub = held[(held.arm == "core") & (held.split == split)]
        core_daily = core_sub.groupby("scenario_seed")["total_cost_core"].mean()
        core_seed = core_sub.groupby("model_seed")["total_cost_core"].mean()
        core_service = core_sub.groupby("model_seed")[SERVICE_COLS].mean()
        for arm in ARMS:
            sub = held[(held.arm == arm) & (held.split == split)]
            daily = sub.groupby("scenario_seed")["total_cost_core"].mean()
            arm_seed = sub.groupby("model_seed")["total_cost_core"].mean()
            for seed in SEEDS:
                s = sub[sub.model_seed == seed]
                config_sha, checkpoint_sha = _hashes(arm, seed)
                summary_rows.append(
                    {
                        "arm": arm,
                        "split": split,
                        "seed": seed,
                        "config_sha256": config_sha,
                        "checkpoint_sha256": checkpoint_sha,
                        "mean_cost": float(s.total_cost_core.mean()),
                        "std_cost_across_days": float(s.total_cost_core.std(ddof=1)),
                        "mean_wait": float(s.mean_wait.mean()),
                        "p95_wait": float(s.p95_wait.mean()),
                        "worst_route_wait": float(s.worst_route_mean_wait.mean()),
                        "unfinished_share": float(s.unfinished_share.mean()),
                        "abandoned_share": float(s.abandoned_share.mean()),
                        "completed_share": float(s.completed_share.mean()),
                        "censored_share": float(s.censored_share.mean()),
                        "unique_denied_share": float(s.unique_denied_share.mean()),
                        "active_bus_min": float(s.active_bus_min.mean()),
                        "deadhead_bus_min": float(s.deadhead_bus_min.mean()),
                    }
                )
            aligned = core_daily.index.intersection(daily.index)
            point, low, high = paired_bootstrap(
                daily.loc[aligned].to_numpy(), core_daily.loc[aligned].to_numpy()
            )
            ds = sub.groupby("model_seed")[SERVICE_COLS].mean() - core_service
            contrast_rows.append(
                {
                    "arm": arm,
                    "split": split,
                    "core_mean": float(core_daily.loc[aligned].mean()),
                    "arm_mean": float(daily.loc[aligned].mean()),
                    "mean_delta": point,
                    "ci_low": low,
                    "ci_high": high,
                    "pct_of_core": 100 * point / core_daily.loc[aligned].mean(),
                    "seeds_lower_than_core": int((arm_seed < core_seed).sum()),
                    "seed_std": float(arm_seed.std(ddof=1)),
                    "service_limits_satisfied": bool(
                        (ds["unfinished_share"] <= 0.005).all()
                        and (ds["unfinished_share"] >= -0.005).all()
                        and (ds["abandoned_share"] <= 0.002).all()
                        and (ds["p95_wait"] <= 1.0).all()
                        and (ds["worst_route_mean_wait"] <= 1.0).all()
                    ),
                    "delta_unfinished": float(ds["unfinished_share"].max()),
                    "delta_abandoned": float(ds["abandoned_share"].max()),
                    "delta_p95": float(ds["p95_wait"].max()),
                    "delta_worst": float(ds["worst_route_mean_wait"].max()),
                }
            )
            for day in aligned:
                paired_rows.append(
                    {
                        "arm": arm,
                        "split": split,
                        "scenario_seed": int(day),
                        "arm_cost": float(daily.loc[day]),
                        "core_cost": float(core_daily.loc[day]),
                        "delta": float(daily.loc[day] - core_daily.loc[day]),
                    }
                )
    baselines = []
    for split in SPLITS:
        frame = pd.read_csv(RUNS / EVAL_RUN / "baselines" / split / "results.csv")
        for method, group in frame.groupby("method"):
            baselines.append(
                {
                    "method": method,
                    "split": split,
                    "days": len(group),
                    "mean_cost": float(group.total_cost_core.mean()),
                    "mean_wait": float(group.mean_wait.mean()),
                    "p95_wait": float(group.p95_wait.mean()),
                    "unfinished_share": float(group.unfinished_share.mean()),
                    "completed_share": float(group.completed_share.mean()),
                }
            )
    return {
        "l3_held_out_summary": pd.DataFrame(summary_rows),
        "l3_contrasts": pd.DataFrame(contrast_rows),
        "l3_paired_days": pd.DataFrame(paired_rows),
        "l3_baselines_summary": pd.DataFrame(baselines),
    }


def build_verification(held: pd.DataFrame) -> dict:
    checks = []
    reference = {}
    for split in SPLITS:
        for arm in ARMS:
            for seed in SEEDS:
                frame = pd.read_csv(eval_dir(arm, seed, split) / "results.csv")
                meta = json.loads((training_dir(arm, seed) / "metadata.json").read_text())
                days = tuple(sorted(frame["scenario_seed"].tolist()))
                hashes = frame.sort_values("scenario_seed")["scenario_hash"].tolist()
                reference.setdefault(split, (days, hashes))
                checks.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "split": split,
                        "rows": len(frame),
                        "unique_days": int(frame.scenario_seed.nunique()),
                        "days_paired": days == reference[split][0],
                        "scenario_hashes_paired": hashes == reference[split][1],
                        "checkpoint_sha256": sha256(
                            (training_dir(arm, seed) / "best.zip").read_bytes()
                        ).hexdigest(),
                        "config_sha256": sha256((CONFIGS / f"{arm}.toml").read_bytes()).hexdigest(),
                        "reward_fairness": meta["reward"]["fairness"],
                        "native_hash": meta["native_build"]["library_sha256_runtime"],
                        "core_weight_recompute_differs": bool(
                            not np.allclose(frame["total_cost"], frame["total_cost_core"])
                        ),
                    }
                )
    return {
        "checks": checks,
        "all_rows_200": all(c["rows"] == 200 for c in checks),
        "all_unique_days": all(c["unique_days"] == 200 for c in checks),
        "all_days_paired": all(c["days_paired"] for c in checks),
        "all_scenario_hashes_paired": all(c["scenario_hashes_paired"] for c in checks),
        "single_native_hash": sorted({c["native_hash"] for c in checks}),
        "single_native_hash_across_matrix": len({c["native_hash"] for c in checks}) == 1,
        "fairness_zero_core_recompute_differs": all(
            c["core_weight_recompute_differs"] for c in checks if c["arm"] == "fairness_zero"
        ),
        "fairness_zero_reward_is_zero": all(
            c["reward_fairness"] == 0.0 for c in checks if c["arm"] == "fairness_zero"
        ),
    }


def make_plots(curves: pd.DataFrame, held: pd.DataFrame, tables: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOTS.mkdir(parents=True, exist_ok=True)
    palette = {
        "core": "#1b6ca8",
        "no_reassign": "#d1495b",
        "no_short": "#edae49",
        "fairness_zero": "#66a182",
    }

    # validation cost vs transitions and vs training wall clock
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for arm in ARMS:
        sub = curves[curves.arm == arm]
        by_step = sub.groupby("timesteps")["val_cost"].mean()
        axes[0].plot(by_step.index, by_step.values, label=arm, color=palette[arm])
        by_wall = sub.groupby("train_elapsed_wall_s")["val_cost"].mean().sort_index()
        axes[1].plot(by_wall.index, by_wall.values, label=arm, color=palette[arm])
    axes[0].set(
        xlabel="training transitions",
        ylabel="validation total_cost_core (100 days)",
        title="Validation cost vs transitions",
    )
    axes[1].set(
        xlabel="training wall clock (s)",
        ylabel="validation total_cost_core (100 days)",
        title="Validation cost vs training wall clock",
    )
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "l3_validation_curves.png", dpi=140)
    plt.close(fig)

    # held-out cost per split with per-seed points
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0), sharey=False)
    for ax, split in zip(axes, SPLITS):
        means, labels = [], []
        for arm in ARMS:
            points = [
                held[
                    (held.arm == arm) & (held.split == split) & (held.model_seed == seed)
                ].total_cost_core.mean()
                for seed in SEEDS
            ]
            ax.bar(len(means), np.mean(points), color=palette[arm], alpha=0.85)
            ax.scatter([len(means)] * len(points), points, color="black", s=14, zorder=3)
            means.append(np.mean(points))
            labels.append(arm)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_title(split)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("held-out total_cost_core")
    fig.suptitle("Frozen T9 matrix: held-out cost (bars = seed mean, dots = seeds)")
    fig.tight_layout()
    fig.savefig(PLOTS / "l3_held_out_costs.png", dpi=140)
    plt.close(fig)

    # service trade-off: cost delta vs p95-wait delta
    contrast = tables["l3_contrasts"]
    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {"test_id": "o", "test_ood_burst": "s", "test_ood_traffic": "^"}
    for arm in ARMS:
        for split in SPLITS:
            row = contrast[(contrast.arm == arm) & (contrast.split == split)].iloc[0]
            ax.scatter(
                row.pct_of_core,
                row.delta_p95,
                marker=markers[split],
                color=palette[arm],
                s=60,
                label=f"{arm}/{split}",
            )
    ax.axhline(1.0, color="red", ls="--", lw=1, label="registered +1.0 min P95 limit")
    ax.axvline(0.0, color="grey", ls=":", lw=1)
    ax.set(xlabel="mean cost delta vs core (%)", ylabel="max per-seed P95 wait delta (min)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(PLOTS / "l3_service_tradeoffs.png", dpi=140)
    plt.close(fig)


def assert_plot_inputs(held: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    """Guard the figure inputs against NaN / empty ranges before plotting."""
    assert held["total_cost_core"].notna().all() and (held["total_cost_core"] > 0).all()
    assert held["p95_wait"].notna().all() and (held["p95_wait"] >= 0).all()
    contrast = tables["l3_contrasts"]
    for column in ("mean_delta", "ci_low", "ci_high", "pct_of_core", "delta_p95", "delta_worst"):
        assert np.isfinite(contrast[column]).all(), column
    assert (contrast["ci_low"] <= contrast["mean_delta"]).all()
    assert (contrast["mean_delta"] <= contrast["ci_high"]).all()


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    curves = load_training_curves()
    held = load_held_out()
    tables = build_tables(held)
    assert_plot_inputs(held, tables)
    for name, frame in tables.items():
        frame.to_csv(TABLES / f"{name}.csv", index=False)
    verification = build_verification(held)
    (TABLES / "l3_verification.json").write_text(json.dumps(verification, indent=2))
    make_plots(curves, held, tables)
    for path in ("l3_validation_curves.png", "l3_held_out_costs.png", "l3_service_tradeoffs.png"):
        assert (PLOTS / path).stat().st_size > 10_000, path
    print(json.dumps({k: v for k, v in verification.items() if k != "checks"}, indent=2))
    print(tables["l3_contrasts"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
