"""Auditable report statistics, with scenario pairing before any comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd

METRICS = {
    "total_cost_core": "Core-weighted operational cost, passenger-minute equivalents; lower is better.",
    "mean_wait": "Mean observed passenger wait (minutes), including end-of-episode censored waits.",
    "p95_wait": "Mean daily P95 wait (minutes), rather than the pooled passenger-level percentile.",
    "worst_route_mean_wait": "Mean wait of the worst-performing route in each episode (minutes).",
    "completed_share": "Completed passengers / generated passengers, fraction [0, 1].",
    "unfinished_share": "Passengers waiting or onboard at episode end / generated passengers.",
    "abandoned_share": "Abandoned passengers / generated passengers, fraction [0, 1].",
    "unique_denied_share": "Passengers denied boarding at least once / generated passengers.",
}
METHOD_NAMES = {
    "ppo": "RL (PPO)",
    "maskable_ppo": "RL (PPO)",
    "core": "RL (PPO core)",
    "a2c": "RL (A2C + mask)",
    "reinforce": "RL (REINFORCE + mask)",
    "ppo_free": "RL (PPO, mask-free)",
    "a2c_free": "RL (A2C, mask-free)",
    "reinforce_free": "RL (REINFORCE, mask-free)",
    "fixed": "Fixed",
    "proportional": "Proportional",
    "threshold": "Threshold",
    "random": "Random",
}
SPLIT_NAMES = {
    "test_id": "Normal Demand",
    "test_ood_burst": "Demand Surge",
    "test_ood_traffic": "Traffic Disruption",
    "validation": "Validation",
    "probe_id": "Normal Demand (Probe)",
    "probe_burst": "Demand Surge (Probe)",
    "probe_traffic": "Traffic Disruption (Probe)",
    "unspecified": "Unspecified",
}
COMPARISON_COLUMNS = [
    "split",
    "rl_method",
    "baseline",
    "status",
    "paired_days",
    "rl_model_seeds",
    "rl_cost",
    "baseline_cost",
    "cost_saved",
    "cost_saved_ci95_low",
    "cost_saved_ci95_high",
    "cost_saved_pct",
    "cost_saved_pct_ci95_low",
    "cost_saved_pct_ci95_high",
    "rl_minus_baseline_mean_wait",
    "rl_minus_baseline_p95_wait",
    "rl_minus_baseline_abandoned_share",
    "rl_minus_baseline_unfinished_share",
]
TRAINING_COLUMNS = [
    "run_id",
    "seed",
    "encoder",
    "history_minutes",
    "reference_timestep",
    "reference_kind",
    "reference_val_cost",
    "best_val_cost",
    "best_timesteps",
    "last_timesteps",
    "last_val_cost",
    "validation_reduction_pct",
    "validation_days",
    "wall_time_s",
]


def validate_results(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if missing := {"method", "total_cost_core", "mean_wait"} - set(frame):
        raise ValueError(f"report results missing columns: {', '.join(sorted(missing))}")
    if frame.empty or frame["method"].isna().any():
        raise ValueError("report requires non-empty results with a method on every row")
    for name in METRICS.keys() & set(frame):
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        if np.isinf(frame[name].to_numpy(dtype=float)).any():
            raise ValueError(f"report metric {name} contains infinite values")
    if frame["total_cost_core"].isna().any() or (frame["total_cost_core"] < 0).any():
        raise ValueError("report total_cost_core must be finite and non-negative on every row")
    if "split" not in frame:
        frame["split"] = "unspecified"
    frame["split"] = frame["split"].fillna("unspecified")
    if "model_seed" not in frame:
        frame["model_seed"] = pd.NA
    if {"scenario_seed", "scenario_hash"}.issubset(frame):
        known = frame.dropna(subset=["scenario_seed", "scenario_hash"])
        if (known.groupby(["split", "scenario_seed"])["scenario_hash"].nunique() > 1).any():
            raise ValueError("scenario hash mismatch: same split/seed has different tapes")
        keys = ["split", "method", "model_seed", "scenario_hash"]
        if known.duplicated(keys).any():
            raise ValueError("duplicate evaluation rows: same method/model seed/scenario")
    return frame


def summarize_results(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [name for name in METRICS if name in frame]
    rows = []
    for (split, method), group in frame.groupby(["split", "method"], dropna=False, sort=True):
        # Average model seeds on each scenario day before averaging days.
        day_key = (
            "scenario_hash"
            if "scenario_hash" in group and group.scenario_hash.notna().all()
            else "scenario_seed"
        )
        daily = (
            group.groupby(day_key)[metrics].mean()
            if day_key in group and group[day_key].notna().all()
            else group[metrics]
        )
        seed_costs = group.groupby("model_seed")["total_cost_core"].mean()
        rows.append(
            {
                "split": split,
                "method": method,
                "episodes": len(group),
                "scenario_days": len(daily),
                "model_seeds": group.model_seed.nunique(),
                "seed_std_cost": float(seed_costs.std()) if len(seed_costs) > 1 else np.nan,
                **daily.mean().to_dict(),
            }
        )
    return pd.DataFrame(rows)


def compare_baselines(frame: pd.DataFrame, rl_method: str) -> pd.DataFrame:
    """Use identical scenario sets and average model seeds before bootstrapping days."""
    rows = []
    if rl_method not in set(frame.method):
        return pd.DataFrame(columns=COMPARISON_COLUMNS)
    for split, group in frame.groupby("split", sort=True):
        rl = group[group.method == rl_method]
        if rl.empty:
            continue
        for baseline, other in group[group.method != rl_method].groupby("method", sort=True):
            pair = pd.concat([rl, other])
            entry = {"split": split, "rl_method": rl_method, "baseline": baseline}
            if (
                not {"scenario_hash", "scenario_seed"}.issubset(pair)
                or pair[["scenario_hash", "scenario_seed"]].isna().any().any()
            ):
                rows.append({**entry, "status": "missing_scenario_identity"})
                continue
            days = set(rl.scenario_hash)
            for _, model in pair.groupby(["method", "model_seed"], dropna=False):
                if set(model.scenario_hash) != days:
                    raise ValueError(
                        f"unpaired scenarios in {split}: {rl_method} vs {baseline}; evaluate all models on the same days"
                    )
            metrics = [name for name in METRICS if name in pair]
            left = rl.groupby("scenario_hash")[metrics].mean().sort_index()
            right = other.groupby("scenario_hash")[metrics].mean().reindex(left.index)
            r, b = left.total_cost_core.to_numpy(), right.total_cost_core.to_numpy()
            delta = b - r
            mean_baseline = float(b.mean())
            pct = float(100 * delta.mean() / mean_baseline) if mean_baseline > 0 else np.nan
            low = high = low_pct = high_pct = np.nan
            status = "insufficient_days"
            if len(r) >= 2:
                draw = np.random.default_rng(6001).integers(0, len(r), (2000, len(r)))
                sampled_delta = delta[draw].mean(axis=1)
                low, high = np.quantile(sampled_delta, [0.025, 0.975])
                if mean_baseline > 0:
                    denominator = b[draw].mean(axis=1)
                    valid = denominator > 0
                    if valid.all():
                        low_pct, high_pct = np.quantile(
                            100 * sampled_delta / denominator, [0.025, 0.975]
                        )
                status = (
                    "rl_lower_cost" if low > 0 else "rl_higher_cost" if high < 0 else "inconclusive"
                )
            row = {
                **entry,
                "status": status,
                "paired_days": len(r),
                "rl_model_seeds": rl.model_seed.nunique(),
                "rl_cost": float(r.mean()),
                "baseline_cost": mean_baseline,
                "cost_saved": float(delta.mean()),
                "cost_saved_ci95_low": low,
                "cost_saved_ci95_high": high,
                "cost_saved_pct": pct,
                "cost_saved_pct_ci95_low": low_pct,
                "cost_saved_pct_ci95_high": high_pct,
            }
            for metric in ("mean_wait", "p95_wait", "abandoned_share", "unfinished_share"):
                if metric in left:
                    row[f"rl_minus_baseline_{metric}"] = float(
                        (left[metric] - right[metric]).mean()
                    )
            rows.append(row)
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def selected_checkpoint(group: pd.DataFrame) -> pd.Series:
    eligible = (
        group[group.checkpoint_eligible.isin([True, "True", "true", 1])]
        if "checkpoint_eligible" in group
        else group
    )
    if eligible.empty:
        raise ValueError("training history has no checkpoint-eligible evaluation")
    eligible = eligible.sort_values("timesteps")
    return eligible.loc[eligible.val_cost.idxmin()]


def summarize_training(history: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if history.empty:
        return pd.DataFrame(columns=TRAINING_COLUMNS)
    for run_id, group in history.groupby("run_id", sort=False):
        group = group.sort_values("timesteps")
        best = selected_checkpoint(group)
        first, last = group.iloc[0], group.iloc[-1]
        reduction = (
            100 * (first.val_cost - best.val_cost) / first.val_cost
            if first.val_cost > 0
            else np.nan
        )
        rows.append(
            {
                "run_id": run_id,
                "seed": first.seed,
                "encoder": first.encoder,
                "history_minutes": first.history_minutes,
                "reference_timestep": first.timesteps,
                "reference_kind": "untrained" if first.timesteps == 0 else "first_logged",
                "reference_val_cost": first.val_cost,
                "best_val_cost": best.val_cost,
                "best_timesteps": best.timesteps,
                "last_timesteps": last.timesteps,
                "last_val_cost": last.val_cost,
                "validation_reduction_pct": reduction,
                "validation_days": best.get("val_days", np.nan),
                "wall_time_s": last.get("train_elapsed_wall_s", np.nan),
            }
        )
    return pd.DataFrame(rows, columns=TRAINING_COLUMNS)
