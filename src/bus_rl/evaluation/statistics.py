"""Paired bootstrap over scenario days after averaging model seeds."""

from __future__ import annotations

import numpy as np
import pandas as pd


def paired_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    n_resamples: int = 2_000,
    seed: int = 6_001,
) -> tuple[float, float, float]:
    """Return mean(left-right) and a 95% percentile CI, resampling days."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("paired bootstrap requires aligned day arrays")
    if left.size == 0:
        raise ValueError("paired bootstrap requires at least one day")
    diffs = left - right
    rng = np.random.default_rng(seed)
    n_days = diffs.size
    samples = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        draw = rng.integers(0, n_days, n_days)
        samples[index] = diffs[draw].mean()
    point = float(diffs.mean())
    low, high = np.quantile(samples, [0.025, 0.975])
    return point, float(low), float(high)


def mean_over_seeds(frame: pd.DataFrame, value: str = "total_cost_core") -> pd.Series:
    """Average model seeds per scenario day; baselines without seeds stay as-is."""
    grouped = frame.groupby("scenario_seed", sort=True)[value]
    return grouped.mean()


def compare_methods(
    frame: pd.DataFrame,
    left_method: str,
    right_method: str,
    value: str = "total_cost_core",
    n_resamples: int = 2_000,
    seed: int = 6_001,
) -> dict[str, float]:
    left = mean_over_seeds(frame[frame["method"] == left_method], value)
    right = mean_over_seeds(frame[frame["method"] == right_method], value)
    aligned = left.index.intersection(right.index)
    point, low, high = paired_bootstrap(
        left.loc[aligned].to_numpy(),
        right.loc[aligned].to_numpy(),
        n_resamples=n_resamples,
        seed=seed,
    )
    seed_std = float(
        frame[frame["method"] == left_method].groupby("model_seed")[value].mean().std()
    )
    return {
        "mean_diff": point,
        "ci_low": low,
        "ci_high": high,
        "n_days": float(len(aligned)),
        "left_seed_std": seed_std if np.isfinite(seed_std) else 0.0,
    }
