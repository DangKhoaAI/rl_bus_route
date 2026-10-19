#!/usr/bin/env python
"""Build the two human-facing benchmark tables from curated L3 tables.

Inputs (committed):
  reports/tables/baselines_summary.csv
  reports/tables/baseline_contrasts.csv
  reports/tables/held_out_per_seed.csv
  reports/tables/ablation_contrasts.csv

Outputs:
  reports/tables/main_benchmark.csv
  reports/tables/ablations.csv

Run after ``scripts/l3_analysis.py``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
BUDGET = 245_760
EVAL_DAYS = 200
SPLITS = ["test_id", "test_ood_burst", "test_ood_traffic"]
BASELINES = ["fixed", "proportional", "threshold"]
ABLATIONS = ["no_reassign", "no_short", "fairness_zero"]


def _core_row(held: pd.DataFrame, split: str) -> dict:
    sub = held[(held["arm"] == "core") & (held["split"] == split)]
    assert len(sub) == 3, (split, len(sub))
    return {
        "split": split,
        "method": "core",
        "eval_days": EVAL_DAYS,
        "seeds": len(sub),
        "mean_cost_per_episode": float(sub["mean_cost"].mean()),
        "seed_std_cost": float(sub["mean_cost"].std(ddof=1)),
        "delta_vs_core": 0.0,
        "delta_ci95_low": 0.0,
        "delta_ci95_high": 0.0,
        "delta_pct_of_core": 0.0,
        "seeds_lower_than_core": 0,
        "mean_wait_min": float(sub["mean_wait"].mean()),
        "p95_wait_min": float(sub["p95_wait"].mean()),
        "unfinished_share": float(sub["unfinished_share"].mean()),
        "completed_share": float(sub["completed_share"].mean()),
    }


def build_main(held: pd.DataFrame, baselines: pd.DataFrame, contrast: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        rows.append(_core_row(held, split))
        for method in BASELINES:
            base = baselines[(baselines["method"] == method) & (baselines["split"] == split)]
            delta = contrast[(contrast["baseline"] == method) & (contrast["split"] == split)]
            assert len(base) == 1 and len(delta) == 1, (split, method)
            base = base.iloc[0]
            delta = delta.iloc[0]
            rows.append(
                {
                    "split": split,
                    "method": method,
                    "eval_days": int(base["days"]),
                    "seeds": 1,
                    "mean_cost_per_episode": float(base["mean_cost"]),
                    "seed_std_cost": 0.0,
                    "delta_vs_core": float(delta["mean_delta"]),
                    "delta_ci95_low": float(delta["ci_low"]),
                    "delta_ci95_high": float(delta["ci_high"]),
                    "delta_pct_of_core": float(delta["pct_of_core"]),
                    "seeds_lower_than_core": pd.NA,
                    "mean_wait_min": float(base["mean_wait"]),
                    "p95_wait_min": float(base["p95_wait"]),
                    "unfinished_share": float(base["unfinished_share"]),
                    "completed_share": float(base["completed_share"]),
                }
            )
    frame = pd.DataFrame(rows)
    order = {split: i for i, split in enumerate(SPLITS)}
    frame["_split_order"] = frame["split"].map(order)
    frame["_method_order"] = frame["method"].map(
        {"core": 0, "fixed": 1, "proportional": 2, "threshold": 3}
    )
    frame = frame.sort_values(["_split_order", "_method_order"]).drop(
        columns=["_split_order", "_method_order"]
    )
    return frame.reset_index(drop=True)


def _violations(row: pd.Series) -> str:
    failed = []
    if row["delta_unfinished"] > 0.005:
        failed.append("unfinished")
    if row["delta_abandoned"] > 0.002:
        failed.append("abandoned")
    if row["delta_p95"] > 1.0:
        failed.append("p95_wait")
    if row["delta_worst"] > 1.0:
        failed.append("worst_route_wait")
    return ";".join(failed)


def build_ablations(contrast: pd.DataFrame) -> pd.DataFrame:
    sub = contrast[contrast["arm"].isin(ABLATIONS)].copy()
    assert len(sub) == len(ABLATIONS) * len(SPLITS), len(sub)
    sub["service_limit_violations"] = sub.apply(_violations, axis=1)
    out = sub.rename(
        columns={
            "arm": "ablation",
            "mean_delta": "delta_cost",
            "ci_low": "delta_ci95_low",
            "ci_high": "delta_ci95_high",
            "pct_of_core": "delta_pct_of_core",
            "delta_p95": "delta_p95_wait_min",
            "delta_worst": "delta_worst_route_wait_min",
            "service_limits_satisfied": "service_limit_pass",
        }
    )
    out["budget_transitions"] = BUDGET
    out["eval_days"] = EVAL_DAYS
    out = out[
        [
            "split",
            "ablation",
            "budget_transitions",
            "eval_days",
            "delta_cost",
            "delta_ci95_low",
            "delta_ci95_high",
            "delta_pct_of_core",
            "seeds_lower_than_core",
            "seed_std",
            "delta_p95_wait_min",
            "delta_worst_route_wait_min",
            "delta_unfinished",
            "delta_abandoned",
            "service_limit_pass",
            "service_limit_violations",
        ]
    ]
    order = {split: i for i, split in enumerate(SPLITS)}
    arm_order = {arm: i for i, arm in enumerate(ABLATIONS)}
    out["_split_order"] = out["split"].map(order)
    out["_arm_order"] = out["ablation"].map(arm_order)
    out = out.sort_values(["_split_order", "_arm_order"]).drop(
        columns=["_split_order", "_arm_order"]
    )
    return out.reset_index(drop=True)


def main() -> None:
    held = pd.read_csv(TABLES / "held_out_per_seed.csv")
    baselines = pd.read_csv(TABLES / "baselines_summary.csv")
    baseline_contrast = pd.read_csv(TABLES / "baseline_contrasts.csv")
    ablation_contrast = pd.read_csv(TABLES / "ablation_contrasts.csv")

    main_table = build_main(held, baselines, baseline_contrast)
    ablations = build_ablations(ablation_contrast)

    main_table.to_csv(TABLES / "main_benchmark.csv", index=False)
    ablations.to_csv(TABLES / "ablations.csv", index=False)
    print(main_table.round(4).to_string(index=False))
    print()
    print(ablations.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
