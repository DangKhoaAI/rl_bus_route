"""Post-hoc "two-line rule" read off the PPO core's behaviour, evaluated on the held-out splits.

Rule: send the three reserve buses out at once, one per route, set every route's headway
target to 10 minutes, then NOOP. One command per decision, so it takes the first six
decisions. It was written after looking at PPO traces and is not a registered baseline.

The core's per-day costs come from the frozen `reports/tables/paired_days.csv` (L3
checkpoints are not kept), and the paired bootstrap is the registered one (2,000
resamples, seed 6001), so the deltas are comparable with `baseline_contrasts.csv`.

    uv run python scripts/two_line_rule.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bus_rl.config import load_run_config
from bus_rl.evaluation.runner import evaluate_scenarios
from bus_rl.evaluation.statistics import paired_bootstrap
from bus_rl.execution.scenarios.io import load_split
from bus_rl.learning.baselines.threshold import _depot_buses
from bus_sim.oracle.actions import action_id

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/generated/core/manifest.json"
SPLITS = ("test_id", "test_ood_burst", "test_ood_traffic")
OUTPUT = ROOT / "reports/tables/two_line_rule.csv"


class TwoLineRule:
    def act(self, observation, mask):
        for bus in _depot_buses(observation):
            # Reserve buses are ids 9, 10, 11; send them to routes A, B, C.
            choice = action_id("DISPATCH", bus, (bus - 9) % 3)
            if mask[choice]:
                return choice
        for route in range(3):
            choice = action_id("SET_HEADWAY", route_id=route, headway_s=600)
            if mask[choice]:
                return choice
        return 0


def main() -> None:
    from bus_rl.evaluation import runner

    runner.make_controller = lambda method, model=None, seed=0: TwoLineRule()
    run = load_run_config(ROOT / "configs/eval.toml")
    paired = pd.read_csv(ROOT / "reports/tables/paired_days.csv", dtype={"scenario_seed": "uint64"})
    rows = []
    for split in SPLITS:
        frame, _ = evaluate_scenarios(load_split(MANIFEST, split), run, "fixed", split=split)
        frame["scenario_seed"] = frame["scenario_seed"].astype("uint64")
        rule = frame.groupby("scenario_seed", sort=True)["total_cost_core"].mean()
        core = paired[(paired.arm == "core") & (paired.split == split)]
        core = core.set_index("scenario_seed")["core_cost"].sort_index()
        days = core.index.intersection(rule.index)
        delta, low, high = paired_bootstrap(rule.loc[days].to_numpy(), core.loc[days].to_numpy())
        rows.append(
            {
                "split": split,
                "eval_days": len(days),
                "mean_cost_per_episode": rule.loc[days].mean(),
                "core_mean_cost": core.loc[days].mean(),
                "delta_vs_core": delta,
                "delta_ci95_low": low,
                "delta_ci95_high": high,
                "delta_pct_of_core": delta / core.loc[days].mean() * 100,
                "mean_wait_min": frame["mean_wait"].mean(),
                "p95_wait_min": frame["p95_wait"].mean(),
                "abandoned_share": frame["abandoned_share"].mean(),
                "reserve_dispatches": frame["reserve_dispatches"].mean(),
            }
        )
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)
    print(pd.DataFrame(rows).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
