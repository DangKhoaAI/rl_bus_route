"""Record core-11's action mask and action at every decision of test_id day 0.

Writes core11-test_id-day0-masks.csv next to this file: one row per decision with the
minute, the chosen action index, and the 221-bit mask as a 0/1 string.

    uv run python reports/provenance/rerun-2026-10-22/core11_day0_masks.py
"""

from pathlib import Path

import pandas as pd
from sb3_contrib import MaskablePPO

from bus_rl.config import load_run_config
from bus_rl.evaluation.runner import make_controller
from bus_rl.execution.environments.factory import make_env_for_run
from bus_rl.execution.scenarios.io import load_split

ROOT = Path(__file__).resolve().parents[3]

run = load_run_config(ROOT / "configs/eval.toml")
env = make_env_for_run(load_split(ROOT / "data/generated/core/manifest.json", "test_id"), run, forecaster=None)
controller = make_controller("ppo", model=MaskablePPO.load(ROOT / "runs/core-11/best.zip"), seed=11)
observation, _ = env.reset(seed=0, options={"scenario_index": 0})
rows, step = [], 0
while True:
    mask = env.action_masks()
    action = int(controller.act(observation, mask))
    rows.append({"minute": step * 2, "action_index": action, "mask": "".join("1" if bit else "0" for bit in mask)})
    observation, _, terminated, truncated, _ = env.step(action)
    step += 1
    if terminated or truncated:
        break
pd.DataFrame(rows).to_csv(Path(__file__).with_name("core11-test_id-day0-masks.csv"), index=False)
print(len(rows), "decisions")
