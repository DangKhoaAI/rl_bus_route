"""The committed historical checkpoint used as the evaluation anchor."""

from __future__ import annotations

from dataclasses import replace

from bus_rl.config import ControlConfig, load_run_config
from bus_rl.learning.training.checkpoint import load_metadata
from bus_sim.oracle.costs import RewardConfig
from tests.support.paths import CONFIGS, REFERENCE, ROOT


def reference_run(config: str = "experiments/core-threads2.toml", *, backend: str | None = None):
    """Load a run config carrying the reference checkpoint's control/reward."""
    run = load_run_config(CONFIGS / config, ROOT)
    metadata = load_metadata(REFERENCE)
    reward = RewardConfig(
        **{
            key: metadata["reward"][key]
            for key in RewardConfig.__dataclass_fields__
            if key in metadata["reward"]
        }
    )
    run = replace(run, control=ControlConfig(**metadata["control"]), reward=reward)
    if backend is not None:
        run = replace(run, runtime=replace(run.runtime, backend=backend))
    return run
