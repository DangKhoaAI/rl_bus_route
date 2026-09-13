"""Wall-clock and RSS profiling of engine, passengers, sensors and PPO."""

from __future__ import annotations

import cProfile
import io
import pstats
import resource
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
from sb3_contrib import MaskablePPO

from bus_rl.execution.environments.python.environment import BusDispatchEnv
from bus_rl.execution.scenarios.generation import generate_scenario
from bus_rl.learning.policy import POLICY_KWARGS
from bus_sim.oracle.domain import SimConfig, scenario_digest


def _empty_and_light(config: SimConfig) -> tuple:
    peak = generate_scenario(11, config)
    empty_arrivals = np.zeros_like(peak.arrival_tape)
    empty_arrivals.setflags(write=False)
    empty = replace(
        peak,
        arrival_tape=empty_arrivals,
        scenario_hash=scenario_digest(
            peak.config, peak.network, peak.fleet, empty_arrivals, peak.traffic_tape, peak.seed
        ),
        seed=11,
    )
    light_arrivals = np.array(peak.arrival_tape // 4, dtype=np.int32)
    light_arrivals.setflags(write=False)
    light = replace(
        peak,
        arrival_tape=light_arrivals,
        scenario_hash=scenario_digest(
            peak.config, peak.network, peak.fleet, light_arrivals, peak.traffic_tape, 12
        ),
        seed=12,
    )
    return empty, light, peak


def _bucket(stats: pstats.Stats) -> dict[str, float]:
    mapping = {
        "engine": (
            "bus_sim.oracle/engine.py",
            "bus_sim.oracle/vehicles.py",
            "bus_sim.oracle/dispatcher.py",
            "bus_sim.oracle/travel.py",
        ),
        "passenger": ("bus_sim.oracle/passengers.py",),
        "sensor": ("bus_sim.oracle/observation.py",),
        "ppo": ("torch/", "stable_baselines3", "sb3_contrib", "bus_rl/models"),
    }
    totals = {name: 0.0 for name in mapping}
    totals["other"] = 0.0
    for func, stat in stats.stats.items():
        filename = func[0].replace("\\", "/")
        tottime = stat[2]
        matched = False
        for name, needles in mapping.items():
            if any(needle in filename for needle in needles):
                totals[name] += tottime
                matched = True
                break
        if not matched:
            totals["other"] += tottime
    return totals


def _rss_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def profile_demand(name: str, scenario, decisions: int, model) -> dict:
    env = BusDispatchEnv([scenario], scenario.config)
    observation, _ = env.reset(seed=0, options={"scenario_index": 0})
    profiler = cProfile.Profile()
    started = perf_counter()
    profiler.enable()
    terminated = False
    steps = 0
    while steps < decisions and not terminated:
        mask = env.action_masks()
        action, _ = model.predict(observation, action_masks=mask, deterministic=True)
        observation, _, terminated, _, _ = env.step(int(action))
        steps += 1
    profiler.disable()
    elapsed = perf_counter() - started
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    buckets = _bucket(stats)
    return {
        "demand": name,
        "decisions": steps,
        "wall_s": elapsed,
        "decisions_per_s": steps / elapsed if elapsed else 0.0,
        "ticks_per_s": steps * 4 / elapsed if elapsed else 0.0,
        "peak_rss_kb": _rss_kb(),
        **buckets,
    }


def run_profile(output: Path, config: SimConfig, decisions: int = 120) -> list[dict]:
    empty, light, peak = _empty_and_light(config)
    env = BusDispatchEnv([empty], empty.config)
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        gamma=1.0,
        n_steps=16,
        batch_size=16,
        n_epochs=1,
        seed=11,
        device="cpu",
        verbose=0,
        policy_kwargs=POLICY_KWARGS,
    )
    rows = [
        profile_demand("empty", empty, decisions, model),
        profile_demand("light", light, decisions, model),
        profile_demand("peak", peak, decisions, model),
    ]
    output.mkdir(parents=True, exist_ok=True)
    import json

    (output / "profile.json").write_text(json.dumps(rows, indent=2))
    return rows
