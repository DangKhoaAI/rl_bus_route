"""Bounded evaluation env pool (O1).

The pool is owned by the evaluator or validation callback. It shares one
immutable native scenario store across slots, resets cleanly per episode, and
is not a global unbounded cache. Callers must invalidate when the scenario
order/hash, flags, config, forecast or contract change.
"""

from __future__ import annotations

from hashlib import sha256

import numpy as np

from bus_rl.config import RunConfig
from bus_rl.execution.environments.factory import make_env_for_run
from bus_rl.provenance import (
    action_schema_hash,
    observation_schema_hash,
    physical_config_hash,
)


def _forecaster_fingerprint(forecaster) -> tuple:
    if forecaster is None:
        return ("none",)
    payload: list = [type(forecaster).__qualname__]
    for attr in ("bin_s", "tick_s", "version"):
        if hasattr(forecaster, attr):
            payload.append((attr, getattr(forecaster, attr)))
    for attr in ("next15", "past10"):
        value = getattr(forecaster, attr, None)
        if isinstance(value, np.ndarray):
            digest = sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
            payload.append((attr, value.shape, str(value.dtype), digest))
        elif value is not None:
            payload.append((attr, value))
    return tuple(payload)


def eval_pool_key(scenarios, run: RunConfig, forecaster=None) -> tuple:
    """Identity of a pool: scenario order/hash, flags, config, forecast, contract."""
    scenarios = tuple(scenarios)
    return (
        tuple(scenario.scenario_hash for scenario in scenarios),
        tuple(
            (bool(scenario.enable_reassign), bool(scenario.enable_short_turn))
            for scenario in scenarios
        ),
        run.runtime.backend,
        bool(run.runtime.validate_observation),
        int(run.runtime.eval_batch_size),
        bool(run.runtime.native_batch),
        run.control_hash,
        run.reward_hash,
        physical_config_hash(run.physical),
        action_schema_hash(),
        observation_schema_hash(run.physical),
        bool(run.forecast.enabled),
        _forecaster_fingerprint(forecaster),
    )


class EvalEnvPool:
    """Fixed-size env slots sharing an immutable store.

    ``batch_size`` is ``min(requested, n_scenarios)``. Extra requested slots
    above the remaining day count are not created.
    """

    def __init__(self, scenarios, run: RunConfig, forecaster=None):
        scenarios = list(scenarios)
        if not scenarios:
            raise ValueError("eval pool requires at least one scenario")
        requested = int(run.runtime.eval_batch_size)
        if requested < 1:
            raise ValueError(f"runtime.eval_batch_size must be >= 1, got {requested!r}")
        self.scenarios = scenarios
        self.run = run
        self.forecaster = forecaster
        self.requested_batch_size = requested
        self.batch_size = min(requested, len(scenarios))
        self.key = eval_pool_key(scenarios, run, forecaster)
        # Let each env apply control flags, then hit the shared-store cache.
        # Packing before that would freeze the un-applied M-flags into the kernel.
        self.envs = [
            make_env_for_run(scenarios, run, forecaster=forecaster) for _ in range(self.batch_size)
        ]
        self.scenario_store = getattr(self.envs[0], "_scenario_store", None)
        self.closed = False

    def matches(self, scenarios, run: RunConfig, forecaster=None) -> bool:
        if self.closed:
            return False
        return self.key == eval_pool_key(list(scenarios), run, forecaster)

    def refresh(self, scenarios, run: RunConfig, forecaster=None) -> EvalEnvPool:
        if self.matches(scenarios, run, forecaster):
            return self
        self.close()
        return EvalEnvPool(scenarios, run, forecaster)

    def close(self) -> None:
        if self.closed:
            return
        for env in self.envs:
            closer = getattr(env, "close", None)
            if closer is not None:
                closer()
        self.envs.clear()
        self.scenario_store = None
        self.closed = True


def make_eval_pool(scenarios, run: RunConfig, forecaster=None):
    """Build the pool for the configured evaluator (native batch vs Python)."""
    if run.runtime.native_batch:
        from bus_rl.execution.environments.rust.batch import NativeBatchEvalPool

        return NativeBatchEvalPool(scenarios, run, forecaster)
    return EvalEnvPool(scenarios, run, forecaster)
