"""Native O2 batch stepping: one Rust call advances every active env slot.

`NativeBatchKernel` wraps `bus_sim_native.BatchKernel`. `NativeBatchVecEnv` is the SB3
`VecEnv` used by training when `runtime.native_batch=true`, and
`NativeBatchEvalPool` is the evaluator's fixed-slot pool. Both share one packed
scenario store and keep per-slot episode state in Rust.

Semantics deliberately mirror `NativeBusDispatchEnv` / `DummyVecEnv`:
per-slot ``np_random`` scenario selection, auto-reset on done, terminal
observation in info, and the same observation post-processing (forecast +
optional validation).
"""

from __future__ import annotations

from dataclasses import replace

import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import VecEnv

from bus_rl.execution.environments.rust.bridge import shared_store
from bus_sim.oracle.domain import initial_state
from bus_sim.oracle.observation import observe, validate_observation


def postprocess_observations(
    obs_raw: dict, *, times, forecaster, validate: bool
) -> dict[str, np.ndarray]:
    """Add forecast/context and optionally validate, per slot, matching the env.

    ``obs_raw`` fields already carry ``(N, *shape)`` and are returned as-is for
    the no-forecast path so training tensors stay bit-identical.
    """
    observation = {key: np.asarray(value) for key, value in obs_raw.items()}
    if forecaster is None:
        if validate:
            for index in range(observation["context"].shape[0]):
                validate_observation({key: value[index] for key, value in observation.items()})
        return observation
    n = observation["context"].shape[0]
    forecasts = []
    contexts = []
    for index in range(n):
        one = {key: value[index] for key, value in observation.items()}
        forecast = forecaster.predict(one, int(times[index]))
        forecasts.append(np.asarray(forecast.expected, dtype=np.float32))
        contexts.append([one["context"][0], one["context"][1], 1.0])
    observation["forecast"] = np.stack(forecasts)
    observation["context"] = np.asarray(contexts, dtype=np.float32)
    if validate:
        for index in range(n):
            validate_observation({key: value[index] for key, value in observation.items()})
    return observation


def _observation_space(scenarios, forecaster) -> gym.spaces.Dict:
    del forecaster
    sample = observe(initial_state(scenarios[0]), scenarios[0])
    return gym.spaces.Dict(
        {
            key: gym.spaces.Box(-np.inf, np.inf, value.shape, np.float32)
            for key, value in sample.items()
        }
    )


class NativeBatchKernel:
    """Thin owner of a `bus_sim_native.BatchKernel` over shared, packed scenarios."""

    def __init__(self, scenarios, capacity: int, *, store=None, conservation: bool = True):
        scenarios = list(scenarios)
        if not scenarios:
            raise ValueError("native batch kernel requires at least one scenario")
        capacity = int(capacity)
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self.scenarios = scenarios
        self.store = store if store is not None else shared_store(scenarios)
        self.capacity = capacity
        self._kernel = self.store.batch_kernel(capacity, conservation=conservation)
        self.last_reset_indices: dict[int, int] = {}

    def reset(self, slots, scenario_indices) -> tuple[dict, np.ndarray]:
        slots = list(slots)
        scenario_indices = list(scenario_indices)
        raw = self._kernel.reset_batch(slots, scenario_indices)
        for slot, index in zip(slots, scenario_indices, strict=True):
            self.last_reset_indices[slot] = int(index)
        return raw["obs"], np.asarray(raw["mask"])

    def masks(self, slots) -> np.ndarray:
        return np.asarray(self._kernel.mask_batch(list(slots)))

    def step(self, slots, actions) -> dict:
        return self._kernel.step_batch(list(slots), [int(action) for action in actions])

    def current_times(self, slots) -> np.ndarray:
        return np.asarray(self._kernel.current_time_s_batch(list(slots)))

    def summary_inputs(self, slot: int) -> dict:
        return self._kernel.summary_inputs_batch([int(slot)])[0]

    def trace_snapshot(self, slot: int) -> dict:
        return self._kernel.trace_snapshot_slot(int(slot))

    @property
    def poisoned(self) -> bool:
        return bool(self._kernel.poisoned)

    def scenario_indices(self) -> dict[int, int]:
        return dict(self.last_reset_indices)

    def close(self) -> None:
        self._kernel = None
        self.store = None


class NativeBatchVecEnv(VecEnv):
    """SB3 VecEnv stepping all training envs through one native batch call.

    Mirrors ``DummyVecEnv`` over ``NativeBusDispatchEnv``: each slot owns a
    seeded numpy Generator and picks its next scenario on (auto-)reset, and done
    slots are reset immediately with their terminal observation attached.
    """

    def __init__(self, scenarios, run, seed: int, forecaster=None):
        applied = tuple(
            replace(
                scenario,
                enable_reassign=run.control.enable_reassign,
                enable_short_turn=run.control.enable_short_turn,
            )
            for scenario in scenarios
        )
        if not applied:
            raise ValueError("at least one scenario is required")
        self.scenarios = applied
        self.run = run
        self.forecaster = forecaster
        self.validate = bool(run.runtime.validate_observation)
        n_envs = int(run.algorithm.n_envs)
        self.kernel = NativeBatchKernel(applied, n_envs)
        self._actions = np.zeros(n_envs, dtype=np.int64)
        super().__init__(n_envs, _observation_space(applied, forecaster), gym.spaces.Discrete(221))
        # Construction reset mirrors `make_env`: seed each slot and draw one
        # scenario. `learn()` then calls reset() once more; no extra draw here.
        self._np_random = []
        indices = []
        for slot in range(n_envs):
            generator, _ = seeding.np_random(seed + slot)
            self._np_random.append(generator)
            indices.append(int(generator.integers(len(self.scenarios))))
        slots = list(range(n_envs))
        obs_raw, _mask = self.kernel.reset(slots, indices)
        self._obs = postprocess_observations(
            obs_raw,
            times=self.kernel.current_times(slots),
            forecaster=self.forecaster,
            validate=self.validate,
        )
        self.reset_infos = [{} for _ in range(n_envs)]

    def _slots(self, indices):
        if indices is None:
            return list(range(self.num_envs))
        if isinstance(indices, (int, np.integer)):
            return [int(indices)]
        return [int(index) for index in indices]

    def _select(self, slot: int) -> int:
        return int(self._np_random[slot].integers(len(self.scenarios)))

    def reset(self) -> dict[str, np.ndarray]:
        slots = list(range(self.num_envs))
        indices = []
        for slot in slots:
            # Honor `VecEnv.seed` exactly like DummyVecEnv: SB3's
            # `set_random_seed` seeds the VecEnv, so the next reset re-seeds
            # the slot generator. Auto-reset keeps drawing without reseeding.
            seed = self._seeds[slot]
            if seed is not None:
                self._np_random[slot], _ = seeding.np_random(seed)
            indices.append(int(self._np_random[slot].integers(len(self.scenarios))))
        obs_raw, _mask = self.kernel.reset(slots, indices)
        self._obs = postprocess_observations(
            obs_raw,
            times=self.kernel.current_times(slots),
            forecaster=self.forecaster,
            validate=self.validate,
        )
        self.reset_infos = [{} for _ in range(self.num_envs)]
        self._reset_seeds()
        return {key: value.copy() for key, value in self._obs.items()}

    def step_async(self, actions: np.ndarray) -> None:
        actions = np.asarray(actions).reshape(-1)
        if actions.shape[0] != self.num_envs:
            raise ValueError(f"expected {self.num_envs} actions, got {actions.shape[0]}")
        self._actions = actions.astype(np.int64, copy=True)

    def step_wait(self):
        slots = list(range(self.num_envs))
        result = self.kernel.step(slots, self._actions.tolist())
        obs_raw = {key: np.asarray(value) for key, value in result["obs"].items()}
        # `DummyVecEnv.buf_rews` is float32; match it for bit-identical parity.
        rewards = np.asarray(result["reward"], dtype=np.float32)
        terminated = np.asarray(result["terminated"], dtype=bool)
        truncated = np.asarray(result["truncated"], dtype=bool)
        dones = terminated | truncated

        infos: list[dict] = []
        for slot in range(self.num_envs):
            info: dict = {"TimeLimit.truncated": bool(truncated[slot] and not terminated[slot])}
            if dones[slot]:
                info["terminal_observation"] = {
                    key: value[slot].copy() for key, value in obs_raw.items()
                }
            infos.append(info)

        if dones.any():
            reset_slots = [slot for slot in slots if dones[slot]]
            indices = [self._select(slot) for slot in reset_slots]
            reset_raw, _mask = self.kernel.reset(reset_slots, indices)
            for position, slot in enumerate(reset_slots):
                for key, value in reset_raw.items():
                    obs_raw[key][slot] = np.asarray(value)[position]

        self._obs = postprocess_observations(
            obs_raw,
            times=self.kernel.current_times(slots),
            forecaster=self.forecaster,
            validate=self.validate,
        )
        return (
            {key: value.copy() for key, value in self._obs.items()},
            rewards,
            dones,
            infos,
        )

    def get_attr(self, attr_name, indices=None):
        if attr_name == "render_mode":
            return [None] * len(self._slots(indices))
        raise AttributeError(attr_name)

    def set_attr(self, attr_name, value, indices=None):
        raise AttributeError(f"NativeBatchVecEnv does not expose attribute {attr_name!r}")

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        if method_name == "action_masks":
            slots = self._slots(indices)
            return list(self.kernel.masks(slots))
        raise AttributeError(method_name)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._slots(indices))

    def has_attr(self, attr_name, indices=None) -> bool:
        return attr_name == "action_masks"

    def close(self) -> None:
        if self.kernel is not None:
            self.kernel.close()
        self.kernel = None


class NativeBatchEvalPool:
    """Fixed-slot evaluator pool backed by one native batch kernel.

    Shares the same key as `EvalEnvPool`, so a pool built for the scalar/batched
    Python path is refused by the native path and vice versa.
    """

    def __init__(self, scenarios, run, forecaster=None):
        from bus_rl.evaluation.pool import eval_pool_key

        scenarios = list(scenarios)
        if not scenarios:
            raise ValueError("eval pool requires at least one scenario")
        requested = int(run.runtime.eval_batch_size)
        if requested < 1:
            raise ValueError(f"runtime.eval_batch_size must be >= 1, got {requested!r}")
        applied = tuple(
            replace(
                scenario,
                enable_reassign=run.control.enable_reassign,
                enable_short_turn=run.control.enable_short_turn,
            )
            for scenario in scenarios
        )
        self.scenarios = scenarios
        self.applied = applied
        self.run = run
        self.forecaster = forecaster
        self.requested_batch_size = requested
        self.batch_size = min(requested, len(scenarios))
        self.key = eval_pool_key(scenarios, run, forecaster)
        self.kernel = NativeBatchKernel(applied, self.batch_size)
        self.validate = bool(run.runtime.validate_observation)
        self.closed = False

    def matches(self, scenarios, run, forecaster=None) -> bool:
        from bus_rl.evaluation.pool import eval_pool_key

        if self.closed:
            return False
        return self.key == eval_pool_key(list(scenarios), run, forecaster)

    def refresh(self, scenarios, run, forecaster=None):
        if self.matches(scenarios, run, forecaster):
            return self
        self.close()
        return NativeBatchEvalPool(scenarios, run, forecaster)

    def reset(self, slots, scenario_indices):
        obs_raw, mask = self.kernel.reset(slots, scenario_indices)
        obs = postprocess_observations(
            obs_raw,
            times=self.kernel.current_times(slots),
            forecaster=self.forecaster,
            validate=self.validate,
        )
        return obs, mask

    def step(self, slots, actions) -> dict:
        return self.kernel.step(slots, actions)

    def close(self) -> None:
        if self.closed:
            return
        self.kernel.close()
        self.kernel = None
        self.closed = True
