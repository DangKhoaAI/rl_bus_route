"""Gym wrapper over the native Rust kernel (R3.1).

Mirrors :class:`bus_rl.env.bus_dispatch.BusDispatchEnv`: same spaces, seeding,
scenario selection, step tuple, invalid-action rejection and forecast handling.
The native kernel is packed once per scenario at construction; no Python
``WorldState`` is rebuilt during training.
"""

from __future__ import annotations

from dataclasses import replace

import gymnasium as gym
import numpy as np

from bus_rl.backend.native import build_kernel
from bus_rl.config import RunConfig
from bus_rl.domain import StepCosts, initial_state
from bus_rl.env.observation import observe, validate_observation
from bus_rl.evaluation.summary import from_native_payload
from bus_rl.rewards.costs import RewardConfig


def _step_costs(values) -> StepCosts:
    (
        waiting,
        onboard,
        crowding,
        active,
        deadhead,
        excessive,
        denied,
        abandoned,
        missions,
        terminal,
    ) = list(values)
    return StepCosts(
        waiting_pm=float(waiting),
        onboard_pm=float(onboard),
        crowding_pm=float(crowding),
        active_bus_min=float(active),
        deadhead_bus_min=float(deadhead),
        excessive_wait_pm=float(excessive),
        first_denied_count=int(denied),
        abandoned_count=int(abandoned),
        mission_changes=int(missions),
        terminal_unfinished_count=int(terminal),
    )


class NativeBusDispatchEnv(gym.Env):
    def __init__(self, scenarios, config, forecaster=None, reward=None, control=None):
        applied = tuple(scenarios)
        if control is not None:
            applied = tuple(
                replace(
                    scenario,
                    enable_reassign=control.enable_reassign,
                    enable_short_turn=control.enable_short_turn,
                )
                for scenario in applied
            )
        if not applied:
            raise ValueError("at least one scenario is required")
        self.scenarios, self.config = applied, config
        self.forecaster = forecaster
        self.reward = reward or RewardConfig()
        self.action_space = gym.spaces.Discrete(221)
        self.observation_space = gym.spaces.Dict(
            {
                key: gym.spaces.Box(-np.inf, np.inf, value.shape, np.float32)
                for key, value in observe(
                    initial_state(self.scenarios[0]), self.scenarios[0]
                ).items()
            }
        )
        # Pack every scenario once; each kernel fully owns its tape buffers.
        self._kernels = tuple(build_kernel(scenario) for scenario in self.scenarios)
        self.kernel = None
        self.scenario = None
        self._scenario_index: int | None = None

    @property
    def scenario_index(self) -> int | None:
        return self._scenario_index

    def _observation(self, raw) -> dict[str, np.ndarray]:
        observation = {key: np.asarray(value) for key, value in raw.items()}
        if self.forecaster is not None:
            forecast = self.forecaster.predict(observation, int(self.kernel.current_time_s))
            observation["forecast"] = np.asarray(forecast.expected, dtype=np.float32)
            observation["context"] = np.array(
                [observation["context"][0], observation["context"][1], 1.0], np.float32
            )
        validate_observation(observation)
        return observation

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        index = (options or {}).get(
            "scenario_index", int(self.np_random.integers(len(self.scenarios)))
        )
        self._scenario_index = index
        self.scenario = self.scenarios[index]
        self.kernel = self._kernels[index]
        result = self.kernel.reset_contract()
        return self._observation(result["obs"]), {}

    def action_masks(self) -> np.ndarray:
        return np.asarray(self.kernel.action_mask(), dtype=bool)

    def step(self, action_index):
        result = self.kernel.step_contract(int(action_index))
        observation = self._observation(result["obs"])
        return (
            observation,
            float(result["reward"]),
            bool(result["terminated"]),
            bool(result["truncated"]),
            {"costs": _step_costs(result["costs"])},
        )

    def summary_inputs(self):
        """Episode-end inputs for the shared evaluator interface."""
        return from_native_payload(self.kernel.episode_summary_inputs())

    def trace_snapshot(self) -> dict:
        return dict(self.kernel.trace_snapshot())


def env_from_run(scenarios, run: RunConfig, forecaster=None) -> NativeBusDispatchEnv:
    return NativeBusDispatchEnv(
        scenarios,
        run.physical,
        forecaster=forecaster,
        reward=run.reward,
        control=run.control,
    )
