from __future__ import annotations

from dataclasses import replace

import gymnasium as gym
import numpy as np

from bus_rl.control.actions import ACTION_TABLE
from bus_rl.control.guards import valid_action_mask
from bus_rl.domain import initial_state
from bus_rl.env.observation import observe, validate_observation
from bus_rl.rewards.costs import RewardConfig, interval_cost
from bus_rl.sim.engine import advance_interval


class BusDispatchEnv(gym.Env):
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
        self.state = None
        self.scenario = None

    def _observation(self):
        observation = observe(self.state, self.scenario)
        if self.forecaster is not None:
            forecast = self.forecaster.predict(observation, self.state.current_time_s)
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
        self.scenario, self.state = self.scenarios[index], initial_state(self.scenarios[index])
        return self._observation(), {}

    def action_masks(self):
        return valid_action_mask(self.state, self.scenario)

    def step(self, action_index):
        if not self.action_masks()[action_index]:
            raise ValueError(f"invalid action index: {action_index}")
        costs = advance_interval(self.state, self.scenario, ACTION_TABLE[action_index])
        terminated = self.state.current_time_s >= self.config.horizon_s
        return (
            self._observation(),
            -interval_cost(costs, self.reward) / self.reward.n_ref,
            terminated,
            False,
            {"costs": costs},
        )
