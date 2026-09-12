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
from bus_rl.timing import TIMERS


class BusDispatchEnv(gym.Env):
    def __init__(
        self,
        scenarios,
        config,
        forecaster=None,
        reward=None,
        control=None,
        validate=True,
    ):
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
        self.validate = validate

    def _observation(self):
        with TIMERS.span("env.observe"):
            observation = observe(self.state, self.scenario)
        if self.forecaster is not None:
            with TIMERS.span("env.forecast"):
                forecast = self.forecaster.predict(observation, self.state.current_time_s)
            observation["forecast"] = np.asarray(forecast.expected, dtype=np.float32)
            observation["context"] = np.array(
                [observation["context"][0], observation["context"][1], 1.0], np.float32
            )
        with TIMERS.span("env.validate_obs"):
            if self.validate:
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
        with TIMERS.span("env.action_masks"):
            return valid_action_mask(self.state, self.scenario)

    def summary_inputs(self):
        """Episode-end inputs for the shared evaluator interface (R3.2)."""
        from bus_rl.evaluation.summary import from_python_state

        return from_python_state(self.state)

    def trace_snapshot(self) -> dict:
        """Per-decision trace row shared with the native backend (R3.2)."""
        state = self.state
        queues = [
            sum(cohort.count for cohort in state.cohorts if cohort.route_id == route)
            for route in range(self.config.route_count)
        ]
        buses = [
            {
                "id": vehicle.vehicle_id,
                "phase": vehicle.phase.name,
                "route_id": vehicle.route_id,
                "pattern": vehicle.pattern.name,
                "load": vehicle.load,
            }
            for vehicle in state.vehicles.values()
        ]
        return {
            "time_s": state.current_time_s,
            "queues": queues,
            "headway_targets": [
                state.headway_targets_s[route] for route in range(self.config.route_count)
            ],
            "buses": buses,
            "waiting": state.waiting_count,
            "onboard": state.onboard_count,
            "generated": state.generated_count,
            "abandoned": state.abandoned_count,
            "completed": state.completed_count,
        }

    def step(self, action_index):
        with TIMERS.span("env.step"):
            if not self.action_masks()[action_index]:
                raise ValueError(f"invalid action index: {action_index}")
            with TIMERS.span("env.advance_interval"):
                costs = advance_interval(self.state, self.scenario, ACTION_TABLE[action_index])
            terminated = self.state.current_time_s >= self.config.horizon_s
            observation = self._observation()
            with TIMERS.span("env.interval_cost"):
                reward = -interval_cost(costs, self.reward) / self.reward.n_ref
            return observation, reward, terminated, False, {"costs": costs}
