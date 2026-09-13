"""Native training-path forecast wiring and causality at the environment boundary.

The isolated causality contract is pinned in
``tests/unit/bus_rl/learning/test_forecasting.py``. This integration check pins
the other half required by the L2.1 experiment card: the native VecEnv actually
populates the ``forecast`` channel from the past-only observation, sets the
enabled context flag, and never lets a future tape or scenario identity change a
forecast at the same simulated time.
"""

from __future__ import annotations

import numpy as np
import pytest

from bus_rl.config import load_run_config
from bus_rl.execution.scenarios.generation import generate_scenario
from bus_rl.learning.forecasting import HistoricalForecaster
from tests.support.paths import CONFIGS

pytestmark = pytest.mark.native


def test_native_forecast_channel_is_past_only_and_wired():
    pytest.importorskip("bus_sim_native")
    from bus_rl.execution.environments.rust.batch import NativeBatchVecEnv

    run = load_run_config(CONFIGS / "experiments" / "rl-improvement" / "forecast.toml")
    train = [generate_scenario(1001), generate_scenario(1002)]
    forecaster = HistoricalForecaster(tick_s=run.physical.tick_s)
    forecaster.fit([scenario.arrival_tape for scenario in train], run.physical)
    env = NativeBatchVecEnv(train, run, run.algorithm.seed, forecaster=forecaster)
    try:
        observation = env.reset()
        forecast = observation["forecast"][0]
        assert np.isfinite(forecast).all()
        assert forecast.any()
        assert observation["context"][0, 2] == 1.0

        history = observation["arrival_history"][0]
        expected = forecaster.predict({"arrival_history": history}, 0).expected
        np.testing.assert_array_equal(forecast, expected)

        # Same past, different other observation fields / scenario identity at
        # the same time must not move the forecast.
        perturbed = {
            "arrival_history": history,
            "vehicles": np.ones_like(observation["vehicles"][0]),
            "routes": np.ones_like(observation["routes"][0]),
            "forecast": np.zeros_like(forecast),
            "context": np.array([0.0, 1.0, 0.0], np.float32),
        }
        np.testing.assert_array_equal(forecaster.predict(perturbed, 0).expected, expected)
    finally:
        env.close()
