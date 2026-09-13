from __future__ import annotations

import numpy as np
import pytest

from bus_rl.execution.scenarios.generation import generate_scenario
from bus_rl.learning.forecasting import HistoricalForecaster, history_from_tape


def test_forecast_is_causal_and_rejects_future_tape():
    scenario = generate_scenario(7)
    time_s = 100 * scenario.config.tick_s
    future = np.array(scenario.arrival_tape, copy=True)
    future[101] = future[101] + 1
    history = history_from_tape(scenario.arrival_tape, time_s, scenario.config.tick_s)
    other = history_from_tape(future, time_s, scenario.config.tick_s)
    np.testing.assert_array_equal(history["arrival_history"], other["arrival_history"])
    forecaster = HistoricalForecaster(tick_s=scenario.config.tick_s)
    forecaster.fit([scenario.arrival_tape], scenario.config)
    left = forecaster.predict(history, time_s)
    right = forecaster.predict(other, time_s)
    np.testing.assert_array_equal(left.expected, right.expected)
    with pytest.raises(ValueError):
        forecaster.predict({"arrival_history": history["arrival_history"], "seed": 7}, time_s)
    with pytest.raises(ValueError):
        forecaster.predict(
            {"arrival_history": history["arrival_history"], "arrival_tape": future},
            time_s,
        )


def test_forecast_zero_history_is_finite():
    scenario = generate_scenario(3)
    tape = np.zeros_like(scenario.arrival_tape)
    forecaster = HistoricalForecaster()
    forecaster.fit([tape, tape])
    history = history_from_tape(tape, 0)
    predicted = forecaster.predict(history, 0)
    assert np.isfinite(predicted.expected).all()


class _SentinelTape:
    def __getitem__(self, _index):
        raise AssertionError("future tape accessed")


def test_predictor_does_not_accept_scenario_seed_or_future_tape():
    forecaster = HistoricalForecaster()
    forecaster.fit([generate_scenario(1).arrival_tape])
    history = {"arrival_history": np.zeros((4, 2, 8, 5), np.float32)}
    with pytest.raises(TypeError):
        forecaster.predict(history, 0, _SentinelTape())  # type: ignore[misc]
