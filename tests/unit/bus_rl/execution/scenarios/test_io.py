from dataclasses import replace

import numpy as np
import pytest

from bus_rl.execution.scenarios.io import load_scenario, save_scenario
from bus_sim.oracle.domain import SimConfig
from tests.support.factories import empty_scenario


def test_scenario_roundtrip_has_same_demand(tmp_path):
    scenario = empty_scenario()
    save_scenario(scenario, tmp_path / "scenario")
    restored = load_scenario(tmp_path / "scenario")
    np.testing.assert_array_equal(restored.arrival_tape, scenario.arrival_tape)
    assert restored.scenario_hash == scenario.scenario_hash


def test_invalid_config_and_tapes_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        SimConfig(route_count=5)
    with pytest.raises(ValueError):
        SimConfig(fleet_size=17)
    with pytest.raises(ValueError):
        SimConfig(demand_end_s=1)
    scenario = empty_scenario()
    bad = np.array(scenario.arrival_tape, copy=True)
    bad[0, 0, 0, 0, 0] = 1
    with pytest.raises(ValueError):
        save_scenario(replace(scenario, arrival_tape=bad), tmp_path / "upstream")
    negative = np.array(scenario.arrival_tape, copy=True)
    negative[0, 0, 0, 0, 1] = -3
    with pytest.raises(ValueError):
        save_scenario(replace(scenario, arrival_tape=negative), tmp_path / "negative")
