from dataclasses import replace

import numpy as np
import pytest

from bus_rl.execution.environments.python.environment import BusDispatchEnv
from bus_rl.execution.scenarios.generation import generate_manifest, generate_scenario
from bus_rl.execution.scenarios.io import load_scenario, save_scenario
from bus_sim.oracle.domain import SimConfig
from tests.fixtures import empty_scenario


def test_scenario_roundtrip_has_same_demand(tmp_path):
    scenario = empty_scenario()
    save_scenario(scenario, tmp_path / "scenario")
    restored = load_scenario(tmp_path / "scenario")
    np.testing.assert_array_equal(restored.arrival_tape, scenario.arrival_tape)
    assert restored.scenario_hash == scenario.scenario_hash


def test_seed_is_deterministic_and_has_no_late_arrivals():
    config = SimConfig()
    left, right = generate_scenario(123, config), generate_scenario(123, config)
    np.testing.assert_array_equal(left.arrival_tape, right.arrival_tape)
    assert left.scenario_hash == right.scenario_hash
    assert not left.arrival_tape[config.demand_end_s // config.tick_s :].any()


def test_manifest_has_unique_child_seeds_and_shared_topology():
    scenarios = generate_manifest("train", 3)
    assert len({scenario.seed for scenario in scenarios}) == 3
    assert len({scenario.scenario_hash for scenario in scenarios}) == 3
    assert all(scenario.network == scenarios[0].network for scenario in scenarios)


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


def test_reset_and_control_do_not_mutate_scenario():
    scenario = generate_scenario(19)
    tape = np.array(scenario.arrival_tape)
    digest = scenario.scenario_hash
    env = BusDispatchEnv([scenario], scenario.config)
    env.reset(seed=3)
    env.step(0)
    np.testing.assert_array_equal(scenario.arrival_tape, tape)
    assert scenario.scenario_hash == digest
