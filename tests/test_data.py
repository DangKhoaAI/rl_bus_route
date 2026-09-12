import numpy as np

from bus_rl.data.io import load_scenario, save_scenario
from bus_rl.data.scenario import generate_manifest, generate_scenario
from bus_rl.domain import SimConfig
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
