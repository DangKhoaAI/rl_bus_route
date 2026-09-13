import numpy as np

from bus_rl.execution.scenarios.generation import generate_manifest, generate_scenario
from bus_sim.oracle.domain import SimConfig


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
