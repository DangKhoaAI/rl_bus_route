from bus_sim.oracle.travel import edge_duration_s
from tests.support.factories import empty_scenario


def test_reverse_terminal_edge_uses_last_valid_edge():
    scenario = empty_scenario()
    assert edge_duration_s(scenario, 0, 5, 0) >= scenario.config.tick_s
