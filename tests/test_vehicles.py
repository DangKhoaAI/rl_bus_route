from dataclasses import replace

import numpy as np

from bus_rl.domain import Phase, initial_state
from bus_rl.sim.travel import edge_duration_s
from bus_rl.sim.vehicles import begin_service_edge
from tests.fixtures import empty_scenario


def test_edge_duration_is_frozen_at_entry():
    scenario = empty_scenario()
    state = initial_state(scenario)
    bus = state.vehicles[0]
    begin_service_edge(state, scenario, 0)
    duration = bus.remaining_s
    assert bus.phase is Phase.SERVICE_MOVING
    assert duration % scenario.config.tick_s == 0
    assert duration >= scenario.config.tick_s
    traffic = np.array(scenario.traffic_tape, copy=True)
    traffic.fill(2.5)
    later = edge_duration_s(replace(scenario, traffic_tape=traffic), 0, 0, 0)
    assert bus.remaining_s == duration
    assert later != duration


def test_reverse_terminal_edge_uses_last_valid_edge():
    scenario = empty_scenario()
    assert edge_duration_s(scenario, 0, 5, 0) >= scenario.config.tick_s
