import pytest

from bus_sim.oracle.domain import PassengerCohort, initial_state
from bus_sim.oracle.observation import observe
from bus_sim.oracle.passengers import alight_visit, board_visit
from bus_sim.parity.scenarios import build_scenario


def test_completed_passengers_remain_in_arrival_history_but_not_boarded_channel():
    scenario = build_scenario({"kind": "empty", "seed": 2001})
    state = initial_state(scenario)
    state.add_waiting(PassengerCohort(0, 0, 0, 1, 0, 5, 0, 3))
    board_visit(state, 0, 0, 1, 0)  # boards at tick 0
    alight_visit(state, 0, 5)  # completes at tick 0
    state.current_time_s = 120
    observation = observe(state, scenario)

    # Finished passengers still populate the 5-bin arrival history.
    assert observation["arrival_history"][0, 0, 0, 0] == pytest.approx(3 / 40)
    # Completed channel counts them; boarded channel only counts currently ONBOARD.
    assert observation["stops"][0, 0, 0, 5] == pytest.approx(3 / 40)
    assert observation["stops"][0, 0, 0, 4] == pytest.approx(0.0)
    assert observation["stops"][0, 0, 0, 0] == pytest.approx(0.0)
