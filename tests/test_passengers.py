from bus_rl.domain import PassengerStatus
from bus_rl.sim.passengers import abandon_expired, alight_visit, board_visit
from tests.fixtures import waiting_state


def test_capacity_denial_keeps_waiting_passengers():
    state = waiting_state(45)
    event = board_visit(state, bus_id=0, route_id=0, direction=1, stop_index=0)
    assert event.boarded_count == 40
    assert event.first_denied_count == 5
    assert state.waiting_count == 5
    assert state.onboard_count == 40
    assert state.generated_count == state.waiting_count + state.onboard_count


def test_second_visit_does_not_repeat_first_denial():
    state = waiting_state(45)
    board_visit(state, 0, 0, 1, 0)
    assert board_visit(state, 0, 0, 1, 0).first_denied_count == 0


def test_alighting_happens_only_at_destination_and_patience_is_checked_before_boarding():
    state = waiting_state(1, destination=1)
    board_visit(state, 0, 0, 1, 0)
    assert alight_visit(state, 0, 0) == 0
    assert alight_visit(state, 0, 1) == 1
    state = waiting_state(1)
    state.current_time_s = 2_700
    assert abandon_expired(state, 2_700, 30) == 1
    assert state.cohorts[0].status is PassengerStatus.ABANDONED
