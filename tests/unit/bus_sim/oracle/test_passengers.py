from bus_sim.oracle.costs import integrate_tick_costs
from bus_sim.oracle.domain import PassengerCohort, PassengerStatus, Pattern, initial_state
from bus_sim.oracle.passengers import abandon_expired, alight_visit, board_visit
from tests.support.factories import empty_scenario, short_state, waiting_state


def test_capacity_split_keeps_lineage():
    state = waiting_state(45)
    lineage = state.cohorts[0].lineage_id
    board_visit(state, bus_id=0, route_id=0, direction=1, stop_index=0)
    onboard = next(c for c in state.vehicles[0].passengers)
    waiting = next(c for c in state.cohorts)
    assert onboard.lineage_id == waiting.lineage_id == lineage


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


def test_partial_boarding_denies_once_and_preserves_lineage():
    state = waiting_state(45, destination=5, stage="M3")
    first = board_visit(state, 0, 0, 1, 0)
    assert first.boarded_count == 40
    assert first.first_denied_count == 5
    assert state.waiting_count == 5 and state.onboard_count == 40
    assert state.vehicles[0].load == 40

    # A second visit with no free capacity must not re-count the same denial.
    second = board_visit(state, 0, 0, 1, 0)
    assert second.boarded_count == 0 and second.first_denied_count == 0

    # Free capacity, then the remainder boards with the original lineage.
    alight_visit(state, 0, 5)
    third = board_visit(state, 0, 0, 1, 0)
    assert third.boarded_count == 5
    assert state.waiting_count == 0
    assert state.vehicles[0].passengers[-1].lineage_id == 0
    assert state.vehicles[0].load <= state.vehicles[0].capacity


def test_alighting_happens_only_at_destination_and_patience_is_checked_before_boarding():
    state = waiting_state(1, destination=1)
    board_visit(state, 0, 0, 1, 0)
    assert alight_visit(state, 0, 0) == 0
    assert alight_visit(state, 0, 1) == 1
    state = waiting_state(1)
    state.current_time_s = 2_700
    assert abandon_expired(state, 2_700, 30) == 1
    assert state.finished[0].status is PassengerStatus.ABANDONED


def test_abandonment_uses_the_patience_boundary():
    state = waiting_state(4, destination=5)
    state.current_time_s = 2670
    assert abandon_expired(state, 2700, 30) == 0
    state.current_time_s = 2700
    assert abandon_expired(state, 2700, 30) == 4
    assert state.waiting_count == 0 and state.abandoned_count == 4


def test_short_turn_does_not_strand_long_distance_passenger():
    state = short_state()
    event = board_visit(state, bus_id=0, route_id=0, direction=1, stop_index=0)
    assert event.boarded_count == 0
    assert event.first_denied_count == 0
    assert state.waiting_count == 1
    assert state.cohorts[0].route_id == 0
    state.current_time_s = 900
    assert integrate_tick_costs(state, 30).excessive_wait_pm > 0


def test_short_turn_boards_inbound_after_turnaround():
    scenario = empty_scenario("M3")
    state = initial_state(scenario)
    bus = state.vehicles[0]
    bus.pattern = Pattern.SHORT
    bus.turn_stop = 3
    bus.node = scenario.network.routes[0].stops[3]
    bus.direction = -1
    state.add_waiting(PassengerCohort(0, 0, 0, -1, 3, 1, 0, 1))
    event = board_visit(state, bus.vehicle_id, 0, -1, 3)
    assert event.boarded_count == 1
    assert alight_visit(state, bus.vehicle_id, 2) == 0
    assert state.onboard_count == 1
    assert alight_visit(state, bus.vehicle_id, 1) == 1
