from bus_sim.oracle.costs import add_costs, integrate_tick_costs, interval_cost
from bus_sim.oracle.domain import Phase, StepCosts
from bus_sim.oracle.engine import advance_tick
from tests.support.factories import empty_scenario, waiting_state


def test_ten_people_waiting_five_minutes_is_fifty():
    costs = integrate_tick_costs(waiting_state(10), duration_s=300)
    assert costs.waiting_pm == 50.0
    parts = [integrate_tick_costs(waiting_state(10), duration_s=30) for _ in range(10)]
    assert sum(part.waiting_pm for part in parts) == costs.waiting_pm


def test_terminal_unfinished_and_event_costs_have_distinct_terms():
    costs = integrate_tick_costs(waiting_state(5), duration_s=30)
    costs.first_denied_count = 5
    costs.abandoned_count = 5
    costs.terminal_unfinished_count = 5
    assert interval_cost(costs) == 629.75


def test_first_denied_and_abandon_are_once_only_events():
    scenario = empty_scenario()
    state = waiting_state(45)
    totals = StepCosts()
    for _ in range(8):
        totals = add_costs(totals, advance_tick(state, scenario))
    assert totals.first_denied_count == 5
    parked = waiting_state(3)
    for bus in parked.vehicles.values():
        bus.phase = Phase.DEPOT_IDLE
        bus.route_id = None
    abandoned = StepCosts()
    for _ in range(100):
        abandoned = add_costs(abandoned, advance_tick(parked, scenario))
        if parked.abandoned_count:
            break
    assert parked.abandoned_count == 3
    assert abandoned.abandoned_count == 3
    extra = StepCosts()
    for _ in range(4):
        extra = add_costs(extra, advance_tick(parked, scenario))
    assert extra.abandoned_count == 0
