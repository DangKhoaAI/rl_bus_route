from bus_rl.rewards.costs import integrate_tick_costs, interval_cost
from tests.fixtures import waiting_state


def test_ten_people_waiting_five_minutes_is_fifty():
    costs = integrate_tick_costs(waiting_state(10), duration_s=300)
    assert costs.waiting_pm == 50.0


def test_terminal_unfinished_and_event_costs_have_distinct_terms():
    costs = integrate_tick_costs(waiting_state(5), duration_s=30)
    costs.first_denied_count = 5
    costs.abandoned_count = 5
    costs.terminal_unfinished_count = 5
    assert interval_cost(costs) == 629.75
