from bus_rl.execution.environments.python.environment import BusDispatchEnv
from bus_rl.execution.scenarios.generation import generate_scenario
from bus_sim.oracle.costs import (
    add_costs,
    integrate_tick_costs,
    interval_cost,
    mean_waiting_minutes,
)
from bus_sim.oracle.domain import (
    Action,
    PassengerCohort,
    Phase,
    SimConfig,
    StepCosts,
    initial_state,
)
from bus_sim.oracle.engine import advance_interval, advance_tick
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


def test_horizon_settles_unfinished_and_skips_already_abandoned():
    config = SimConfig(horizon_s=120, demand_end_s=0)
    scenario = generate_scenario(5, config)
    state = initial_state(scenario)
    state.add_waiting(PassengerCohort(0, 0, 0, 1, 0, 5, 0, 4))
    state.add_waiting(PassengerCohort(1, 1, 0, 1, 0, 5, -90, 2))
    for bus in state.vehicles.values():
        bus.phase = Phase.DEPOT_IDLE
        bus.route_id = None
    first = advance_interval(state, scenario, Action())
    last = advance_interval(state, scenario, Action())
    assert state.current_time_s == 120
    assert state.abandoned_count == 2
    assert state.waiting_count == 4
    assert first.terminal_unfinished_count == 4
    assert last.terminal_unfinished_count == 0
    assert interval_cost(first) >= 60 * 4


def test_env_reward_matches_negative_raw_costs_over_nref():
    scenario = empty_scenario()
    env = BusDispatchEnv([scenario], scenario.config)
    env.reset(seed=2)
    reward_sum = 0.0
    cost_sum = 0.0
    for _ in range(8):
        _, reward, _, _, info = env.step(0)
        reward_sum += reward
        cost_sum += interval_cost(info["costs"])
        assert abs(reward - (-interval_cost(info["costs"]) / 3000)) < 1e-12
    assert abs(reward_sum + cost_sum / 3000) < 1e-12
    assert mean_waiting_minutes(initial_state(scenario)) is None
