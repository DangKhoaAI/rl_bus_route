"""R0.2 targeted coverage: guards, capacity, abandonment, channels, settlement."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bus_sim.oracle.actions import ACTION_TABLE, action_id
from bus_sim.oracle.dispatcher import apply_action
from bus_sim.oracle.domain import Action, PassengerCohort, Phase, initial_state
from bus_sim.oracle.guards import valid_action_mask
from bus_sim.oracle.observation import observe
from bus_sim.oracle.passengers import abandon_expired, alight_visit, board_visit
from bus_sim.parity.fixtures import load_fixture, new_env
from bus_sim.parity.scenarios import CATALOG, build_scenario
from bus_sim.parity.snapshot import numpy_state_snapshot
from tests.fixtures import empty_scenario, waiting_state

FIXTURES = Path(__file__).parent / "fixtures"
ALL_FAMILIES = {"NOOP", "DISPATCH", "REASSIGN", "SHORT_TURN", "RECALL", "SET_HEADWAY"}


def test_every_action_family_is_exercised_by_the_catalog():
    seen = set()
    for name in CATALOG:
        payload, _, _ = load_fixture(FIXTURES / name)
        seen.update(ACTION_TABLE[action].kind for action in payload["actions"])
    assert seen == ALL_FAMILIES


def test_invalid_action_is_rejected_without_mutating_state():
    env = new_env(CATALOG["normal_m3"])
    before = numpy_state_snapshot(env.state)
    invalid = int(np.flatnonzero(~env.action_masks())[0])
    with pytest.raises(ValueError):
        env.step(invalid)
    after = numpy_state_snapshot(env.state)
    for key in before:
        assert np.array_equal(before[key], after[key]), f"state[{key}] mutated on rejection"


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


def test_abandonment_uses_the_patience_boundary():
    state = waiting_state(4, destination=5)
    state.current_time_s = 2670
    assert abandon_expired(state, 2700, 30) == 0
    state.current_time_s = 2700
    assert abandon_expired(state, 2700, 30) == 4
    assert state.waiting_count == 0 and state.abandoned_count == 4


def test_donor_guard_floor_and_replacement():
    scenario = empty_scenario("M3")
    state = initial_state(scenario)
    recall = action_id("RECALL", bus_id=0)
    assert valid_action_mask(state, scenario)[recall]

    # Removing one committed FULL bus violates the two-bus floor.
    state.vehicles[2].phase = Phase.DEADHEAD
    assert not valid_action_mask(state, scenario)[recall]

    # Without a ready replacement donor at the same terminal the action is invalid.
    state = initial_state(empty_scenario("M3"))
    state.vehicles[1].route_id = None
    state.vehicles[1].phase = Phase.DEPOT_IDLE
    assert not valid_action_mask(state, scenario)[recall]


def test_cooldown_blocks_retargeting_a_bus():
    scenario = empty_scenario("M3")
    state = initial_state(scenario)
    reserve = min(
        vehicle.vehicle_id
        for vehicle in state.vehicles.values()
        if vehicle.phase is Phase.DEPOT_IDLE
    )
    dispatch = action_id("DISPATCH", bus_id=reserve, route_id=0)
    assert valid_action_mask(state, scenario)[dispatch]
    apply_action(state, scenario, Action("DISPATCH", reserve, 0))
    assert state.vehicles[reserve].cooldown_until_s == 1200
    # The bus is now deadheading and on cooldown.
    assert not valid_action_mask(state, scenario)[dispatch]
    assert not valid_action_mask(state, scenario)[action_id("REASSIGN", bus_id=reserve, route_id=1)]


def test_terminal_settlement_is_charged_once_with_unfinished_backlog():
    _, expected, _ = load_fixture(FIXTURES / "backlog_m3")
    terminal = expected["costs__terminal_unfinished_count"]
    counters = expected["counters"]
    settled = expected["terminal_settled"]
    assert terminal[:-1].sum() == 0.0
    assert terminal[-1] > 0.0
    assert terminal[-1] == counters[-1][1] + counters[-1][2]
    assert settled[:-1].sum() == 0 and settled[-1] == 1


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
