from __future__ import annotations

import numpy as np

from bus_rl.control.actions import ACTION_TABLE
from bus_rl.domain import Phase, Scenario, WorldState


def _committed(state: WorldState, route_id: int) -> int:
    return sum(
        v.route_id == route_id and v.phase not in (Phase.DEPOT_IDLE, Phase.DEADHEAD)
        for v in state.vehicles.values()
    )


def _donor_guard(state: WorldState, bus_id: int) -> bool:
    bus = state.vehicles[bus_id]
    if bus.route_id is None or bus.phase is not Phase.TERMINAL_IDLE or bus.load:
        return False
    if _committed(state, bus.route_id) - 1 < 2:
        return False
    return any(
        other.vehicle_id != bus_id
        and other.route_id == bus.route_id
        and other.phase is Phase.TERMINAL_IDLE
        and other.node == bus.node
        and other.load == 0
        for other in state.vehicles.values()
    )


def valid_action_mask(state: WorldState, scenario: Scenario) -> np.ndarray:
    mask = np.zeros(len(ACTION_TABLE), dtype=bool)
    mask[0] = True
    for index, action in enumerate(ACTION_TABLE[1:], 1):
        if action.kind == "DISPATCH":
            bus = state.vehicles.get(action.bus_id)
            mask[index] = bool(
                bus
                and action.route_id is not None
                and action.route_id < scenario.config.route_count
                and bus.phase is Phase.DEPOT_IDLE
                and state.current_time_s >= bus.cooldown_until_s
            )
        elif action.kind == "RECALL":
            bus = state.vehicles.get(action.bus_id)
            mask[index] = bool(
                bus
                and state.current_time_s >= bus.cooldown_until_s
                and _donor_guard(state, action.bus_id)
            )
        elif action.kind == "REASSIGN":
            bus = state.vehicles.get(action.bus_id)
            mask[index] = bool(
                bus
                and action.route_id is not None
                and action.route_id < scenario.config.route_count
                and bus.route_id != action.route_id
                and state.current_time_s >= bus.cooldown_until_s
                and _donor_guard(state, action.bus_id)
            )
        elif action.kind == "SET_HEADWAY":
            route = action.route_id
            mask[index] = bool(
                route is not None
                and route < scenario.config.route_count
                and state.headway_targets_s[route] != action.headway_s
                and state.current_time_s - state.headway_changed_at_s[route] >= 600
            )
    return mask
