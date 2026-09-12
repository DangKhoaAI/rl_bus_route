from __future__ import annotations

import numpy as np

from bus_rl.control.actions import ACTION_TABLE
from bus_rl.domain import Phase, Scenario, WorldState


def _committed(state: WorldState, route_id: int) -> int:
    return sum(
        v.route_id == route_id and v.phase is not Phase.DEPOT_IDLE for v in state.vehicles.values()
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
                and bus.route_id is not None
                and bus.phase is Phase.TERMINAL_IDLE
                and bus.load == 0
                and state.current_time_s >= bus.cooldown_until_s
                and _committed(state, bus.route_id) > 2
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
