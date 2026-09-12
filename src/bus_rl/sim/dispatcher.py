"""M1 terminal dispatcher and accepted-action effects."""

from __future__ import annotations

from bus_rl.domain import Action, Phase, Scenario, WorldState
from bus_rl.sim.vehicles import begin_service_edge


def apply_action(state: WorldState, scenario: Scenario, action: Action) -> int:
    if action.kind == "NOOP":
        return 0
    from bus_rl.control.actions import action_id
    from bus_rl.control.guards import valid_action_mask

    if not valid_action_mask(state, scenario)[
        action_id(action.kind, action.bus_id, action.route_id, action.headway_s)
    ]:
        raise ValueError(f"invalid action: {action}")
    if action.kind == "DISPATCH":
        bus = state.vehicles[action.bus_id]
        bus.route_id = action.route_id
        bus.phase = Phase.DEADHEAD
        bus.next_node = scenario.network.routes[action.route_id].stops[0]
        bus.remaining_s = 360
        bus.cooldown_until_s = state.current_time_s + 1200
    elif action.kind == "RECALL":
        bus = state.vehicles[action.bus_id]
        bus.route_id = None
        bus.phase = Phase.DEADHEAD
        bus.next_node = scenario.network.depot_node
        bus.remaining_s = 360
        bus.cooldown_until_s = state.current_time_s + 1200
    elif action.kind == "REASSIGN":
        bus = state.vehicles[action.bus_id]
        bus.route_id = action.route_id
        bus.phase = Phase.DEADHEAD
        bus.next_node = scenario.network.routes[action.route_id].stops[0]
        bus.remaining_s = 360
        bus.cooldown_until_s = state.current_time_s + 1200
    elif action.kind == "SET_HEADWAY":
        state.headway_targets_s[action.route_id] = action.headway_s
        state.headway_changed_at_s[action.route_id] = state.current_time_s
        return 0
    return 1


def dispatch_ready(state: WorldState, scenario: Scenario) -> None:
    for route in scenario.network.routes:
        for direction, terminal_index in ((1, 0), (-1, len(route.stops) - 1)):
            key = (route.route_id, direction)
            gap = state.current_time_s - state.last_departure_s[key]
            if gap < max(120, state.headway_targets_s[route.route_id]):
                continue
            candidates = [
                v
                for v in state.vehicles.values()
                if v.route_id == route.route_id
                and v.phase is Phase.TERMINAL_IDLE
                and v.node == route.stops[terminal_index]
                and v.direction == direction
            ]
            if not candidates:
                continue
            bus = min(candidates, key=lambda v: v.vehicle_id)
            begin_service_edge(state, scenario, bus.vehicle_id)
            state.last_departure_s[key] = state.current_time_s
