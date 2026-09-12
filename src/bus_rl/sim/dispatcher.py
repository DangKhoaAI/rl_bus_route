"""Terminal dispatcher and accepted-action effects."""

from __future__ import annotations

from bus_rl.domain import Action, Pattern, Phase, Scenario, WorldState
from bus_rl.sim.vehicles import begin_service_edge


def _assign_short(state: WorldState, scenario: Scenario, bus_id: int, route_id: int) -> None:
    bus = state.vehicles[bus_id]
    bus.route_id = route_id
    bus.pattern = Pattern.SHORT
    bus.turn_stop = scenario.network.routes[route_id].short_turn_stop
    bus.direction = 1
    bus.pending_extra = True
    bus.cooldown_until_s = state.current_time_s + 1200


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
        bus.direction = 1
        bus.pending_extra = True
        bus.cooldown_until_s = state.current_time_s + 1200
    elif action.kind == "RECALL":
        bus = state.vehicles[action.bus_id]
        bus.route_id = None
        bus.phase = Phase.DEADHEAD
        bus.next_node = scenario.network.depot_node
        bus.remaining_s = 360
        bus.pattern = Pattern.FULL
        bus.turn_stop = None
        bus.pending_extra = False
        bus.cooldown_until_s = state.current_time_s + 1200
    elif action.kind == "REASSIGN":
        bus = state.vehicles[action.bus_id]
        bus.route_id = action.route_id
        bus.phase = Phase.DEADHEAD
        bus.next_node = scenario.network.routes[action.route_id].stops[0]
        bus.remaining_s = 360
        bus.direction = 1
        bus.pattern = Pattern.FULL
        bus.turn_stop = None
        bus.pending_extra = True
        bus.cooldown_until_s = state.current_time_s + 1200
    elif action.kind == "SHORT_TURN":
        bus = state.vehicles[action.bus_id]
        _assign_short(state, scenario, action.bus_id, action.route_id)
        if bus.phase is Phase.DEPOT_IDLE:
            bus.phase = Phase.DEADHEAD
            bus.next_node = scenario.network.routes[action.route_id].stops[0]
            bus.remaining_s = 360
    elif action.kind == "SET_HEADWAY":
        state.headway_targets_s[action.route_id] = action.headway_s
        state.headway_changed_at_s[action.route_id] = state.current_time_s
        return 0
    return 1


def _terminals(route):
    return (
        (1, route.stops[0]),
        (-1, route.stops[-1]),
        (-1, route.stops[route.short_turn_stop]),
    )


def dispatch_ready(state: WorldState, scenario: Scenario) -> None:
    for route in scenario.network.routes:
        for direction, node in _terminals(route):
            key = (route.route_id, direction, node)
            last_any = state.last_departure_s.get(key, -900)
            if state.current_time_s - last_any < 120:
                continue
            extra = [
                vehicle
                for vehicle in state.vehicles.values()
                if vehicle.pending_extra
                and vehicle.route_id == route.route_id
                and vehicle.phase is Phase.TERMINAL_IDLE
                and vehicle.node == node
                and vehicle.direction == direction
            ]
            if extra:
                bus = min(extra, key=lambda vehicle: vehicle.vehicle_id)
                begin_service_edge(state, scenario, bus.vehicle_id)
                bus.pending_extra = False
                state.last_departure_s[key] = state.current_time_s
                if bus.pattern is Pattern.FULL:
                    state.last_full_departure_s[key] = state.current_time_s
                continue
            last_full = state.last_full_departure_s.get(key, -900)
            if state.current_time_s - last_full < state.headway_targets_s[route.route_id]:
                continue
            candidates = [
                vehicle
                for vehicle in state.vehicles.values()
                if vehicle.route_id == route.route_id
                and vehicle.pattern is Pattern.FULL
                and vehicle.phase is Phase.TERMINAL_IDLE
                and vehicle.node == node
                and vehicle.direction == direction
            ]
            if not candidates:
                continue
            bus = min(candidates, key=lambda vehicle: vehicle.vehicle_id)
            begin_service_edge(state, scenario, bus.vehicle_id)
            state.last_departure_s[key] = state.current_time_s
            state.last_full_departure_s[key] = state.current_time_s
