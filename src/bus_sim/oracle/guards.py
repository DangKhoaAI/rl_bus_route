from __future__ import annotations

import numpy as np

from bus_sim.oracle.actions import ACTION_TABLE
from bus_sim.oracle.domain import Pattern, Phase, Scenario, WorldState


def _is_full_committed(vehicle, route_id: int) -> bool:
    return (
        vehicle.route_id == route_id
        and vehicle.pattern is Pattern.FULL
        and vehicle.phase not in (Phase.DEPOT_IDLE, Phase.DEADHEAD)
    )


def _committed(state: WorldState, route_id: int) -> int:
    return sum(_is_full_committed(vehicle, route_id) for vehicle in state.vehicles.values())


def _departure_key(route_id: int, direction: int, node: int) -> tuple[int, int, int]:
    return route_id, direction, node


def _donor_guard(state: WorldState, bus_id: int) -> bool:
    bus = state.vehicles[bus_id]
    if (
        bus.route_id is None
        or bus.phase is not Phase.TERMINAL_IDLE
        or bus.load
        or bus.pattern is not Pattern.FULL
    ):
        return False
    if _committed(state, bus.route_id) - 1 < 2:
        return False
    replacement = any(
        other.vehicle_id != bus_id
        and other.route_id == bus.route_id
        and other.pattern is Pattern.FULL
        and other.phase is Phase.TERMINAL_IDLE
        and other.node == bus.node
        and other.load == 0
        for other in state.vehicles.values()
    )
    if not replacement:
        return False
    last_any = state.last_departure_s.get(
        _departure_key(bus.route_id, bus.direction, bus.node), -900
    )
    last_full = state.last_full_departure_s.get(
        _departure_key(bus.route_id, bus.direction, bus.node), -900
    )
    earliest = max(state.current_time_s, last_any + 120)
    return earliest <= last_full + 1200


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
                scenario.enable_reassign
                and bus
                and action.route_id is not None
                and action.route_id < scenario.config.route_count
                and bus.route_id != action.route_id
                and state.current_time_s >= bus.cooldown_until_s
                and _donor_guard(state, action.bus_id)
            )
        elif action.kind == "SHORT_TURN":
            bus = state.vehicles.get(action.bus_id)
            route_id = action.route_id
            if not (
                scenario.enable_short_turn
                and bus
                and route_id is not None
                and route_id < scenario.config.route_count
                and state.current_time_s >= bus.cooldown_until_s
                and bus.load == 0
            ):
                continue
            if bus.phase is Phase.DEPOT_IDLE:
                mask[index] = True
            else:
                origin = scenario.network.routes[route_id].stops[0]
                mask[index] = bool(
                    bus.route_id == route_id
                    and bus.pattern is Pattern.FULL
                    and bus.phase is Phase.TERMINAL_IDLE
                    and bus.node == origin
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
