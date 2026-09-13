"""Vehicle phase transitions for full service and short-turn missions."""

from __future__ import annotations

from bus_sim.oracle.domain import Pattern, Phase, Scenario, Vehicle, WorldState
from bus_sim.oracle.passengers import alight_visit, board_visit
from bus_sim.oracle.travel import edge_duration_s


def _turn_stop(bus: Vehicle, scenario: Scenario) -> int:
    if bus.turn_stop is not None:
        return bus.turn_stop
    assert bus.route_id is not None
    return scenario.network.routes[bus.route_id].short_turn_stop


def _service_end_index(bus: Vehicle, scenario: Scenario) -> int:
    assert bus.route_id is not None
    if bus.pattern is Pattern.SHORT:
        return _turn_stop(bus, scenario) if bus.direction == 1 else 0
    last = scenario.config.stops_per_route - 1
    return last if bus.direction == 1 else 0


def enter_next_edge(state: WorldState, scenario: Scenario, bus_id: int) -> None:
    bus = state.vehicles[bus_id]
    if bus.route_id is None:
        raise ValueError("unassigned bus cannot enter a service edge")
    route = scenario.network.routes[bus.route_id]
    current_index = route.stops.index(bus.node)
    target_index = current_index + bus.direction
    if bus.pattern is Pattern.SHORT:
        limit = _turn_stop(bus, scenario)
        if not 0 <= target_index <= limit:
            raise ValueError("short pattern would leave the allowed stops")
    elif not 0 <= target_index < len(route.stops):
        raise ValueError("service edge would leave the route")
    bus.next_node = route.stops[target_index]
    bus.remaining_s = edge_duration_s(scenario, bus.route_id, current_index, state.current_time_s)
    bus.phase = Phase.SERVICE_MOVING


def begin_service_edge(state: WorldState, scenario: Scenario, bus_id: int) -> None:
    bus = state.vehicles[bus_id]
    if bus.route_id is None or bus.phase is not Phase.TERMINAL_IDLE:
        raise ValueError("only an assigned terminal-idle bus may begin service")
    enter_next_edge(state, scenario, bus_id)


def complete_expired_phase(state: WorldState, scenario: Scenario, bus: Vehicle) -> tuple[int, int]:
    """Complete one phase event, returning boarded and first-denied counters."""
    if bus.phase is Phase.DEADHEAD:
        assert bus.next_node is not None
        bus.node, bus.next_node = bus.next_node, None
        bus.phase = (
            Phase.DEPOT_IDLE if bus.node == scenario.network.depot_node else Phase.TERMINAL_IDLE
        )
        return 0, 0
    if bus.phase is Phase.SERVICE_MOVING:
        assert bus.next_node is not None and bus.route_id is not None
        bus.node = bus.next_node
        bus.next_node = None
        stop_index = scenario.network.routes[bus.route_id].stops.index(bus.node)
        alight_visit(state, bus.vehicle_id, stop_index)
        if stop_index == _service_end_index(bus, scenario):
            bus.phase, bus.remaining_s = Phase.LAYOVER, scenario.config.layover_s
            bus.direction *= -1
            return 0, 0
        event = board_visit(state, bus.vehicle_id, bus.route_id, bus.direction, stop_index)
        bus.phase, bus.remaining_s = Phase.SERVICE_DWELL, scenario.config.dwell_s
        return event.boarded_count, event.first_denied_count
    if bus.phase is Phase.SERVICE_DWELL:
        enter_next_edge(state, scenario, bus.vehicle_id)
        return 0, 0
    if bus.phase is Phase.LAYOVER:
        bus.phase = Phase.TERMINAL_IDLE
        if bus.pattern is Pattern.SHORT and bus.route_id is not None:
            stop_index = scenario.network.routes[bus.route_id].stops.index(bus.node)
            if stop_index == 0:
                bus.pattern = Pattern.FULL
                bus.turn_stop = None
                bus.pending_extra = False
            else:
                bus.pending_extra = True
    return 0, 0
