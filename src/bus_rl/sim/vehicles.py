"""Vehicle phase transitions; actions/dispatch are intentionally deferred to Task 3."""

from __future__ import annotations

from bus_rl.domain import Phase, Scenario, Vehicle, WorldState
from bus_rl.sim.passengers import alight_visit, board_visit
from bus_rl.sim.travel import edge_duration_s


def begin_service_edge(state: WorldState, scenario: Scenario, bus_id: int) -> None:
    bus = state.vehicles[bus_id]
    if bus.route_id is None or bus.phase is not Phase.TERMINAL_IDLE:
        raise ValueError("only an assigned terminal-idle bus may begin service")
    stop_count = scenario.config.stops_per_route
    current_index = scenario.network.routes[bus.route_id].stops.index(bus.node)
    target_index = current_index + bus.direction
    if not 0 <= target_index < stop_count:
        raise ValueError("service edge would leave the route")
    bus.next_node = scenario.network.routes[bus.route_id].stops[target_index]
    bus.remaining_s = edge_duration_s(scenario, bus.route_id, current_index, state.current_time_s)
    bus.phase = Phase.SERVICE_MOVING


def complete_expired_phase(state: WorldState, scenario: Scenario, bus: Vehicle) -> tuple[int, int]:
    """Complete one phase event, returning boarded and first-denied counters."""
    if bus.phase is Phase.SERVICE_MOVING:
        assert bus.next_node is not None and bus.route_id is not None
        bus.node = bus.next_node
        bus.next_node = None
        stop_index = scenario.network.routes[bus.route_id].stops.index(bus.node)
        alight_visit(state, bus.vehicle_id, stop_index)
        # A terminal has mandatory layover; intermediate stops serve and dwell.
        if stop_index in (0, scenario.config.stops_per_route - 1):
            bus.phase, bus.remaining_s = Phase.LAYOVER, scenario.config.layover_s
            bus.direction *= -1
            return 0, 0
        event = board_visit(state, bus.vehicle_id, bus.route_id, bus.direction, stop_index)
        bus.phase, bus.remaining_s = Phase.SERVICE_DWELL, scenario.config.dwell_s
        return event.boarded_count, event.first_denied_count
    if bus.phase is Phase.SERVICE_DWELL or bus.phase is Phase.LAYOVER:
        bus.phase = Phase.TERMINAL_IDLE
    return 0, 0
