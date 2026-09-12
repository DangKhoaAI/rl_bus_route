from __future__ import annotations

import numpy as np

from bus_rl.domain import PassengerStatus, Phase, Scenario, WorldState


def observe(state: WorldState, scenario: Scenario) -> dict[str, np.ndarray]:
    stops, vehicles, routes = (
        np.zeros((4, 2, 8, 7), np.float32),
        np.zeros((16, 27), np.float32),
        np.zeros((4, 8), np.float32),
    )
    stop_valid, vehicle_valid, route_valid = (
        np.zeros((4, 2, 8), np.float32),
        np.zeros(16, np.float32),
        np.zeros(4, np.float32),
    )
    for cohort in state.cohorts:
        if cohort.status is PassengerStatus.WAITING:
            direction = 0 if cohort.direction == 1 else 1
            stops[cohort.route_id, direction, cohort.origin_index, 0] += cohort.count / 40
            stops[cohort.route_id, direction, cohort.origin_index, 2] = max(
                stops[cohort.route_id, direction, cohort.origin_index, 2],
                (state.current_time_s - cohort.arrival_tick * scenario.config.tick_s) / 2700,
            )
    for route in scenario.network.routes:
        route_valid[route.route_id] = 1
        stop_valid[route.route_id, :, : len(route.stops)] = 1
        routes[route.route_id, 0] = state.headway_targets_s[route.route_id] / 1200
    for vehicle in state.vehicles.values():
        vehicle_valid[vehicle.vehicle_id] = 1
        vehicles[vehicle.vehicle_id, list(Phase).index(vehicle.phase)] = 1
        vehicles[vehicle.vehicle_id, 21] = vehicle.load / 40
    time = state.current_time_s / scenario.config.horizon_s
    return {
        "stops": stops,
        "arrival_history": np.zeros((4, 2, 8, 5), np.float32),
        "forecast": np.zeros((4, 2, 8), np.float32),
        "vehicles": vehicles,
        "routes": routes,
        "stop_valid": stop_valid,
        "vehicle_valid": vehicle_valid,
        "route_valid": route_valid,
        "context": np.array([time, 1 - time, 0], np.float32),
    }
