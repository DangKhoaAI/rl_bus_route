from __future__ import annotations

import numpy as np

from bus_rl.domain import PassengerStatus, Pattern, Phase, Scenario, WorldState

PHASES = tuple(Phase)
ROUTE_OFFSET = 6  # depot at 6, routes 0..3 at 7..10
PATTERN_OFFSET = 11
DIRECTION_OFFSET = 13
NODE_OFFSET = 15
TARGET_OFFSET = 17
SCALAR_OFFSET = 22


def validate_observation(observation: dict[str, np.ndarray]) -> None:
    for key, value in observation.items():
        if not np.isfinite(value).all():
            raise ValueError(f"NaN or inf in observation[{key}]")


def _age_s(state: WorldState, cohort, tick_s: int) -> float:
    return max(0.0, float(state.current_time_s - cohort.arrival_tick * tick_s))


def observe(state: WorldState, scenario: Scenario) -> dict[str, np.ndarray]:
    config = scenario.config
    stops = np.zeros((4, 2, 8, 7), np.float32)
    vehicles = np.zeros((16, 27), np.float32)
    routes = np.zeros((4, 8), np.float32)
    stop_valid = np.zeros((4, 2, 8), np.float32)
    vehicle_valid = np.zeros(16, np.float32)
    route_valid = np.zeros(4, np.float32)
    history = np.zeros((4, 2, 8, 5), np.float32)
    node_count = scenario.network.depot_node + 1
    age_sum = np.zeros((4, 2, 8), np.float64)
    age_count = np.zeros((4, 2, 8), np.float64)
    last_start = state.current_time_s - config.control_interval_s

    for cohort in state.cohorts:
        direction = 0 if cohort.direction == 1 else 1
        origin = cohort.origin_index
        if cohort.status is PassengerStatus.WAITING:
            stops[cohort.route_id, direction, origin, 0] += cohort.count / 40
            age = _age_s(state, cohort, config.tick_s)
            age_sum[cohort.route_id, direction, origin] += age * cohort.count
            age_count[cohort.route_id, direction, origin] += cohort.count
            stops[cohort.route_id, direction, origin, 2] = max(
                stops[cohort.route_id, direction, origin, 2], age / 2700
            )
            if age >= 900:
                stops[cohort.route_id, direction, origin, 3] += cohort.count / 40
        arrival_s = cohort.arrival_tick * config.tick_s
        if 0 <= arrival_s < state.current_time_s:
            lag = (state.current_time_s - 1 - arrival_s) // config.control_interval_s
            if 0 <= lag < 5:
                history[cohort.route_id, direction, origin, lag] += cohort.count / 40
            if last_start <= arrival_s < state.current_time_s:
                stops[cohort.route_id, direction, origin, 6] += cohort.count / 40
        if (
            cohort.status is PassengerStatus.ONBOARD
            and cohort.boarding_tick is not None
            and last_start <= cohort.boarding_tick * config.tick_s < state.current_time_s
        ):
            stops[cohort.route_id, direction, origin, 4] += cohort.count / 40
        if (
            cohort.status is PassengerStatus.COMPLETED
            and cohort.completion_tick is not None
            and last_start <= cohort.completion_tick * config.tick_s < state.current_time_s
        ):
            stops[cohort.route_id, direction, origin, 5] += cohort.count / 40

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_age = np.divide(age_sum, age_count, out=np.zeros_like(age_sum), where=age_count > 0)
    stops[..., 1] = (mean_age / 2700).astype(np.float32)

    for route in scenario.network.routes:
        route_valid[route.route_id] = 1
        stop_valid[route.route_id, :, : len(route.stops)] = 1
        plus = route.stops[0]
        minus = route.stops[-1]
        routes[route.route_id, 0] = state.headway_targets_s[route.route_id] / 1200
        routes[route.route_id, 1] = (
            state.current_time_s - state.last_full_departure_s.get((route.route_id, 1, plus), -900)
        ) / 1200
        routes[route.route_id, 2] = (
            state.current_time_s
            - state.last_full_departure_s.get((route.route_id, -1, minus), -900)
        ) / 1200
        full = incoming = short = 0
        for vehicle in state.vehicles.values():
            if vehicle.route_id != route.route_id:
                continue
            if vehicle.pattern is Pattern.SHORT:
                short += 1
            if vehicle.phase is Phase.DEADHEAD:
                incoming += 1
            elif vehicle.pattern is Pattern.FULL:
                full += 1
        routes[route.route_id, 3] = full / 16
        routes[route.route_id, 4] = incoming / 16
        routes[route.route_id, 5] = short / 16
        elapsed = state.current_time_s - state.headway_changed_at_s[route.route_id]
        routes[route.route_id, 6] = max(0.0, 600 - elapsed) / 600
        routes[route.route_id, 7] = state.depot_count / 16

    def encode_node(node: int | None) -> float:
        return 0.0 if node is None else (node + 1) / (node_count + 1)

    for vehicle in state.vehicles.values():
        vehicle_valid[vehicle.vehicle_id] = 1
        row = vehicles[vehicle.vehicle_id]
        row[PHASES.index(vehicle.phase)] = 1
        if vehicle.route_id is None:
            row[ROUTE_OFFSET] = 1
            row[TARGET_OFFSET] = 1
        else:
            row[ROUTE_OFFSET + 1 + vehicle.route_id] = 1
            row[TARGET_OFFSET + 1 + vehicle.route_id] = 1
        row[PATTERN_OFFSET + (0 if vehicle.pattern is Pattern.FULL else 1)] = 1
        row[DIRECTION_OFFSET + (0 if vehicle.direction == 1 else 1)] = 1
        row[NODE_OFFSET] = encode_node(vehicle.node)
        row[NODE_OFFSET + 1] = encode_node(vehicle.next_node)
        horizon = config.horizon_s
        row[SCALAR_OFFSET] = vehicle.load / 40
        row[SCALAR_OFFSET + 1] = vehicle.remaining_s / horizon
        row[SCALAR_OFFSET + 2] = (
            0.0 if vehicle.phase is Phase.TERMINAL_IDLE else vehicle.remaining_s / horizon
        )
        row[SCALAR_OFFSET + 3] = max(0, vehicle.cooldown_until_s - state.current_time_s) / 1200
        row[SCALAR_OFFSET + 4] = vehicle.remaining_s / horizon

    time = state.current_time_s / config.horizon_s
    observation = {
        "stops": stops,
        "arrival_history": history,
        "forecast": np.zeros((4, 2, 8), np.float32),
        "vehicles": vehicles,
        "routes": routes,
        "stop_valid": stop_valid,
        "vehicle_valid": vehicle_valid,
        "route_valid": route_valid,
        "context": np.array([time, 1 - time, 0], np.float32),
    }
    validate_observation(observation)
    return observation
