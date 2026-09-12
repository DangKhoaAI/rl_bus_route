from __future__ import annotations

import math

from bus_rl.control.actions import action_id
from bus_rl.env.observation import ROUTE_OFFSET


def short_eligible_share(stops, route_id: int, short_stop: int = 3, stop_count: int = 6) -> float:
    """Estimate short-turn share from origin queues and the declared destination prior."""
    eligible = 0.0
    total = 0.0
    for direction_index, direction in enumerate((1, -1)):
        origins = range(stop_count - 1) if direction == 1 else range(1, stop_count)
        for origin in origins:
            queue = float(stops[route_id, direction_index, origin, 0]) * 40
            if queue <= 0:
                continue
            destinations = range(origin + 1, stop_count) if direction == 1 else range(origin)
            weights = [math.exp(-abs(destination - origin) / 2) for destination in destinations]
            weight_total = sum(weights)
            if not weight_total:
                continue
            short_weight = sum(
                weight
                for destination, weight in zip(destinations, weights, strict=True)
                if (destination <= short_stop if direction == 1 else origin <= short_stop)
            )
            total += queue
            eligible += queue * short_weight / weight_total
    return eligible / total if total else 0.0


def _queues(observation):
    return observation["stops"][:, :, :, 0].sum(axis=(1, 2)) * 40


def _urgency(observation, route: int) -> float:
    queue = float(_queues(observation)[route])
    full = max(1.0, float(observation["routes"][route, 3]) * 16)
    max_age = float(observation["stops"][route, :, :, 2].max()) * 2700
    max_gap = float(max(observation["routes"][route, 1], observation["routes"][route, 2])) * 1200
    return queue / (40 * full) + max_age / 900 + max_gap / 1200


def _best_route(observation) -> int:
    best, best_score = 0, -1.0
    for route in range(int(observation["route_valid"].sum()) or 3):
        score = _urgency(observation, route)
        if score > best_score:
            best, best_score = route, score
    return best


def _depot_buses(observation) -> list[int]:
    return [bus for bus in range(16) if observation["vehicles"][bus, 0] == 1]


def _bus_route(observation, bus: int) -> int | None:
    row = observation["vehicles"][bus]
    if row[ROUTE_OFFSET] == 1:
        return None
    for route in range(4):
        if row[ROUTE_OFFSET + 1 + route] == 1:
            return route
    return None


class ThresholdController:
    def act(self, observation, mask):
        route = _best_route(observation)
        queue = float(_queues(observation)[route])
        share = short_eligible_share(observation["stops"], route)
        depot = _depot_buses(observation)
        if queue >= 40:
            if share >= 0.7:
                for bus in depot:
                    choice = action_id("SHORT_TURN", bus, route)
                    if mask[choice]:
                        return choice
            for bus in depot:
                choice = action_id("DISPATCH", bus, route)
                if mask[choice]:
                    return choice
            donors = []
            for bus in range(16):
                choice = action_id("REASSIGN", bus, route)
                if not mask[choice]:
                    continue
                donor_route = _bus_route(observation, bus)
                urgency = _urgency(observation, donor_route) if donor_route is not None else 0.0
                donors.append((urgency, bus, choice))
            if donors:
                return min(donors)[2]
            if share >= 0.7:
                for bus in range(16):
                    choice = action_id("SHORT_TURN", bus, route)
                    if mask[choice]:
                        return choice
        headway = 360 if queue >= 80 else 600 if queue >= 40 else 900
        choice = action_id("SET_HEADWAY", route_id=route, headway_s=headway)
        if mask[choice]:
            return choice
        if queue < 5:
            for bus in range(16):
                choice = action_id("RECALL", bus)
                if mask[choice]:
                    return choice
        return 0
