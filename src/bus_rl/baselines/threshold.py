from __future__ import annotations

import math

from bus_rl.control.actions import action_id


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


class ThresholdController:
    def act(self, observation, mask):
        queues = observation["stops"][:, :, :, 0].sum(axis=(1, 2)) * 40
        route = int(queues.argmax())
        if queues[route] >= 40:
            share = short_eligible_share(observation["stops"], route)
            depot = [bus for bus in range(16) if observation["vehicles"][bus, 0] == 1]
            if share >= 0.7:
                for bus in depot:
                    choice = action_id("SHORT_TURN", bus, route)
                    if mask[choice]:
                        return choice
            for bus in depot:
                choice = action_id("DISPATCH", bus, route)
                if mask[choice]:
                    return choice
            if share >= 0.7:
                for bus in range(16):
                    choice = action_id("SHORT_TURN", bus, route)
                    if mask[choice]:
                        return choice
        return 0
