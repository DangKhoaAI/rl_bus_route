from __future__ import annotations

from bus_rl.control.actions import HEADWAYS, action_id
from bus_rl.env.observation import ROUTE_OFFSET

NOMINAL_CYCLE_S = 2280
FLOOR = 2


def desired_fleet(rates, total_queue: float, route_count: int = 3) -> list[int]:
    budget = 12 if total_queue >= 120 else 9
    remainder = budget - FLOOR * route_count
    weights = [float(rate) for rate in rates]
    if sum(weights) <= 0:
        weights = [1.0] * route_count
    raw = [remainder * weight / sum(weights) for weight in weights]
    base = [int(value) for value in raw]
    leftover = remainder - sum(base)
    order = sorted(range(route_count), key=lambda index: (-(raw[index] - base[index]), index))
    for index in order[:leftover]:
        base[index] += 1
    return [FLOOR + extra for extra in base]


def nearest_headway(desired: int) -> int:
    target = NOMINAL_CYCLE_S / max(1, desired)
    return min(HEADWAYS, key=lambda headway: (abs(headway - target), headway))


def _bus_route(observation, bus: int) -> int | None:
    row = observation["vehicles"][bus]
    if row[ROUTE_OFFSET] == 1:
        return None
    for route in range(4):
        if row[ROUTE_OFFSET + 1 + route] == 1:
            return route
    return None


class ProportionalController:
    def act(self, observation, mask):
        route_count = int(observation["route_valid"].sum()) or 3
        queues = observation["stops"][:route_count, :, :, 0].sum(axis=(1, 2)) * 40
        rates = observation["arrival_history"][:route_count].sum(axis=(1, 2, 3))
        desired = desired_fleet(rates, float(queues.sum()), route_count)
        committed = [
            round(float(observation["routes"][route, 3]) * 16) for route in range(route_count)
        ]
        shortage = sorted(
            range(route_count),
            key=lambda route: (committed[route] - desired[route], route),
        )
        surplus = sorted(
            range(route_count),
            key=lambda route: (desired[route] - committed[route], route),
        )
        for route in shortage:
            if committed[route] >= desired[route]:
                continue
            for bus in range(16):
                choice = action_id("DISPATCH", bus, route)
                if mask[choice]:
                    return choice
        for receiver in shortage:
            if committed[receiver] >= desired[receiver]:
                continue
            for donor_route in surplus:
                if committed[donor_route] <= desired[donor_route] or donor_route == receiver:
                    continue
                for bus in range(16):
                    if _bus_route(observation, bus) != donor_route:
                        continue
                    choice = action_id("REASSIGN", bus, receiver)
                    if mask[choice]:
                        return choice
        for route in surplus:
            if committed[route] <= desired[route]:
                continue
            for bus in range(16):
                if _bus_route(observation, bus) != route:
                    continue
                choice = action_id("RECALL", bus)
                if mask[choice]:
                    return choice
        for route, count in enumerate(desired):
            choice = action_id("SET_HEADWAY", route_id=route, headway_s=nearest_headway(count))
            if mask[choice]:
                return choice
        return 0
