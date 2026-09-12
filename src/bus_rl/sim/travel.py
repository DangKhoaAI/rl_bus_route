"""Travel duration is sampled exactly once, when an edge is entered."""

from __future__ import annotations

import math

from bus_rl.domain import Scenario


def edge_duration_s(scenario: Scenario, route_id: int, from_index: int, entry_time_s: int) -> int:
    config = scenario.config
    edge_index = route_id * (config.stops_per_route - 1) + min(
        from_index, config.stops_per_route - 2
    )
    bucket = min(entry_time_s // config.traffic_bucket_s, scenario.traffic_tape.shape[1] - 1)
    seconds = scenario.network.edge_base_s[edge_index] * float(
        scenario.traffic_tape[edge_index, bucket]
    )
    return max(config.tick_s, math.ceil(seconds / config.tick_s) * config.tick_s)
