"""Synthetic, action-independent demand and traffic scenario generation."""

from __future__ import annotations

import math

import numpy as np

from bus_rl.domain import Scenario, SimConfig, VehicleSpec, generate_base_network, scenario_digest


def _fleet(config: SimConfig, network_depot: int) -> tuple[VehicleSpec, ...]:
    assigned = config.fleet_size - 3
    if assigned < config.route_count * 2:
        raise ValueError("fleet is too small to place two assigned vehicles per route")
    result: list[VehicleSpec] = []
    for vehicle_id in range(config.fleet_size):
        route_id = vehicle_id // 3 if vehicle_id < min(assigned, config.route_count * 3) else None
        node = route_id * config.stops_per_route if route_id is not None else network_depot
        result.append(VehicleSpec(vehicle_id, route_id, node, 1))
    return tuple(result)


def generate_scenario(seed: int, config: SimConfig | None = None) -> Scenario:
    config = config or SimConfig()
    network = generate_base_network(config)
    rng = np.random.default_rng(seed)
    ticks = config.horizon_s // config.tick_s
    # [tick, route, direction-index, origin, destination], int32 counts.
    arrivals = np.zeros(
        (ticks, config.route_count, 2, config.stops_per_route, config.stops_per_route),
        dtype=np.int32,
    )
    hourly = (180, 160, 140)
    peak_route = int(rng.integers(config.route_count))
    peak_center = int(rng.integers(45 * 60, 135 * 60))
    peak_width = int(rng.integers(15 * 60, 30 * 60 + 1))
    peak_multiplier = float(rng.uniform(1.5, 3.0))
    for tick in range(config.demand_end_s // config.tick_s):
        time_s = tick * config.tick_s
        for route_id in range(config.route_count):
            rate = hourly[route_id] / 60.0
            if route_id == peak_route:
                rate *= 1 + (peak_multiplier - 1) * math.exp(
                    -0.5 * ((time_s - peak_center) / peak_width) ** 2
                )
            for direction_index, direction in enumerate((1, -1)):
                origins = (
                    range(config.stops_per_route - 1)
                    if direction == 1
                    else range(1, config.stops_per_route)
                )
                for origin in origins:
                    total = int(
                        rng.poisson(rate * config.tick_s / 60 / (2 * (config.stops_per_route - 1)))
                    )
                    if not total:
                        continue
                    destinations = (
                        np.arange(origin + 1, config.stops_per_route)
                        if direction == 1
                        else np.arange(0, origin)
                    )
                    weights = np.exp(-np.abs(destinations - origin) / 2)
                    destination = int(rng.choice(destinations, p=weights / weights.sum()))
                    arrivals[tick, route_id, direction_index, origin, destination] = total
    buckets = math.ceil(config.horizon_s / config.traffic_bucket_s)
    edge_count = config.route_count * (config.stops_per_route - 1)
    traffic = np.clip(
        rng.lognormal(-0.5 * 0.15**2, 0.15, size=(edge_count, buckets)), 0.7, 2.5
    ).astype(np.float32)
    traffic.setflags(write=False)
    arrivals.setflags(write=False)
    fleet = _fleet(config, network.depot_node)
    return Scenario(
        config,
        network,
        fleet,
        arrivals,
        traffic,
        seed,
        scenario_digest(config, network, fleet, arrivals, traffic, seed),
    )


SPLIT_SEEDS = {
    "train": 1001,
    "validation": 2001,
    "test_id": 3001,
    "test_ood_burst": 4001,
    "test_ood_traffic": 5001,
}


def generate_manifest(
    split: str, count: int, config: SimConfig | None = None
) -> tuple[Scenario, ...]:
    """Generate a deterministic split with unique child seeds and shared topology."""
    if split not in SPLIT_SEEDS:
        raise ValueError(f"unknown split: {split}")
    if count <= 0:
        raise ValueError("manifest count must be positive")
    seed_sequence = np.random.SeedSequence(SPLIT_SEEDS[split]).spawn(count)
    seeds = [int(child.generate_state(1, dtype=np.uint64)[0]) for child in seed_sequence]
    if len(seeds) != len(set(seeds)):
        raise AssertionError("child seed collision")
    return tuple(generate_scenario(seed, config) for seed in seeds)
