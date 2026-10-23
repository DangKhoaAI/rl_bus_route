from __future__ import annotations

from bus_rl.station import fleet
from bus_sim.oracle.domain import Phase


def _row(vehicle_id: int, route_id: int, phase: Phase, cooldown_until_s: int = 0, load: int = 0):
    row = [0] * 14
    row[fleet.VEHICLE_ID] = vehicle_id
    row[fleet.ROUTE_ID] = route_id
    row[fleet.PHASE] = int(phase)
    row[fleet.COOLDOWN_UNTIL_S] = cooldown_until_s
    row[fleet.LOAD] = load
    return row


def test_group_by_route_keeps_depot_buses_out():
    rows = [
        _row(0, 0, Phase.SERVICE_MOVING),
        _row(1, 2, Phase.DEADHEAD),
        _row(2, -1, Phase.DEPOT_IDLE),
        _row(3, 0, Phase.LAYOVER),
    ]
    routes = fleet.group_by_route(rows, route_count=3)
    assert [[row[fleet.VEHICLE_ID] for row in route] for route in routes] == [[0, 3], [], [1]]


def test_reserve_queue_puts_ready_idle_buses_first():
    rows = [
        _row(9, -1, Phase.DEPOT_IDLE, cooldown_until_s=600),
        _row(10, -1, Phase.DEADHEAD),
        _row(11, -1, Phase.DEPOT_IDLE, cooldown_until_s=0),
        _row(4, 1, Phase.SERVICE_DWELL),
        _row(8, -1, Phase.DEPOT_IDLE, cooldown_until_s=0),
    ]
    assert [row[fleet.VEHICLE_ID] for row in fleet.reserve_queue(rows)] == [8, 11, 9, 10]


def test_display_status_flags_crowding_above_comfort():
    assert fleet.display_status(Phase.SERVICE_MOVING, load=31, comfort_capacity=30) == "crowded"
    assert fleet.display_status(Phase.SERVICE_MOVING, load=30, comfort_capacity=30) == "moving"
    assert fleet.display_status(Phase.LAYOVER, load=0, comfort_capacity=30) == "terminal"
    assert fleet.display_status(Phase.DEPOT_IDLE, load=0, comfort_capacity=30) == "depot"
