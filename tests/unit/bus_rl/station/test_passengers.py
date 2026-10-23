from __future__ import annotations

from bus_rl.station import passengers


def _row(route_id: int, direction: int, origin: int, arrival_tick: int, count: int, kind: int = 0):
    row = [0] * 14
    row[passengers.ROUTE_ID] = route_id
    row[passengers.DIRECTION] = direction
    row[passengers.ORIGIN_INDEX] = origin
    row[passengers.ARRIVAL_TICK] = arrival_tick
    row[passengers.COUNT] = count
    row[passengers.STATUS] = kind
    return row


def _cells(rows, kinds, route_count=2, stops_per_route=6):
    return passengers.queues(
        rows,
        kinds,
        time_s=3600,
        tick_s=30,
        route_count=route_count,
        stops_per_route=stops_per_route,
    )


def test_queues_cover_every_cell_and_split_by_route_direction_and_stop():
    rows = [
        _row(0, 1, 2, arrival_tick=100, count=4),
        _row(0, 1, 2, arrival_tick=110, count=6),
        _row(0, -1, 2, arrival_tick=110, count=5),
        _row(1, 1, 0, arrival_tick=110, count=3),
    ]
    cells = _cells(rows, [passengers.WAITING] * 4)
    assert len(cells) == 2 * 2 * 6
    by_key = {(cell.route_id, cell.direction, cell.stop_index): cell for cell in cells}
    same_cell = by_key[(0, 1, 2)]
    assert (same_cell.waiting, same_cell.max_wait_s) == (10, 600)
    assert same_cell.mean_wait_s == round((4 * 600 + 6 * 300) / 10, 1)
    assert by_key[(0, -1, 2)].waiting == 5
    assert by_key[(1, 1, 0)].waiting == 3
    assert (by_key[(0, 1, 0)].waiting, by_key[(0, 1, 0)].mean_wait_s) == (0, 0.0)


def test_queues_flag_excessive_wait_and_skip_cohorts_that_are_not_waiting():
    rows = [
        _row(0, 1, 3, arrival_tick=0, count=7),
        _row(0, 1, 3, arrival_tick=120, count=2),
        _row(0, 1, 3, arrival_tick=0, count=99, kind=1),
    ]
    cells = _cells(rows, [passengers.WAITING, passengers.WAITING, 1], route_count=1)
    cell = next(cell for cell in cells if (cell.direction, cell.stop_index) == (1, 3))
    assert cell.waiting == 9
    assert cell.excessive == 7
