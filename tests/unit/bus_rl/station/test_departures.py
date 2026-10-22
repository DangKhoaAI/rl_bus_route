from __future__ import annotations

from bus_rl.station import departures


def test_clocks_report_the_gap_since_the_last_full_departure():
    clocks = departures.clocks([[0, 1, 0], [0, -1, 5]], [600, 3000], time_s=3600)
    assert [(clock.route_id, clock.direction, clock.node) for clock in clocks] == [
        (0, 1, 0),
        (0, -1, 5),
    ]
    assert [clock.full_gap_s for clock in clocks] == [3000, 600]
    assert [clock.violates_guard for clock in clocks] == [True, False]


def test_clocks_report_unknown_gaps_for_streams_that_never_departed():
    # The kernel seeds the terminals at -900s and leaves other streams at a large
    # negative sentinel; neither is a departure time, so neither is a violation.
    clocks = departures.clocks([[0, 1, 0], [0, -1, 3]], [-900, -1_000_000_000], time_s=4800)
    assert [clock.last_full_s for clock in clocks] == [-900, -1_000_000_000]
    assert [clock.full_gap_s for clock in clocks] == [None, None]
    assert [clock.violates_guard for clock in clocks] == [False, False]
