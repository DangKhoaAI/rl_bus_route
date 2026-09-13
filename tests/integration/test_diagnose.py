from bus_rl.execution.timing import TIMERS


def test_timers_are_noop_until_enabled():
    TIMERS.reset()
    TIMERS.enabled = False
    with TIMERS.span("engine.board"):
        pass
    assert TIMERS.snapshot() == []
    TIMERS.enabled = True
    with TIMERS.span("engine.board"):
        pass
    rows = TIMERS.snapshot()
    TIMERS.enabled = False
    TIMERS.reset()
    assert rows[0]["name"] == "engine.board"
    assert rows[0]["calls"] == 1
