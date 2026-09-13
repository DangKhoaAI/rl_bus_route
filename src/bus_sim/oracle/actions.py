"""The fixed 221-slot action table defined by schema v2."""

from __future__ import annotations

from bus_sim.oracle.domain import Action

F_MAX, R_MAX = 16, 4
HEADWAYS = (360, 600, 900)


def build_action_table() -> tuple[Action, ...]:
    actions = [Action()]
    actions += [Action("DISPATCH", b, r) for b in range(F_MAX) for r in range(R_MAX)]
    actions += [Action("REASSIGN", b, r) for b in range(F_MAX) for r in range(R_MAX)]
    actions += [Action("SHORT_TURN", b, r) for b in range(F_MAX) for r in range(R_MAX)]
    actions += [Action("RECALL", b) for b in range(F_MAX)]
    actions += [
        Action("SET_HEADWAY", route_id=r, headway_s=h) for r in range(R_MAX) for h in HEADWAYS
    ]
    assert len(actions) == 221
    return tuple(actions)


ACTION_TABLE = build_action_table()


def action_id(
    kind: str, bus_id: int | None = None, route_id: int | None = None, headway_s: int | None = None
) -> int:
    wanted = Action(kind, bus_id, route_id, headway_s)
    try:
        return ACTION_TABLE.index(wanted)
    except ValueError as error:
        raise ValueError(f"unknown action: {wanted}") from error
