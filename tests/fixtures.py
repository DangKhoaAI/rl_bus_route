from __future__ import annotations

import numpy as np

from bus_rl.data.scenario import generate_scenario
from bus_rl.domain import PassengerCohort, Pattern, SimConfig, initial_state, scenario_digest

_STAGE_FLAGS = {
    "M1": (False, False),
    "M2": (True, False),
    "M3": (True, True),
}


def empty_scenario(stage: str = "M1"):
    scenario = generate_scenario(77, SimConfig(horizon_s=14_400, demand_end_s=10_800))
    arrivals = np.zeros_like(scenario.arrival_tape)
    arrivals.setflags(write=False)
    digest = scenario_digest(
        scenario.config,
        scenario.network,
        scenario.fleet,
        arrivals,
        scenario.traffic_tape,
        scenario.seed,
    )
    enable_reassign, enable_short_turn = _STAGE_FLAGS[stage]
    return scenario.__class__(
        scenario.config,
        scenario.network,
        scenario.fleet,
        arrivals,
        scenario.traffic_tape,
        scenario.seed,
        digest,
        enable_reassign,
        enable_short_turn,
    )


def waiting_state(n: int, destination: int = 5, stage: str = "M1"):
    state = initial_state(empty_scenario(stage))
    state.cohorts.append(PassengerCohort(0, 0, 0, 1, 0, destination, 0, n))
    state.next_cohort_id = 1
    state.generated_total = n
    return state


def short_state():
    state = waiting_state(1, destination=5, stage="M3")
    bus = state.vehicles[0]
    bus.pattern = Pattern.SHORT
    bus.turn_stop = 3
    return state
