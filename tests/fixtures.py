from __future__ import annotations

import numpy as np

from bus_rl.data.scenario import generate_scenario
from bus_rl.domain import PassengerCohort, SimConfig, initial_state, scenario_digest


def empty_scenario():
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
    return scenario.__class__(
        scenario.config,
        scenario.network,
        scenario.fleet,
        arrivals,
        scenario.traffic_tape,
        scenario.seed,
        digest,
    )


def waiting_state(n: int, destination: int = 5):
    state = initial_state(empty_scenario())
    state.cohorts.append(PassengerCohort(0, 0, 0, 1, 0, destination, 0, n))
    state.next_cohort_id = 1
    state.generated_total = n
    return state
