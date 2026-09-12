//! FIFO passenger boarding, alighting, and abandonment.
//!
//! Port of `src/bus_rl/sim/passengers.py`. Tick stamps use the literal `30`
//! exactly like the oracle (discrepancy D4); only `tick_s=30` is supported.

use crate::domain::{PassengerCohort, PassengerStatus, Pattern, WorldState};

/// Boarding/completion/abandonment tick stamp divisor (oracle discrepancy D4).
const TICK_STAMP_S: i64 = 30;

#[derive(Clone, Copy, Debug, Default)]
pub struct PassengerEvents {
    pub boarded_count: i64,
    pub first_denied_count: i64,
}

fn eligible(
    cohort: &PassengerCohort,
    pattern: Pattern,
    turn_stop: u32,
    direction: i32,
    stop_index: u32,
) -> bool {
    if cohort.status != PassengerStatus::Waiting
        || cohort.direction != direction
        || cohort.origin_index != stop_index
    {
        return false;
    }
    if pattern == Pattern::Full {
        return true;
    }
    if direction == 1 {
        cohort.destination_index <= turn_stop
    } else {
        stop_index <= turn_stop
    }
}

fn split_for_boarding(state: &mut WorldState, index: usize, count: i64, bus_id: usize) {
    let source = state.cohorts[index];
    let boarded = PassengerCohort {
        cohort_id: state.next_cohort_id,
        lineage_id: source.lineage_id,
        route_id: source.route_id,
        direction: source.direction,
        origin_index: source.origin_index,
        destination_index: source.destination_index,
        arrival_tick: source.arrival_tick,
        count,
        status: PassengerStatus::Onboard,
        first_denied: source.first_denied,
        boarding_tick: Some(state.current_time_s / TICK_STAMP_S),
        completion_tick: None,
        abandonment_tick: None,
    };
    state.next_cohort_id += 1;
    state.cohorts[index].count -= count;
    state.waiting_total -= count;
    state.onboard_total += count;
    let bus = &mut state.vehicles[bus_id];
    bus.passengers.push(boarded);
    bus.load += count;
}

/// Board FIFO eligible cohorts at a real visit; never moves the vehicle itself.
pub fn board_visit(
    state: &mut WorldState,
    bus_id: usize,
    route_id: u32,
    direction: i32,
    stop_index: u32,
) -> PassengerEvents {
    let (pattern, turn_stop, capacity, load, bus_route) = {
        let bus = &state.vehicles[bus_id];
        (
            bus.pattern,
            bus.turn_stop.unwrap_or(3),
            bus.capacity,
            bus.load,
            bus.route_id,
        )
    };
    assert_eq!(
        bus_route,
        Some(route_id),
        "bus route does not match visit route"
    );
    assert!(load <= capacity, "bus is already over capacity");

    let mut candidates: Vec<(i64, i64, usize)> = state
        .cohorts
        .iter()
        .enumerate()
        .filter(|(_, cohort)| eligible(cohort, pattern, turn_stop, direction, stop_index))
        .filter(|(_, cohort)| cohort.route_id == route_id)
        .map(|(index, cohort)| (cohort.arrival_tick, cohort.cohort_id, index))
        .collect();
    candidates.sort_unstable();

    let mut boarded = 0i64;
    let mut denied = 0i64;
    for (_, _, index) in candidates {
        let available = capacity - state.vehicles[bus_id].load;
        let count = state.cohorts[index].count;
        let take = available.min(count);
        if take > 0 {
            split_for_boarding(state, index, take, bus_id);
            boarded += take;
        }
        if state.cohorts[index].count > 0 && !state.cohorts[index].first_denied {
            state.cohorts[index].first_denied = true;
            denied += state.cohorts[index].count;
        }
    }
    state.cohorts.retain(|cohort| cohort.count > 0);
    PassengerEvents {
        boarded_count: boarded,
        first_denied_count: denied,
    }
}

pub fn alight_visit(state: &mut WorldState, bus_id: usize, stop_index: u32) -> i64 {
    let mut alighted = 0i64;
    let mut remaining: Vec<PassengerCohort> = Vec::new();
    let passengers = std::mem::take(&mut state.vehicles[bus_id].passengers);
    for mut cohort in passengers {
        if cohort.destination_index == stop_index && cohort.status == PassengerStatus::Onboard {
            cohort.status = PassengerStatus::Completed;
            cohort.completion_tick = Some(state.current_time_s / TICK_STAMP_S);
            alighted += cohort.count;
            state.onboard_total -= cohort.count;
            state.completed_total += cohort.count;
            state.finished.push(cohort);
            state.push_recent_finished(cohort);
        } else {
            remaining.push(cohort);
        }
    }
    state.vehicles[bus_id].passengers = remaining;
    state.vehicles[bus_id].load -= alighted;
    alighted
}

pub fn abandon_expired(state: &mut WorldState, patience_s: i64, tick_s: i64) -> i64 {
    let mut abandoned = 0i64;
    let mut remaining: Vec<PassengerCohort> = Vec::with_capacity(state.cohorts.len());
    let cohorts = std::mem::take(&mut state.cohorts);
    for mut cohort in cohorts {
        if state.current_time_s - cohort.arrival_tick * tick_s >= patience_s {
            cohort.status = PassengerStatus::Abandoned;
            cohort.abandonment_tick = Some(state.current_time_s / TICK_STAMP_S);
            abandoned += cohort.count;
            state.waiting_total -= cohort.count;
            state.abandoned_total += cohort.count;
            state.finished.push(cohort);
            state.push_recent_finished(cohort);
        } else {
            remaining.push(cohort);
        }
    }
    state.cohorts = remaining;
    abandoned
}
