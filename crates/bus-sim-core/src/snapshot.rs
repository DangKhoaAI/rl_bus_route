//! Backend-agnostic deterministic snapshots mirroring
//! `bus_sim.parity.snapshot.numpy_state_snapshot`.

use crate::domain::{PassengerCohort, PassengerStatus, Pattern, Phase, Vehicle, WorldState};

pub const COHORT_KIND_WAITING: i64 = 0;
pub const COHORT_KIND_ONBOARD: i64 = 1;
pub const COHORT_KIND_FINISHED: i64 = 2;

pub struct StateSnapshot {
    pub time_s: i64,
    pub counters: [i64; 6],
    pub vehicles: Vec<[i64; 14]>,
    pub cohorts: Vec<[i64; 14]>,
    pub cohort_kind: Vec<i64>,
    pub headway_targets: Vec<i64>,
    pub headway_changed_at: Vec<i64>,
    pub departure_keys: Vec<[i64; 3]>,
    pub last_departure: Vec<i64>,
    pub last_full_departure: Vec<i64>,
    pub terminal_settled: i64,
}

fn optional(value: Option<i64>) -> i64 {
    value.unwrap_or(-1)
}

pub fn vehicle_row(vehicle: &Vehicle) -> [i64; 14] {
    [
        vehicle.vehicle_id as i64,
        vehicle.capacity,
        vehicle.route_id.map(|r| r as i64).unwrap_or(-1),
        vehicle.node as i64,
        vehicle.direction as i64,
        vehicle.phase.as_i64(),
        vehicle.remaining_s,
        vehicle.next_node.map(|n| n as i64).unwrap_or(-1),
        vehicle.visit_id,
        vehicle.cooldown_until_s,
        vehicle.pattern.as_i64(),
        vehicle.turn_stop.map(|t| t as i64).unwrap_or(-1),
        vehicle.pending_extra as i64,
        vehicle.load,
    ]
}

pub fn cohort_row(cohort: &PassengerCohort) -> [i64; 13] {
    [
        cohort.cohort_id,
        cohort.lineage_id,
        cohort.route_id as i64,
        cohort.direction as i64,
        cohort.origin_index as i64,
        cohort.destination_index as i64,
        cohort.arrival_tick,
        cohort.count,
        cohort.status.as_i64(),
        cohort.first_denied as i64,
        optional(cohort.boarding_tick),
        optional(cohort.completion_tick),
        optional(cohort.abandonment_tick),
    ]
}

fn prefixed(bus_id: i64, row: [i64; 13]) -> [i64; 14] {
    [
        bus_id, row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9],
        row[10], row[11], row[12],
    ]
}

pub fn full_snapshot(state: &WorldState) -> StateSnapshot {
    let mut vehicles: Vec<[i64; 14]> = state.vehicles.iter().map(vehicle_row).collect();
    vehicles.sort_by_key(|row| row[0]);

    let mut cohorts: Vec<[i64; 14]> = Vec::new();
    let mut cohort_kind: Vec<i64> = Vec::new();
    for cohort in &state.cohorts {
        cohorts.push(prefixed(-1, cohort_row(cohort)));
        cohort_kind.push(COHORT_KIND_WAITING);
    }
    for vehicle in &state.vehicles {
        for cohort in &vehicle.passengers {
            cohorts.push(prefixed(vehicle.vehicle_id as i64, cohort_row(cohort)));
            cohort_kind.push(COHORT_KIND_ONBOARD);
        }
    }
    for cohort in &state.finished {
        cohorts.push(prefixed(-1, cohort_row(cohort)));
        cohort_kind.push(COHORT_KIND_FINISHED);
    }

    let mut keys: Vec<(u32, i32, u32)> = state.last_departure_s.keys().copied().collect();
    for key in state.last_full_departure_s.keys() {
        if !keys.contains(key) {
            keys.push(*key);
        }
    }
    keys.sort_unstable();

    let departure_keys: Vec<[i64; 3]> = keys
        .iter()
        .map(|(r, d, n)| [*r as i64, *d as i64, *n as i64])
        .collect();
    let last_departure = keys
        .iter()
        .map(|key| *state.last_departure_s.get(key).unwrap_or(&-1_000_000_000))
        .collect();
    let last_full_departure = keys
        .iter()
        .map(|key| {
            *state
                .last_full_departure_s
                .get(key)
                .unwrap_or(&-1_000_000_000)
        })
        .collect();

    StateSnapshot {
        time_s: state.current_time_s,
        counters: [
            state.generated_count(),
            state.waiting_count(),
            state.onboard_count(),
            state.completed_count(),
            state.abandoned_count(),
            state.depot_count(),
        ],
        vehicles,
        cohorts,
        cohort_kind,
        headway_targets: state.headway_targets_s.clone(),
        headway_changed_at: state.headway_changed_at_s.clone(),
        departure_keys,
        last_departure,
        last_full_departure,
        terminal_settled: state.terminal_settled as i64,
    }
}

/// Lightweight per-tick snapshot (vehicles, counters, clock, load) without the
/// full cohort archive, used by the tick-level parity records.
pub struct TickState {
    pub time_s: i64,
    pub counters: [i64; 6],
    pub vehicles: Vec<[i64; 14]>,
}

pub fn tick_snapshot(state: &WorldState) -> TickState {
    TickState {
        time_s: state.current_time_s,
        counters: [
            state.generated_count(),
            state.waiting_count(),
            state.onboard_count(),
            state.completed_count(),
            state.abandoned_count(),
            state.depot_count(),
        ],
        vehicles: state.vehicles.iter().map(vehicle_row).collect(),
    }
}

#[allow(dead_code)]
pub fn pattern_of(value: i64) -> Option<Pattern> {
    match value {
        0 => Some(Pattern::Full),
        1 => Some(Pattern::Short),
        _ => None,
    }
}

#[allow(dead_code)]
pub fn phase_of(value: i64) -> Option<Phase> {
    match value {
        0 => Some(Phase::DepotIdle),
        1 => Some(Phase::Deadhead),
        2 => Some(Phase::TerminalIdle),
        3 => Some(Phase::ServiceMoving),
        4 => Some(Phase::ServiceDwell),
        5 => Some(Phase::Layover),
        _ => None,
    }
}

#[allow(dead_code)]
pub fn status_of(value: i64) -> Option<PassengerStatus> {
    match value {
        0 => Some(PassengerStatus::Waiting),
        1 => Some(PassengerStatus::Onboard),
        2 => Some(PassengerStatus::Completed),
        3 => Some(PassengerStatus::Abandoned),
        _ => None,
    }
}
