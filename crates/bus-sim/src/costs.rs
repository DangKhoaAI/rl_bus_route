//! Cost integration and weighted reward. Port of `src/bus_rl/rewards/costs.py`.
//!
//! Literal oracle constants: comfort capacity 30 and excessive-wait 900s
//! (discrepancy D5); waiting age uses `arrival_tick * 30` (D4).

use crate::domain::{PassengerStatus, Phase, StepCosts, WorldState};

pub const COMFORT_CAPACITY: i64 = 30;
pub const EXCESSIVE_WAIT_S: i64 = 900;
pub const TICK_STAMP_S: i64 = 30;

#[derive(Clone, Copy, Debug)]
pub struct RewardConfig {
    pub waiting: f64,
    pub onboard: f64,
    pub crowding: f64,
    pub active: f64,
    pub deadhead: f64,
    pub fairness: f64,
    pub first_denied: f64,
    pub abandoned: f64,
    pub mission: f64,
    pub unfinished: f64,
    pub n_ref: f64,
}

impl Default for RewardConfig {
    fn default() -> Self {
        Self {
            waiting: 1.0,
            onboard: 0.25,
            crowding: 0.5,
            active: 0.5,
            deadhead: 0.5,
            fairness: 1.0,
            first_denied: 5.0,
            abandoned: 60.0,
            mission: 2.0,
            unfinished: 60.0,
            n_ref: 3_000.0,
        }
    }
}

pub fn integrate_tick_costs(state: &WorldState, duration_s: i64) -> StepCosts {
    let minutes = duration_s as f64 / 60.0;
    let waiting = state.waiting_count() as f64 * minutes;
    let onboard = state.onboard_count() as f64 * minutes;
    let crowding = state
        .vehicles
        .iter()
        .map(|vehicle| (vehicle.load - COMFORT_CAPACITY).max(0) as f64)
        .sum::<f64>()
        * minutes;
    let active = state
        .vehicles
        .iter()
        .filter(|vehicle| vehicle.phase != Phase::DepotIdle)
        .count() as f64
        * minutes;
    let deadhead = state
        .vehicles
        .iter()
        .filter(|vehicle| vehicle.phase == Phase::Deadhead)
        .count() as f64
        * minutes;
    let excessive = state
        .cohorts
        .iter()
        .filter(|cohort| {
            state.current_time_s - cohort.arrival_tick * TICK_STAMP_S >= EXCESSIVE_WAIT_S
        })
        .map(|cohort| cohort.count as f64)
        .sum::<f64>()
        * minutes;
    StepCosts {
        waiting_pm: waiting,
        onboard_pm: onboard,
        crowding_pm: crowding,
        active_bus_min: active,
        deadhead_bus_min: deadhead,
        excessive_wait_pm: excessive,
        ..StepCosts::default()
    }
}

pub fn interval_cost(costs: &StepCosts, reward: &RewardConfig) -> f64 {
    reward.waiting * costs.waiting_pm
        + reward.onboard * costs.onboard_pm
        + reward.crowding * costs.crowding_pm
        + reward.active * costs.active_bus_min
        + reward.deadhead * costs.deadhead_bus_min
        + reward.fairness * costs.excessive_wait_pm
        + reward.first_denied * costs.first_denied_count as f64
        + reward.abandoned * costs.abandoned_count as f64
        + reward.mission * costs.mission_changes as f64
        + reward.unfinished * costs.terminal_unfinished_count as f64
}

/// Mean observed wait; `None` when nobody was generated.
pub fn mean_waiting_minutes(state: &WorldState) -> Option<f64> {
    if state.generated_count() == 0 {
        return None;
    }
    let mut total = 0.0f64;
    let mut add = |cohort: &crate::domain::PassengerCohort| {
        let end_s = match (cohort.status, cohort.boarding_tick, cohort.abandonment_tick) {
            (PassengerStatus::Completed, Some(tick), _) => tick * TICK_STAMP_S,
            (PassengerStatus::Abandoned, _, Some(tick)) => tick * TICK_STAMP_S,
            _ => state.current_time_s,
        };
        total += cohort.count as f64 * (end_s - cohort.arrival_tick * TICK_STAMP_S) as f64 / 60.0;
    };
    for cohort in &state.cohorts {
        add(cohort);
    }
    for vehicle in &state.vehicles {
        for cohort in &vehicle.passengers {
            add(cohort);
        }
    }
    for cohort in &state.finished {
        add(cohort);
    }
    Some(total / state.generated_count() as f64)
}
