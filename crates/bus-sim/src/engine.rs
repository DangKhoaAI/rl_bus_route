//! Deterministic fixed-tick engine. Port of `src/bus_rl/sim/engine.py`.
//!
//! Ordering is the oracle contract: the action is applied before the tick loop;
//! each tick is complete-phase -> arrivals -> abandon -> terminal boarding ->
//! dispatch -> integrate costs -> decay timers -> advance clock -> conservation
//! -> event log -> one-time terminal settlement.

use crate::costs::integrate_tick_costs;
use crate::dispatcher::{apply_action, dispatch_ready};
use crate::domain::{
    add_costs, EventLogEntry, PassengerCohort, PassengerStatus, Phase, Scenario, StepCosts,
    WorldState,
};
use crate::error::KernelError;
use crate::passengers::{abandon_expired, board_visit};
use crate::snapshot::{tick_snapshot, TickState};
use crate::vehicles::complete_expired_phase;

fn add_arrivals(state: &mut WorldState, scenario: &Scenario) {
    let config = &scenario.config;
    if state.current_time_s >= config.demand_end_s {
        return;
    }
    let tick = state.current_time_s / config.tick_s;
    for route in 0..config.route_count {
        for d_index in 0..2usize {
            let direction = if d_index == 0 { 1 } else { -1 };
            for origin in 0..config.stops_per_route {
                for destination in 0..config.stops_per_route {
                    let count =
                        scenario.arrival(tick as usize, route, d_index, origin, destination) as i64;
                    if count == 0 {
                        continue;
                    }
                    let cohort_id = state.next_cohort_id;
                    state.cohorts.push(PassengerCohort {
                        cohort_id,
                        lineage_id: cohort_id,
                        route_id: route as u32,
                        direction,
                        origin_index: origin as u32,
                        destination_index: destination as u32,
                        arrival_tick: tick,
                        count,
                        status: PassengerStatus::Waiting,
                        first_denied: false,
                        boarding_tick: None,
                        completion_tick: None,
                        abandonment_tick: None,
                    });
                    state.next_cohort_id += 1;
                    state.generated_total += count;
                    state.waiting_total += count;
                }
            }
        }
    }
}

pub fn advance_tick(state: &mut WorldState, scenario: &Scenario) -> Result<StepCosts, KernelError> {
    let config = &scenario.config;
    if state.current_time_s >= config.horizon_s {
        return Ok(StepCosts::default());
    }
    let mut boarded = 0i64;
    let mut denied = 0i64;
    for bus_id in 0..state.vehicles.len() {
        if state.vehicles[bus_id].remaining_s == 0 {
            let (b, d) = complete_expired_phase(state, scenario, bus_id);
            boarded += b;
            denied += d;
        }
    }
    add_arrivals(state, scenario);
    let abandoned = abandon_expired(state, config.patience_s, config.tick_s);
    for bus_id in 0..state.vehicles.len() {
        let (phase, route_id, node) = {
            let bus = &state.vehicles[bus_id];
            (bus.phase, bus.route_id, bus.node)
        };
        if phase == Phase::TerminalIdle {
            if let Some(route_id) = route_id {
                let stop_index = scenario
                    .route(route_id)
                    .stops
                    .iter()
                    .position(|stop| *stop == node)
                    .expect("terminal node on route");
                let direction = state.vehicles[bus_id].direction;
                let event = board_visit(state, bus_id, route_id, direction, stop_index as u32);
                boarded += event.boarded_count;
                denied += event.first_denied_count;
            }
        }
    }
    dispatch_ready(state, scenario);
    let mut costs = integrate_tick_costs(state, config.tick_s);
    for vehicle in &mut state.vehicles {
        if vehicle.remaining_s > 0 {
            vehicle.remaining_s = (vehicle.remaining_s - config.tick_s).max(0);
        }
    }
    state.current_time_s += config.tick_s;
    if state.conservation_checks {
        state.assert_conservation()?;
    }
    state.event_log.push(EventLogEntry {
        time_s: state.current_time_s,
        boarded,
        denied,
        abandoned,
    });
    costs.first_denied_count = denied;
    costs.abandoned_count = abandoned;
    if state.current_time_s == config.horizon_s && !state.terminal_settled {
        costs.terminal_unfinished_count = state.waiting_total + state.onboard_total;
        state.terminal_settled = true;
    }
    Ok(costs)
}

pub fn advance_interval(
    state: &mut WorldState,
    scenario: &Scenario,
    action_index: usize,
) -> Result<StepCosts, KernelError> {
    Ok(advance_interval_traced(state, scenario, action_index)?.costs)
}

pub struct TickTrace {
    pub state: TickState,
    pub costs: StepCosts,
    pub event: EventLogEntry,
}

pub struct IntervalTrace {
    pub ticks: Vec<TickTrace>,
    pub costs: StepCosts,
}

/// As [`advance_interval`], but captures the world after every tick for parity
/// tests. The action is applied before the first tick, matching the oracle.
pub fn advance_interval_traced(
    state: &mut WorldState,
    scenario: &Scenario,
    action_index: usize,
) -> Result<IntervalTrace, KernelError> {
    let missions = apply_action(state, scenario, action_index)?;
    let mut result = StepCosts::default();
    let ticks = scenario.config.control_interval_s / scenario.config.tick_s;
    let mut traces = Vec::with_capacity(ticks as usize);
    for _ in 0..ticks {
        let event_cursor = state.event_log.len();
        let tick_costs = advance_tick(state, scenario)?;
        result = add_costs(&result, &tick_costs);
        traces.push(TickTrace {
            state: tick_snapshot(state),
            costs: tick_costs,
            event: state.event_log[event_cursor],
        });
    }
    result.mission_changes += missions;
    Ok(IntervalTrace {
        ticks: traces,
        costs: result,
    })
}
