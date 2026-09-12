//! Incremental observation tensors, port of `bus_rl.env.observation.observe`.
//!
//! The oracle scans every cohort (waiting, onboard, then the full finished
//! archive). The native kernel reproduces the same values without scanning the
//! finished archive: finished cohorts that can still contribute are mirrored in
//! a bounded recent-finished ring (`WorldState::recent_finished`). Iteration
//! order is waiting -> onboard -> recent-finished, matching `iter_cohorts`, so
//! float32 accumulation order is preserved.

use crate::domain::{PassengerCohort, PassengerStatus, Pattern, Phase, Scenario, WorldState};

pub const MAX_ROUTES: usize = 4;
pub const MAX_DIRECTIONS: usize = 2;
pub const MAX_STOPS: usize = 8;
pub const STOP_CHANNELS: usize = 7;
pub const HISTORY_LAGS: usize = 5;
pub const MAX_VEHICLES: usize = 16;
pub const VEHICLE_FEATURES: usize = 27;
pub const ROUTE_FEATURES: usize = 8;

pub const ROUTE_OFFSET: usize = 6;
pub const PATTERN_OFFSET: usize = 11;
pub const DIRECTION_OFFSET: usize = 13;
pub const NODE_OFFSET: usize = 15;
pub const TARGET_OFFSET: usize = 17;
pub const SCALAR_OFFSET: usize = 22;

const QUEUE_SCALE: f64 = 40.0;
const AGE_SCALE: f64 = 2700.0;
const EXCESSIVE_WAIT_S: f64 = 900.0;
const HEADWAY_SCALE: f64 = 1200.0;
const LOAD_SCALE: f64 = 40.0;
const COOLDOWN_SCALE: f64 = 1200.0;
const HEADWAY_FLOOR_S: i64 = 600;
const FLEET_SCALE: f64 = 16.0;

/// Flat observation buffers in the frozen R0 shapes.
pub struct Observation {
    pub stops: Vec<f32>,           // [4, 2, 8, 7]
    pub arrival_history: Vec<f32>, // [4, 2, 8, 5]
    pub forecast: Vec<f32>,        // [4, 2, 8]
    pub vehicles: Vec<f32>,        // [16, 27]
    pub routes: Vec<f32>,          // [4, 8]
    pub stop_valid: Vec<f32>,      // [4, 2, 8]
    pub vehicle_valid: Vec<f32>,   // [16]
    pub route_valid: Vec<f32>,     // [4]
    pub context: Vec<f32>,         // [3]
}

impl Observation {
    fn zeros() -> Self {
        Self {
            stops: vec![0.0; MAX_ROUTES * MAX_DIRECTIONS * MAX_STOPS * STOP_CHANNELS],
            arrival_history: vec![0.0; MAX_ROUTES * MAX_DIRECTIONS * MAX_STOPS * HISTORY_LAGS],
            forecast: vec![0.0; MAX_ROUTES * MAX_DIRECTIONS * MAX_STOPS],
            vehicles: vec![0.0; MAX_VEHICLES * VEHICLE_FEATURES],
            routes: vec![0.0; MAX_ROUTES * ROUTE_FEATURES],
            stop_valid: vec![0.0; MAX_ROUTES * MAX_DIRECTIONS * MAX_STOPS],
            vehicle_valid: vec![0.0; MAX_VEHICLES],
            route_valid: vec![0.0; MAX_ROUTES],
            context: vec![0.0; 3],
        }
    }
}

fn stop_base(route: usize, direction_index: usize, origin: usize) -> usize {
    (route * MAX_DIRECTIONS + direction_index) * MAX_STOPS + origin
}

fn stop_index(route: usize, direction_index: usize, origin: usize, channel: usize) -> usize {
    stop_base(route, direction_index, origin) * STOP_CHANNELS + channel
}

fn history_index(route: usize, direction_index: usize, origin: usize, lag: usize) -> usize {
    stop_base(route, direction_index, origin) * HISTORY_LAGS + lag
}

fn encode_node(node: Option<i64>, node_count: i64) -> f32 {
    match node {
        None => 0.0,
        Some(value) => ((value + 1) as f64 / (node_count + 1) as f64) as f32,
    }
}

/// The oracle scales counts with `count / 40` in float64 before storing into a
/// float32 tensor; reproduce that exact rounding.
fn scaled(count: i64) -> f32 {
    (count as f64 / QUEUE_SCALE) as f32
}

#[allow(clippy::too_many_arguments)]
fn accumulate(
    cohort: &PassengerCohort,
    obs: &mut Observation,
    age_sum: &mut [f64],
    age_count: &mut [f64],
    current_time_s: i64,
    tick_s: i64,
    interval_s: i64,
    last_start_s: i64,
) {
    let direction_index = if cohort.direction == 1 { 0 } else { 1 };
    let route = cohort.route_id as usize;
    let origin = cohort.origin_index as usize;
    let base = stop_base(route, direction_index, origin);
    let add = scaled(cohort.count);

    if cohort.status == PassengerStatus::Waiting {
        obs.stops[stop_index(route, direction_index, origin, 0)] += add;
        let raw = (current_time_s - cohort.arrival_tick * tick_s) as f64;
        let age = raw.max(0.0);
        age_sum[base] += age * cohort.count as f64;
        age_count[base] += cohort.count as f64;
        let age_norm = age / AGE_SCALE;
        let channel = stop_index(route, direction_index, origin, 2);
        if (obs.stops[channel] as f64) < age_norm {
            obs.stops[channel] = age_norm as f32;
        }
        if age >= EXCESSIVE_WAIT_S {
            obs.stops[stop_index(route, direction_index, origin, 3)] += add;
        }
    }

    let arrival_s = cohort.arrival_tick * tick_s;
    if (0..current_time_s).contains(&arrival_s) {
        let lag = (current_time_s - 1 - arrival_s) / interval_s;
        if (0..HISTORY_LAGS as i64).contains(&lag) {
            obs.arrival_history[history_index(route, direction_index, origin, lag as usize)] += add;
        }
        if (last_start_s..current_time_s).contains(&arrival_s) {
            obs.stops[stop_index(route, direction_index, origin, 6)] += add;
        }
    }

    if cohort.status == PassengerStatus::Onboard {
        if let Some(tick) = cohort.boarding_tick {
            let boarded_s = tick * tick_s;
            if (last_start_s..current_time_s).contains(&boarded_s) {
                obs.stops[stop_index(route, direction_index, origin, 4)] += add;
            }
        }
    }

    if cohort.status == PassengerStatus::Completed {
        if let Some(tick) = cohort.completion_tick {
            let completed_s = tick * tick_s;
            if (last_start_s..current_time_s).contains(&completed_s) {
                obs.stops[stop_index(route, direction_index, origin, 5)] += add;
            }
        }
    }
}

pub fn observe(state: &WorldState, scenario: &Scenario) -> Observation {
    let config = &scenario.config;
    let mut obs = Observation::zeros();
    let tick_s = config.tick_s;
    let interval_s = config.control_interval_s;
    let current_time_s = state.current_time_s;
    let last_start_s = current_time_s - interval_s;
    let node_count = scenario.network.depot_node as i64 + 1;

    let cell_count = MAX_ROUTES * MAX_DIRECTIONS * MAX_STOPS;
    let mut age_sum = vec![0.0f64; cell_count];
    let mut age_count = vec![0.0f64; cell_count];

    // Waiting -> onboard -> recent finished: the oracle's `iter_cohorts` order.
    for cohort in &state.cohorts {
        accumulate(
            cohort,
            &mut obs,
            &mut age_sum,
            &mut age_count,
            current_time_s,
            tick_s,
            interval_s,
            last_start_s,
        );
    }
    for vehicle in &state.vehicles {
        for cohort in &vehicle.passengers {
            accumulate(
                cohort,
                &mut obs,
                &mut age_sum,
                &mut age_count,
                current_time_s,
                tick_s,
                interval_s,
                last_start_s,
            );
        }
    }
    for cohort in &state.recent_finished {
        accumulate(
            cohort,
            &mut obs,
            &mut age_sum,
            &mut age_count,
            current_time_s,
            tick_s,
            interval_s,
            last_start_s,
        );
    }

    for base in 0..cell_count {
        let value = if age_count[base] > 0.0 {
            age_sum[base] / age_count[base] / AGE_SCALE
        } else {
            0.0
        };
        obs.stops[base * STOP_CHANNELS + 1] = value as f32;
    }

    let depot_count = state.depot_count() as f64;
    for route in &scenario.network.routes {
        let rid = route.route_id as usize;
        obs.route_valid[rid] = 1.0;
        for direction_index in 0..MAX_DIRECTIONS {
            for origin in 0..route.stops.len() {
                obs.stop_valid[stop_base(rid, direction_index, origin)] = 1.0;
            }
        }
        let plus = route.stops[0];
        let minus = *route.stops.last().expect("route has stops");
        obs.routes[rid * ROUTE_FEATURES] =
            (state.headway_targets_s[rid] as f64 / HEADWAY_SCALE) as f32;
        let last_plus = *state
            .last_full_departure_s
            .get(&(route.route_id, 1, plus))
            .unwrap_or(&-900);
        let last_minus = *state
            .last_full_departure_s
            .get(&(route.route_id, -1, minus))
            .unwrap_or(&-900);
        obs.routes[rid * ROUTE_FEATURES + 1] =
            ((current_time_s - last_plus) as f64 / HEADWAY_SCALE) as f32;
        obs.routes[rid * ROUTE_FEATURES + 2] =
            ((current_time_s - last_minus) as f64 / HEADWAY_SCALE) as f32;

        let mut full = 0i64;
        let mut incoming = 0i64;
        let mut short = 0i64;
        for vehicle in &state.vehicles {
            if vehicle.route_id != Some(route.route_id) {
                continue;
            }
            if vehicle.pattern == Pattern::Short {
                short += 1;
            }
            if vehicle.phase == Phase::Deadhead {
                incoming += 1;
            } else if vehicle.pattern == Pattern::Full {
                full += 1;
            }
        }
        obs.routes[rid * ROUTE_FEATURES + 3] = (full as f64 / FLEET_SCALE) as f32;
        obs.routes[rid * ROUTE_FEATURES + 4] = (incoming as f64 / FLEET_SCALE) as f32;
        obs.routes[rid * ROUTE_FEATURES + 5] = (short as f64 / FLEET_SCALE) as f32;
        let elapsed = current_time_s - state.headway_changed_at_s[rid];
        let cooldown_norm = (HEADWAY_FLOOR_S - elapsed).max(0) as f64 / HEADWAY_FLOOR_S as f64;
        obs.routes[rid * ROUTE_FEATURES + 6] = cooldown_norm as f32;
        obs.routes[rid * ROUTE_FEATURES + 7] = (depot_count / FLEET_SCALE) as f32;
    }

    let horizon = config.horizon_s as f64;
    for vehicle in &state.vehicles {
        let id = vehicle.vehicle_id;
        obs.vehicle_valid[id] = 1.0;
        let base = id * VEHICLE_FEATURES;
        obs.vehicles[base + vehicle.phase.as_i64() as usize] = 1.0;
        match vehicle.route_id {
            None => {
                obs.vehicles[base + ROUTE_OFFSET] = 1.0;
                obs.vehicles[base + TARGET_OFFSET] = 1.0;
            }
            Some(route_id) => {
                obs.vehicles[base + ROUTE_OFFSET + 1 + route_id as usize] = 1.0;
                obs.vehicles[base + TARGET_OFFSET + 1 + route_id as usize] = 1.0;
            }
        }
        let pattern_slot = if vehicle.pattern == Pattern::Full {
            0
        } else {
            1
        };
        obs.vehicles[base + PATTERN_OFFSET + pattern_slot] = 1.0;
        let direction_slot = if vehicle.direction == 1 { 0 } else { 1 };
        obs.vehicles[base + DIRECTION_OFFSET + direction_slot] = 1.0;
        obs.vehicles[base + NODE_OFFSET] = encode_node(Some(vehicle.node as i64), node_count);
        obs.vehicles[base + NODE_OFFSET + 1] =
            encode_node(vehicle.next_node.map(|node| node as i64), node_count);
        let remaining = vehicle.remaining_s as f64 / horizon;
        obs.vehicles[base + SCALAR_OFFSET] = (vehicle.load as f64 / LOAD_SCALE) as f32;
        obs.vehicles[base + SCALAR_OFFSET + 1] = remaining as f32;
        obs.vehicles[base + SCALAR_OFFSET + 2] = if vehicle.phase == Phase::TerminalIdle {
            0.0
        } else {
            remaining as f32
        };
        let cooldown = (vehicle.cooldown_until_s - current_time_s).max(0) as f64 / COOLDOWN_SCALE;
        obs.vehicles[base + SCALAR_OFFSET + 3] = cooldown as f32;
        obs.vehicles[base + SCALAR_OFFSET + 4] = remaining as f32;
    }

    let time = current_time_s as f64 / horizon;
    obs.context = vec![time as f32, (1.0 - time) as f32, 0.0];
    obs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{initial_state, Network, Route, SimConfig, VehicleSpec};

    fn config() -> SimConfig {
        SimConfig {
            route_count: 1,
            stops_per_route: 2,
            fleet_size: 2,
            capacity: 5,
            comfort_capacity: 3,
            tick_s: 30,
            control_interval_s: 120,
            horizon_s: 480,
            demand_end_s: 240,
            patience_s: 300,
            dwell_s: 30,
            layover_s: 120,
            traffic_bucket_s: 300,
            schema_version: 2,
        }
    }

    fn scenario() -> Scenario {
        Scenario {
            config: config(),
            network: Network {
                routes: vec![Route {
                    route_id: 0,
                    stops: vec![0, 1],
                    short_turn_stop: 0,
                }],
                depot_node: 2,
                edge_base_s: vec![180],
            },
            fleet: (0..2)
                .map(|id| VehicleSpec {
                    vehicle_id: id,
                    route_id: Some(0),
                    node: id as u32,
                    direction: 1,
                })
                .collect(),
            arrivals: crate::domain::SparseArrivals::from_dense(
                &vec![0i32; 16 * 2 * 2 * 2],
                [16, 1, 2, 2, 2],
            ),
            traffic_tape: vec![1.0; 3],
            traffic_dims: [1, 3],
            enable_reassign: true,
            enable_short_turn: true,
        }
    }

    #[test]
    fn reset_observation_has_frozen_shapes_and_validity() {
        let scenario = scenario();
        let state = initial_state(&scenario).unwrap();
        let obs = observe(&state, &scenario);
        assert_eq!(obs.stops.len(), 4 * 2 * 8 * 7);
        assert_eq!(obs.arrival_history.len(), 4 * 2 * 8 * 5);
        assert_eq!(obs.vehicles.len(), 16 * 27);
        assert_eq!(obs.route_valid[0], 1.0);
        assert_eq!(obs.stop_valid[stop_base(0, 0, 0)], 1.0);
        assert_eq!(obs.stop_valid[stop_base(0, 0, 1)], 1.0);
        assert_eq!(obs.stop_valid[stop_base(0, 1, 0)], 1.0);
        assert_eq!(obs.stop_valid[stop_base(0, 0, 2)], 0.0);
        assert_eq!(obs.vehicle_valid[0], 1.0);
        assert_eq!(obs.vehicle_valid[1], 1.0);
        assert_eq!(obs.vehicle_valid[2], 0.0);
        assert_eq!(obs.context, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn waiting_cohort_fills_queue_and_age_channels() {
        let mut scenario = scenario();
        let mut dense = vec![0i32; 16 * 2 * 2 * 2];
        dense[1] = 20; // tick 0, route 0, dir 0, origin 0, dest 1
        scenario.arrivals = crate::domain::SparseArrivals::from_dense(&dense, [16, 1, 2, 2, 2]);
        let mut state = initial_state(&scenario).unwrap();
        state.conservation_checks = true;
        // Advance one interval so the cohort has aged one control interval.
        crate::engine::advance_interval(&mut state, &scenario, 0).unwrap();
        let obs = observe(&state, &scenario);
        let queue = obs.stops[stop_index(0, 0, 0, 0)];
        assert!(queue > 0.0);
        // Age channel is mean age / 2700; the cohort waited one interval.
        let age = obs.stops[stop_index(0, 0, 0, 1)];
        assert!(age > 0.0);
    }

    #[test]
    fn arrival_history_uses_only_the_last_five_intervals() {
        let scenario = scenario();
        let mut state = initial_state(&scenario).unwrap();
        let interval = scenario.config.control_interval_s;
        state.current_time_s = 6 * interval; // 720 s
        for arrival_tick in [4i64, 3] {
            state.cohorts.push(crate::domain::PassengerCohort {
                cohort_id: arrival_tick,
                lineage_id: arrival_tick,
                route_id: 0,
                direction: 1,
                origin_index: 0,
                destination_index: 1,
                arrival_tick,
                count: 40,
                status: crate::domain::PassengerStatus::Waiting,
                first_denied: false,
                boarding_tick: None,
                completion_tick: None,
                abandonment_tick: None,
            });
        }
        let obs = observe(&state, &scenario);
        // arrival_tick 4 (120 s) is inside the window at lag 4; tick 3 (90 s) is out.
        assert_eq!(obs.arrival_history[history_index(0, 0, 0, 4)], 1.0);
        let total: f32 = obs.arrival_history.iter().sum();
        assert_eq!(total, 1.0);
        // Neither arrival lies in the most recent interval [600, 720).
        assert_eq!(obs.stops[stop_index(0, 0, 0, 6)], 0.0);
    }

    #[test]
    fn recent_finished_ring_keeps_completion_in_window() {
        let scenario = scenario();
        let mut state = initial_state(&scenario).unwrap();
        let interval = scenario.config.control_interval_s;
        state.push_recent_finished(crate::domain::PassengerCohort {
            cohort_id: 0,
            lineage_id: 0,
            route_id: 0,
            direction: 1,
            origin_index: 0,
            destination_index: 1,
            arrival_tick: 1,
            count: 5,
            status: crate::domain::PassengerStatus::Completed,
            first_denied: false,
            boarding_tick: Some(2),
            completion_tick: Some(30), // 900 s
            abandonment_tick: None,
        });
        state.current_time_s = 1000;
        state.prune_recent_finished(scenario.config.tick_s, interval);
        assert_eq!(state.recent_finished.len(), 1);
        state.current_time_s = 1200;
        state.prune_recent_finished(scenario.config.tick_s, interval);
        assert_eq!(state.recent_finished.len(), 0);
    }

    #[test]
    fn observation_ignores_future_tape() {
        let mut first = scenario();
        let mut second = scenario();
        // Differ only at tick 5 (t = 150 s), past the first control interval.
        let mut dense_a = vec![0i32; 16 * 2 * 2 * 2];
        dense_a[5 * 8 + 1] = 7;
        let mut dense_b = vec![0i32; 16 * 2 * 2 * 2];
        dense_b[5 * 8 + 1] = 3;
        first.arrivals = crate::domain::SparseArrivals::from_dense(&dense_a, [16, 1, 2, 2, 2]);
        second.arrivals = crate::domain::SparseArrivals::from_dense(&dense_b, [16, 1, 2, 2, 2]);
        let mut state_a = initial_state(&first).unwrap();
        let mut state_b = initial_state(&second).unwrap();
        crate::engine::advance_interval(&mut state_a, &first, 0).unwrap();
        crate::engine::advance_interval(&mut state_b, &second, 0).unwrap();
        let obs_a = observe(&state_a, &first);
        let obs_b = observe(&state_b, &second);
        assert_eq!(obs_a.stops, obs_b.stops);
        assert_eq!(obs_a.arrival_history, obs_b.arrival_history);
        // Once the differing tick is in the past, the observations diverge.
        crate::engine::advance_interval(&mut state_a, &first, 0).unwrap();
        crate::engine::advance_interval(&mut state_b, &second, 0).unwrap();
        let obs_a = observe(&state_a, &first);
        let obs_b = observe(&state_b, &second);
        assert_ne!(obs_a.stops, obs_b.stops);
    }
}
