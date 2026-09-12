//! Domain types: enums, configuration, scenario, and mutable world state.
//!
//! Field-by-field port of `src/bus_rl/domain.py`. Number types are intentionally
//! wide (`i64`) and mutated only through checked helpers so counters cannot
//! silently wrap.

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::error::KernelError;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    DepotIdle = 0,
    Deadhead = 1,
    TerminalIdle = 2,
    ServiceMoving = 3,
    ServiceDwell = 4,
    Layover = 5,
}

impl Phase {
    pub fn as_i64(self) -> i64 {
        self as i64
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Pattern {
    Full = 0,
    Short = 1,
}

impl Pattern {
    pub fn as_i64(self) -> i64 {
        self as i64
    }

    pub fn name(self) -> &'static str {
        match self {
            Pattern::Full => "FULL",
            Pattern::Short => "SHORT",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PassengerStatus {
    Waiting = 0,
    Onboard = 1,
    Completed = 2,
    Abandoned = 3,
}

impl PassengerStatus {
    pub fn as_i64(self) -> i64 {
        self as i64
    }
}

#[derive(Clone, Deserialize, Debug)]
pub struct SimConfig {
    pub route_count: usize,
    pub stops_per_route: usize,
    pub fleet_size: usize,
    pub capacity: i64,
    #[allow(dead_code)]
    pub comfort_capacity: i64,
    pub tick_s: i64,
    pub control_interval_s: i64,
    pub horizon_s: i64,
    pub demand_end_s: i64,
    pub patience_s: i64,
    pub dwell_s: i64,
    pub layover_s: i64,
    pub traffic_bucket_s: i64,
    #[allow(dead_code)]
    pub schema_version: i64,
}

impl SimConfig {
    pub fn validate(&self) -> Result<(), KernelError> {
        if self.route_count > 4 || self.stops_per_route > 8 || self.fleet_size > 16 {
            return Err(KernelError::InvalidConfig(
                "config exceeds fixed observation/action encoding limits".into(),
            ));
        }
        if self.route_count == 0
            || self.stops_per_route == 0
            || self.fleet_size == 0
            || self.capacity <= 0
        {
            return Err(KernelError::InvalidConfig(
                "route, stop, fleet, and capacity counts must be positive".into(),
            ));
        }
        if self.control_interval_s % self.tick_s != 0 || self.horizon_s % self.tick_s != 0 {
            return Err(KernelError::InvalidConfig(
                "control interval and horizon must be multiples of tick_s".into(),
            ));
        }
        if self.demand_end_s > self.horizon_s || self.demand_end_s % self.tick_s != 0 {
            return Err(KernelError::InvalidConfig(
                "demand_end_s must be a tick-aligned time within the horizon".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Deserialize, Debug)]
pub struct Route {
    pub route_id: u32,
    pub stops: Vec<u32>,
    pub short_turn_stop: u32,
}

#[derive(Clone, Deserialize, Debug)]
pub struct Network {
    pub routes: Vec<Route>,
    pub depot_node: u32,
    pub edge_base_s: Vec<i64>,
}

#[derive(Clone, Deserialize, Debug)]
pub struct VehicleSpec {
    pub vehicle_id: usize,
    pub route_id: Option<u32>,
    pub node: u32,
    pub direction: i32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StepCosts {
    pub waiting_pm: f64,
    pub onboard_pm: f64,
    pub crowding_pm: f64,
    pub active_bus_min: f64,
    pub deadhead_bus_min: f64,
    pub excessive_wait_pm: f64,
    pub first_denied_count: i64,
    pub abandoned_count: i64,
    pub mission_changes: i64,
    pub terminal_unfinished_count: i64,
}

impl StepCosts {
    pub fn as_array(self) -> [f64; 10] {
        [
            self.waiting_pm,
            self.onboard_pm,
            self.crowding_pm,
            self.active_bus_min,
            self.deadhead_bus_min,
            self.excessive_wait_pm,
            self.first_denied_count as f64,
            self.abandoned_count as f64,
            self.mission_changes as f64,
            self.terminal_unfinished_count as f64,
        ]
    }
}

pub fn add_costs(left: &StepCosts, right: &StepCosts) -> StepCosts {
    StepCosts {
        waiting_pm: left.waiting_pm + right.waiting_pm,
        onboard_pm: left.onboard_pm + right.onboard_pm,
        crowding_pm: left.crowding_pm + right.crowding_pm,
        active_bus_min: left.active_bus_min + right.active_bus_min,
        deadhead_bus_min: left.deadhead_bus_min + right.deadhead_bus_min,
        excessive_wait_pm: left.excessive_wait_pm + right.excessive_wait_pm,
        first_denied_count: left.first_denied_count + right.first_denied_count,
        abandoned_count: left.abandoned_count + right.abandoned_count,
        mission_changes: left.mission_changes + right.mission_changes,
        terminal_unfinished_count: left.terminal_unfinished_count + right.terminal_unfinished_count,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PassengerCohort {
    pub cohort_id: i64,
    pub lineage_id: i64,
    pub route_id: u32,
    pub direction: i32,
    pub origin_index: u32,
    pub destination_index: u32,
    pub arrival_tick: i64,
    pub count: i64,
    pub status: PassengerStatus,
    pub first_denied: bool,
    pub boarding_tick: Option<i64>,
    pub completion_tick: Option<i64>,
    pub abandonment_tick: Option<i64>,
}

#[derive(Clone, Debug)]
pub struct Vehicle {
    pub vehicle_id: usize,
    pub capacity: i64,
    pub route_id: Option<u32>,
    pub node: u32,
    pub direction: i32,
    pub phase: Phase,
    pub remaining_s: i64,
    pub next_node: Option<u32>,
    pub passengers: Vec<PassengerCohort>,
    pub visit_id: i64,
    pub cooldown_until_s: i64,
    pub pattern: Pattern,
    pub turn_stop: Option<u32>,
    pub pending_extra: bool,
    pub load: i64,
}

#[derive(Clone, Copy, Debug)]
pub struct EventLogEntry {
    pub time_s: i64,
    pub boarded: i64,
    pub denied: i64,
    pub abandoned: i64,
}

#[derive(Clone, Debug)]
pub struct DepartureEvent {
    pub time_s: i64,
    pub route_id: u32,
    pub direction: i32,
    pub bus_id: usize,
    pub pattern_extra: bool,
    pub pattern: Pattern,
}

#[derive(Clone, Debug)]
pub struct AcceptedActionEvent {
    pub time_s: i64,
    pub kind: &'static str,
    pub bus_id: Option<usize>,
    pub route_id: Option<u32>,
}

pub struct WorldState {
    pub current_time_s: i64,
    pub vehicles: Vec<Vehicle>,
    pub cohorts: Vec<PassengerCohort>,
    pub finished: Vec<PassengerCohort>,
    pub generated_total: i64,
    pub waiting_total: i64,
    pub onboard_total: i64,
    pub completed_total: i64,
    pub abandoned_total: i64,
    pub next_cohort_id: i64,
    pub event_log: Vec<EventLogEntry>,
    pub departures: Vec<DepartureEvent>,
    pub accepted_actions: Vec<AcceptedActionEvent>,
    pub headway_targets_s: Vec<i64>,
    pub headway_changed_at_s: Vec<i64>,
    pub last_departure_s: BTreeMap<(u32, i32, u32), i64>,
    pub last_full_departure_s: BTreeMap<(u32, i32, u32), i64>,
    pub terminal_settled: bool,
    pub conservation_checks: bool,
}

impl WorldState {
    pub fn generated_count(&self) -> i64 {
        self.generated_total
    }
    pub fn waiting_count(&self) -> i64 {
        self.waiting_total
    }
    pub fn onboard_count(&self) -> i64 {
        self.onboard_total
    }
    pub fn completed_count(&self) -> i64 {
        self.completed_total
    }
    pub fn abandoned_count(&self) -> i64 {
        self.abandoned_total
    }
    pub fn depot_count(&self) -> i64 {
        self.vehicles
            .iter()
            .filter(|vehicle| vehicle.phase == Phase::DepotIdle)
            .count() as i64
    }

    /// Always-on conservation of passenger mass and capacity; the deeper
    /// hot-list/archive cross-checks only run when `conservation_checks` is set.
    pub fn assert_conservation(&self) -> Result<(), KernelError> {
        let passengers =
            self.waiting_total + self.onboard_total + self.completed_total + self.abandoned_total;
        if passengers != self.generated_total {
            return Err(KernelError::Conservation(format!(
                "passenger conservation failed: {} != {}",
                self.generated_total, passengers
            )));
        }
        for vehicle in &self.vehicles {
            if vehicle.load > vehicle.capacity {
                return Err(KernelError::Conservation(
                    "vehicle capacity exceeded".into(),
                ));
            }
        }
        if !self.conservation_checks {
            return Ok(());
        }
        if self.cohorts.iter().map(|c| c.count).sum::<i64>() != self.waiting_total {
            return Err(KernelError::Conservation(
                "waiting counter drifted from the hot list".into(),
            ));
        }
        let onboard: i64 = self.vehicles.iter().map(|v| v.load).sum();
        if onboard != self.onboard_total {
            return Err(KernelError::Conservation(
                "onboard counter drifted from cached vehicle loads".into(),
            ));
        }
        let done: i64 = self.finished.iter().map(|c| c.count).sum();
        if done != self.completed_total + self.abandoned_total {
            return Err(KernelError::Conservation(
                "finished counter drifted from the archive list".into(),
            ));
        }
        Ok(())
    }
}

/// Immutable, packed scenario. Tapes are copied once at construction; `reset`
/// only rebuilds episode state and never reads files or re-packs.
pub struct Scenario {
    pub config: SimConfig,
    pub network: Network,
    pub fleet: Vec<VehicleSpec>,
    pub arrival_tape: Vec<i32>,
    pub arrival_dims: [usize; 5],
    pub traffic_tape: Vec<f32>,
    pub traffic_dims: [usize; 2],
    pub enable_reassign: bool,
    pub enable_short_turn: bool,
}

impl Scenario {
    pub fn arrival(&self, tick: usize, route: usize, d: usize, origin: usize, dest: usize) -> i32 {
        let [ticks, routes, dirs, stops, stops2] = self.arrival_dims;
        debug_assert!(tick < ticks && route < routes && d < dirs);
        let index = (((tick * routes + route) * dirs + d) * stops + origin) * stops2 + dest;
        self.arrival_tape[index]
    }

    pub fn traffic(&self, edge: usize, bucket: usize) -> f32 {
        let [_edges, buckets] = self.traffic_dims;
        debug_assert!(bucket < buckets);
        self.traffic_tape[edge * buckets + bucket]
    }

    pub fn route(&self, route_id: u32) -> &Route {
        &self.network.routes[route_id as usize]
    }

    pub fn validate(&self) -> Result<(), KernelError> {
        self.config.validate()?;
        if self.network.routes.len() != self.config.route_count {
            return Err(KernelError::InvalidScenario(
                "route_count does not match network".into(),
            ));
        }
        for route in &self.network.routes {
            if route.stops.len() != self.config.stops_per_route {
                return Err(KernelError::InvalidScenario(
                    "stops_per_route does not match network".into(),
                ));
            }
        }
        let ticks = (self.config.horizon_s / self.config.tick_s) as usize;
        if self.arrival_dims
            != [
                ticks,
                self.config.route_count,
                2,
                self.config.stops_per_route,
                self.config.stops_per_route,
            ]
        {
            return Err(KernelError::InvalidScenario(
                "arrival tape has invalid shape".into(),
            ));
        }
        if self.traffic_dims[0] != self.config.route_count * (self.config.stops_per_route - 1) {
            return Err(KernelError::InvalidScenario(
                "traffic tape has invalid edge count".into(),
            ));
        }
        if self.fleet.len() != self.config.fleet_size {
            return Err(KernelError::InvalidScenario(
                "fleet size does not match config".into(),
            ));
        }
        Ok(())
    }
}

/// Build the initial episode state. Vehicle ordering follows `fleet` order and
/// vehicle ids are contiguous `0..fleet_size`.
pub fn initial_state(scenario: &Scenario) -> Result<WorldState, KernelError> {
    let mut vehicles = Vec::with_capacity(scenario.fleet.len());
    for spec in &scenario.fleet {
        let phase = if spec.route_id.is_none() {
            Phase::DepotIdle
        } else {
            Phase::TerminalIdle
        };
        vehicles.push(Vehicle {
            vehicle_id: spec.vehicle_id,
            capacity: scenario.config.capacity,
            route_id: spec.route_id,
            node: spec.node,
            direction: spec.direction,
            phase,
            remaining_s: 0,
            next_node: None,
            passengers: Vec::new(),
            visit_id: 0,
            cooldown_until_s: 0,
            pattern: Pattern::Full,
            turn_stop: None,
            pending_extra: false,
            load: 0,
        });
    }
    for (index, vehicle) in vehicles.iter().enumerate() {
        if vehicle.vehicle_id != index {
            return Err(KernelError::InvalidScenario(
                "vehicle ids must be contiguous from zero".into(),
            ));
        }
    }

    let mut headway_targets_s = vec![0i64; scenario.config.route_count];
    let mut headway_changed_at_s = vec![0i64; scenario.config.route_count];
    for route in &scenario.network.routes {
        headway_targets_s[route.route_id as usize] = 900;
        headway_changed_at_s[route.route_id as usize] = -600;
    }

    let mut last_departure_s = BTreeMap::new();
    for route in &scenario.network.routes {
        let rid = route.route_id;
        let plus = route.stops[0];
        let minus = *route.stops.last().unwrap();
        let short = route.stops[route.short_turn_stop as usize];
        last_departure_s.insert((rid, 1, plus), -900);
        last_departure_s.insert((rid, -1, minus), -900);
        last_departure_s.insert((rid, -1, short), -900);
    }
    let last_full_departure_s = last_departure_s.clone();

    Ok(WorldState {
        current_time_s: 0,
        vehicles,
        cohorts: Vec::new(),
        finished: Vec::new(),
        generated_total: 0,
        waiting_total: 0,
        onboard_total: 0,
        completed_total: 0,
        abandoned_total: 0,
        next_cohort_id: 0,
        event_log: Vec::new(),
        departures: Vec::new(),
        accepted_actions: Vec::new(),
        headway_targets_s,
        headway_changed_at_s,
        last_departure_s,
        last_full_departure_s,
        terminal_settled: false,
        conservation_checks: false,
    })
}
