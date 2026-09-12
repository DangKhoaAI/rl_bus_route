//! Native bus fleet-control kernel: an exact port of the Python oracle
//! (`src/bus_rl`) used for Rust-migration parity (stages R1-R3).
//!
//! The crate is pure (no Python dependency); `bus-sim-py` is the PyO3 bridge.

pub mod actions;
pub mod costs;
pub mod dispatcher;
pub mod domain;
pub mod engine;
pub mod error;
pub mod guards;
pub mod passengers;
pub mod snapshot;
pub mod travel;
pub mod vehicles;

use serde::Deserialize;

pub use domain::{initial_state, Scenario, SimConfig, StepCosts, VehicleSpec, WorldState};
pub use error::KernelError;

#[derive(Deserialize)]
struct ScenarioPayload {
    config: SimConfig,
    network: domain::Network,
    fleet: Vec<VehicleSpec>,
    #[serde(default)]
    enable_reassign: bool,
    #[serde(default)]
    enable_short_turn: bool,
}

/// Build a packed scenario from JSON metadata plus owned tape buffers.
///
/// Arrays are copied once here; `initial_state` never re-reads them.
pub fn scenario_from_json(
    json: &str,
    arrival_tape: Vec<i32>,
    arrival_dims: [usize; 5],
    traffic_tape: Vec<f32>,
    traffic_dims: [usize; 2],
) -> Result<Scenario, KernelError> {
    let payload: ScenarioPayload = serde_json::from_str(json)
        .map_err(|error| KernelError::InvalidScenario(error.to_string()))?;
    let expected_arrivals = arrival_dims.iter().product::<usize>();
    if arrival_tape.len() != expected_arrivals {
        return Err(KernelError::InvalidScenario(format!(
            "arrival buffer length {} != shape product {}",
            arrival_tape.len(),
            expected_arrivals
        )));
    }
    let expected_traffic = traffic_dims.iter().product::<usize>();
    if traffic_tape.len() != expected_traffic {
        return Err(KernelError::InvalidScenario(format!(
            "traffic buffer length {} != shape product {}",
            traffic_tape.len(),
            expected_traffic
        )));
    }
    let scenario = Scenario {
        config: payload.config,
        network: payload.network,
        fleet: payload.fleet,
        arrival_tape,
        arrival_dims,
        traffic_tape,
        traffic_dims,
        enable_reassign: payload.enable_reassign,
        enable_short_turn: payload.enable_short_turn,
    };
    scenario.validate()?;
    Ok(scenario)
}

#[cfg(test)]
mod tests {
    use super::*;
    use domain::{Pattern, Phase, SimConfig};

    fn tiny_config() -> SimConfig {
        SimConfig {
            route_count: 1,
            stops_per_route: 2,
            fleet_size: 4,
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

    fn tiny_scenario() -> Scenario {
        let config = tiny_config();
        let network = domain::Network {
            routes: vec![domain::Route {
                route_id: 0,
                stops: vec![0, 1],
                short_turn_stop: 0,
            }],
            depot_node: 2,
            edge_base_s: vec![180],
        };
        let fleet = (0..4)
            .map(|id| {
                let (route_id, node, direction) = match id {
                    0 => (Some(0), 0, 1),
                    1 => (Some(0), 1, -1),
                    _ => (None, 2, 1),
                };
                VehicleSpec {
                    vehicle_id: id,
                    route_id,
                    node,
                    direction,
                }
            })
            .collect();
        Scenario {
            config,
            network,
            fleet,
            arrival_tape: vec![0; 16 * 1 * 2 * 2 * 2],
            arrival_dims: [16, 1, 2, 2, 2],
            traffic_tape: vec![1.0; 3],
            traffic_dims: [1, 3],
            enable_reassign: true,
            enable_short_turn: true,
        }
    }

    #[test]
    fn initial_state_matches_spec() {
        let scenario = tiny_scenario();
        scenario.validate().unwrap();
        let state = initial_state(&scenario).unwrap();
        assert_eq!(state.vehicles.len(), 4);
        assert_eq!(state.vehicles[0].phase, Phase::TerminalIdle);
        assert_eq!(state.vehicles[2].phase, Phase::DepotIdle);
        assert_eq!(state.headway_targets_s[0], 900);
        assert_eq!(state.headway_changed_at_s[0], -600);
        assert_eq!(state.last_departure_s[&(0, 1, 0)], -900);
        assert_eq!(state.generated_total, 0);
    }

    #[test]
    fn invalid_dimensions_are_rejected() {
        let mut scenario = tiny_scenario();
        scenario.arrival_dims = [15, 1, 2, 2, 2];
        assert!(scenario.validate().is_err());
    }

    #[test]
    fn noop_and_short_turn_apply_without_mutation_of_others() {
        let scenario = tiny_scenario();
        let mut state = initial_state(&scenario).unwrap();
        let mask = guards::valid_action_mask(&state, &scenario);
        assert!(mask[0]);
        // SHORT_TURN(reserve 2, route 0) is legal from the depot.
        let short = actions::action_at(129 + 8).unwrap();
        assert_eq!(short, actions::Action::ShortTurn { bus: 2, route: 0 });
        assert!(mask[129 + 8]);
        engine::advance_interval(&mut state, &scenario, 129 + 8).unwrap();
        assert_eq!(state.vehicles[2].pattern, Pattern::Short);
        assert_eq!(state.vehicles[2].cooldown_until_s, 1200);
        assert_eq!(state.current_time_s, 120);
    }

    #[test]
    fn invalid_action_is_rejected_before_mutation() {
        let scenario = tiny_scenario();
        let mut state = initial_state(&scenario).unwrap();
        let before = state.current_time_s;
        // DISPATCH of a non-depot vehicle is masked out.
        let dispatch = actions::action_at(1).unwrap();
        assert_eq!(dispatch, actions::Action::Dispatch { bus: 0, route: 0 });
        let error = engine::advance_interval(&mut state, &scenario, 1).unwrap_err();
        assert!(matches!(error, KernelError::InvalidAction(1)));
        assert_eq!(state.current_time_s, before);
    }

    #[test]
    fn reset_is_deterministic() {
        let scenario = tiny_scenario();
        let first = snapshot::full_snapshot(&initial_state(&scenario).unwrap());
        let second = snapshot::full_snapshot(&initial_state(&scenario).unwrap());
        assert_eq!(first.time_s, second.time_s);
        assert_eq!(first.counters, second.counters);
        assert_eq!(first.vehicles, second.vehicles);
        assert_eq!(first.cohorts, second.cohorts);
    }

    #[test]
    fn partial_boarding_respects_capacity_and_conserves_mass() {
        let mut scenario = tiny_scenario();
        // 50 arrivals at tick 0, origin 0, destination 1 of route 0.
        scenario.arrival_tape[1] = 50;
        let mut state = initial_state(&scenario).unwrap();
        state.conservation_checks = true;
        engine::advance_interval(&mut state, &scenario, 0).unwrap();
        assert_eq!(state.generated_total, 50);
        assert_eq!(state.waiting_total + state.onboard_total, 50);
        assert!(state.vehicles[0].load <= state.vehicles[0].capacity);
        state.assert_conservation().unwrap();
        // The first visit denies the overflow exactly once.
        assert_eq!(state.cohorts[0].first_denied, true);
        assert_eq!(state.cohorts[0].count, 45);
    }

    #[test]
    fn config_drives_ticks_per_interval() {
        let mut scenario = tiny_scenario();
        scenario.config.control_interval_s = 240;
        scenario.config.horizon_s = 960;
        scenario.arrival_dims = [32, 1, 2, 2, 2];
        scenario.arrival_tape = vec![0; 32 * 1 * 2 * 2 * 2];
        let mut state = initial_state(&scenario).unwrap();
        engine::advance_interval(&mut state, &scenario, 0).unwrap();
        assert_eq!(state.current_time_s, 240);
        assert_eq!(state.event_log.len(), 8);
    }
}
