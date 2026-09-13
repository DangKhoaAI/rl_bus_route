//! Vehicle phase transitions for full service and short-turn missions.

use crate::domain::{Pattern, Phase, Scenario, Vehicle, WorldState};
use crate::passengers::{alight_visit, board_visit};
use crate::travel::edge_duration_s;

fn stop_index(scenario: &Scenario, route_id: u32, node: u32) -> usize {
    scenario
        .route(route_id)
        .stops
        .iter()
        .position(|stop| *stop == node)
        .expect("vehicle node must belong to its route")
}

fn turn_stop(bus: &Vehicle, scenario: &Scenario) -> u32 {
    bus.turn_stop.unwrap_or_else(|| {
        let route_id = bus.route_id.expect("assigned bus");
        scenario.route(route_id).short_turn_stop
    })
}

fn service_end_index(bus: &Vehicle, scenario: &Scenario) -> usize {
    let _ = bus.route_id.expect("assigned bus");
    if bus.pattern == Pattern::Short {
        if bus.direction == 1 {
            turn_stop(bus, scenario) as usize
        } else {
            0
        }
    } else if bus.direction == 1 {
        scenario.config.stops_per_route - 1
    } else {
        0
    }
}

pub fn enter_next_edge(state: &mut WorldState, scenario: &Scenario, bus_id: usize) {
    let route_id = state.vehicles[bus_id]
        .route_id
        .expect("unassigned bus cannot enter a service edge");
    let direction = state.vehicles[bus_id].direction;
    let pattern = state.vehicles[bus_id].pattern;
    let node = state.vehicles[bus_id].node;
    let route = scenario.route(route_id);
    let current_index = route
        .stops
        .iter()
        .position(|stop| *stop == node)
        .expect("node on route");
    let target_index = current_index as i64 + direction as i64;
    if pattern == Pattern::Short {
        let limit = {
            let bus = &state.vehicles[bus_id];
            turn_stop(bus, scenario) as i64
        };
        assert!(
            target_index >= 0 && target_index <= limit,
            "short pattern would leave the allowed stops"
        );
    } else {
        assert!(
            target_index >= 0 && (target_index as usize) < route.stops.len(),
            "service edge would leave the route"
        );
    }
    let target = route.stops[target_index as usize];
    let duration = edge_duration_s(scenario, route_id, current_index, state.current_time_s);
    let bus = &mut state.vehicles[bus_id];
    bus.next_node = Some(target);
    bus.remaining_s = duration;
    bus.phase = Phase::ServiceMoving;
}

pub fn begin_service_edge(state: &mut WorldState, scenario: &Scenario, bus_id: usize) {
    {
        let bus = &state.vehicles[bus_id];
        assert!(
            bus.route_id.is_some() && bus.phase == Phase::TerminalIdle,
            "only an assigned terminal-idle bus may begin service"
        );
    }
    enter_next_edge(state, scenario, bus_id);
}

/// Complete one phase event, returning `(boarded, first_denied)` counters.
pub fn complete_expired_phase(
    state: &mut WorldState,
    scenario: &Scenario,
    bus_id: usize,
) -> (i64, i64) {
    let phase = state.vehicles[bus_id].phase;
    match phase {
        Phase::Deadhead => {
            let next = state.vehicles[bus_id]
                .next_node
                .expect("deadhead next node");
            let bus = &mut state.vehicles[bus_id];
            bus.node = next;
            bus.next_node = None;
            bus.phase = if next == scenario.network.depot_node {
                Phase::DepotIdle
            } else {
                Phase::TerminalIdle
            };
            (0, 0)
        }
        Phase::ServiceMoving => {
            let route_id = state.vehicles[bus_id].route_id.expect("assigned bus");
            let next = state.vehicles[bus_id].next_node.expect("service next node");
            state.vehicles[bus_id].node = next;
            state.vehicles[bus_id].next_node = None;
            let index = stop_index(scenario, route_id, next);
            alight_visit(state, bus_id, index as u32);
            if index == service_end_index(&state.vehicles[bus_id], scenario) {
                let bus = &mut state.vehicles[bus_id];
                bus.phase = Phase::Layover;
                bus.remaining_s = scenario.config.layover_s;
                bus.direction *= -1;
                return (0, 0);
            }
            let direction = state.vehicles[bus_id].direction;
            let events = board_visit(state, bus_id, route_id, direction, index as u32);
            let bus = &mut state.vehicles[bus_id];
            bus.phase = Phase::ServiceDwell;
            bus.remaining_s = scenario.config.dwell_s;
            (events.boarded_count, events.first_denied_count)
        }
        Phase::ServiceDwell => {
            enter_next_edge(state, scenario, bus_id);
            (0, 0)
        }
        Phase::Layover => {
            let pattern = state.vehicles[bus_id].pattern;
            let route_id = state.vehicles[bus_id].route_id;
            state.vehicles[bus_id].phase = Phase::TerminalIdle;
            if pattern == Pattern::Short {
                if let Some(route_id) = route_id {
                    let node = state.vehicles[bus_id].node;
                    let index = stop_index(scenario, route_id, node);
                    if index == 0 {
                        let bus = &mut state.vehicles[bus_id];
                        bus.pattern = Pattern::Full;
                        bus.turn_stop = None;
                        bus.pending_extra = false;
                    } else {
                        state.vehicles[bus_id].pending_extra = true;
                    }
                }
            }
            (0, 0)
        }
        Phase::TerminalIdle | Phase::DepotIdle => (0, 0),
    }
}
