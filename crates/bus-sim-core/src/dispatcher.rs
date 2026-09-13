//! Terminal dispatcher and accepted-action effects.

use crate::actions::{action_at, Action};
use crate::domain::{Pattern, Phase, Scenario, WorldState};
use crate::error::KernelError;
use crate::guards::valid_action_mask;
use crate::vehicles::begin_service_edge;

fn assign_short(state: &mut WorldState, scenario: &Scenario, bus_id: usize, route_id: u32) {
    let short_turn_stop = scenario.route(route_id).short_turn_stop;
    let cooldown = state.current_time_s + 1200;
    let bus = &mut state.vehicles[bus_id];
    bus.route_id = Some(route_id);
    bus.pattern = Pattern::Short;
    bus.turn_stop = Some(short_turn_stop);
    bus.direction = 1;
    bus.pending_extra = true;
    bus.cooldown_until_s = cooldown;
}

/// Apply one decision. Returns the number of mission changes (0 or 1).
pub fn apply_action(
    state: &mut WorldState,
    scenario: &Scenario,
    action_index: usize,
) -> Result<i64, KernelError> {
    let action = action_at(action_index).ok_or(KernelError::InvalidAction(action_index as i64))?;
    if action == Action::Noop {
        return Ok(0);
    }
    if !valid_action_mask(state, scenario)[action_index] {
        return Err(KernelError::InvalidAction(action_index as i64));
    }
    match action {
        Action::Noop => Ok(0),
        Action::Dispatch { bus, route } => {
            let first_stop = scenario.route(route).stops[0];
            let cooldown = state.current_time_s + 1200;
            let vehicle = &mut state.vehicles[bus];
            vehicle.route_id = Some(route);
            vehicle.phase = Phase::Deadhead;
            vehicle.next_node = Some(first_stop);
            vehicle.remaining_s = 360;
            vehicle.direction = 1;
            vehicle.pending_extra = true;
            vehicle.cooldown_until_s = cooldown;
            state
                .accepted_actions
                .push(crate::domain::AcceptedActionEvent {
                    time_s: state.current_time_s,
                    kind: "DISPATCH",
                    bus_id: Some(bus),
                    route_id: Some(route),
                });
            Ok(1)
        }
        Action::Recall { bus } => {
            let depot = scenario.network.depot_node;
            let cooldown = state.current_time_s + 1200;
            let vehicle = &mut state.vehicles[bus];
            vehicle.route_id = None;
            vehicle.phase = Phase::Deadhead;
            vehicle.next_node = Some(depot);
            vehicle.remaining_s = 360;
            vehicle.pattern = Pattern::Full;
            vehicle.turn_stop = None;
            vehicle.pending_extra = false;
            vehicle.cooldown_until_s = cooldown;
            state
                .accepted_actions
                .push(crate::domain::AcceptedActionEvent {
                    time_s: state.current_time_s,
                    kind: "RECALL",
                    bus_id: Some(bus),
                    route_id: None,
                });
            Ok(1)
        }
        Action::Reassign { bus, route } => {
            let first_stop = scenario.route(route).stops[0];
            let cooldown = state.current_time_s + 1200;
            let vehicle = &mut state.vehicles[bus];
            vehicle.route_id = Some(route);
            vehicle.phase = Phase::Deadhead;
            vehicle.next_node = Some(first_stop);
            vehicle.remaining_s = 360;
            vehicle.direction = 1;
            vehicle.pattern = Pattern::Full;
            vehicle.turn_stop = None;
            vehicle.pending_extra = true;
            vehicle.cooldown_until_s = cooldown;
            state
                .accepted_actions
                .push(crate::domain::AcceptedActionEvent {
                    time_s: state.current_time_s,
                    kind: "REASSIGN",
                    bus_id: Some(bus),
                    route_id: Some(route),
                });
            Ok(1)
        }
        Action::ShortTurn { bus, route } => {
            assign_short(state, scenario, bus, route);
            if state.vehicles[bus].phase == Phase::DepotIdle {
                let first_stop = scenario.route(route).stops[0];
                let vehicle = &mut state.vehicles[bus];
                vehicle.phase = Phase::Deadhead;
                vehicle.next_node = Some(first_stop);
                vehicle.remaining_s = 360;
            }
            state
                .accepted_actions
                .push(crate::domain::AcceptedActionEvent {
                    time_s: state.current_time_s,
                    kind: "SHORT_TURN",
                    bus_id: Some(bus),
                    route_id: Some(route),
                });
            Ok(1)
        }
        Action::SetHeadway { route, headway_s } => {
            state.headway_targets_s[route as usize] = headway_s;
            state.headway_changed_at_s[route as usize] = state.current_time_s;
            state
                .accepted_actions
                .push(crate::domain::AcceptedActionEvent {
                    time_s: state.current_time_s,
                    kind: "SET_HEADWAY",
                    bus_id: None,
                    route_id: Some(route),
                });
            Ok(0)
        }
    }
}

fn terminals(route: &crate::domain::Route) -> [(i32, u32); 3] {
    [
        (1, route.stops[0]),
        (-1, *route.stops.last().unwrap()),
        (-1, route.stops[route.short_turn_stop as usize]),
    ]
}

pub fn dispatch_ready(state: &mut WorldState, scenario: &Scenario) {
    for route in &scenario.network.routes {
        let route_id = route.route_id;
        for (direction, node) in terminals(route) {
            let key = (route_id, direction, node);
            let last_any = *state.last_departure_s.get(&key).unwrap_or(&-900);
            if state.current_time_s - last_any < 120 {
                continue;
            }
            let extra: Option<usize> = state
                .vehicles
                .iter()
                .filter(|vehicle| {
                    vehicle.pending_extra
                        && vehicle.route_id == Some(route_id)
                        && vehicle.phase == Phase::TerminalIdle
                        && vehicle.node == node
                        && vehicle.direction == direction
                })
                .map(|vehicle| vehicle.vehicle_id)
                .min();
            if let Some(bus_id) = extra {
                begin_service_edge(state, scenario, bus_id);
                state.vehicles[bus_id].pending_extra = false;
                state.last_departure_s.insert(key, state.current_time_s);
                let pattern = state.vehicles[bus_id].pattern;
                if pattern == Pattern::Full {
                    state
                        .last_full_departure_s
                        .insert(key, state.current_time_s);
                }
                state.departures.push(crate::domain::DepartureEvent {
                    time_s: state.current_time_s,
                    route_id,
                    direction,
                    bus_id,
                    pattern_extra: true,
                    pattern,
                });
                continue;
            }
            let last_full = *state.last_full_departure_s.get(&key).unwrap_or(&-900);
            if state.current_time_s - last_full < state.headway_targets_s[route_id as usize] {
                continue;
            }
            let candidate: Option<usize> = state
                .vehicles
                .iter()
                .filter(|vehicle| {
                    vehicle.route_id == Some(route_id)
                        && vehicle.pattern == Pattern::Full
                        && vehicle.phase == Phase::TerminalIdle
                        && vehicle.node == node
                        && vehicle.direction == direction
                })
                .map(|vehicle| vehicle.vehicle_id)
                .min();
            let Some(bus_id) = candidate else { continue };
            begin_service_edge(state, scenario, bus_id);
            state.last_departure_s.insert(key, state.current_time_s);
            state
                .last_full_departure_s
                .insert(key, state.current_time_s);
            state.departures.push(crate::domain::DepartureEvent {
                time_s: state.current_time_s,
                route_id,
                direction,
                bus_id,
                pattern_extra: false,
                pattern: state.vehicles[bus_id].pattern,
            });
        }
    }
}
