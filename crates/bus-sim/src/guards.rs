//! Action guards. Port of `src/bus_rl/control/guards.py` (NOOP always valid).

use crate::actions::{action_at, ACTION_COUNT};
use crate::domain::{Pattern, Phase, Scenario, WorldState};

fn is_full_committed(vehicle: &crate::domain::Vehicle, route_id: u32) -> bool {
    vehicle.route_id == Some(route_id)
        && vehicle.pattern == Pattern::Full
        && vehicle.phase != Phase::DepotIdle
        && vehicle.phase != Phase::Deadhead
}

fn committed(state: &WorldState, route_id: u32) -> i64 {
    state
        .vehicles
        .iter()
        .filter(|vehicle| is_full_committed(vehicle, route_id))
        .count() as i64
}

fn donor_guard(state: &WorldState, bus_id: usize) -> bool {
    let Some(bus) = state.vehicles.get(bus_id) else {
        return false;
    };
    let Some(route_id) = bus.route_id else {
        return false;
    };
    if bus.phase != Phase::TerminalIdle || bus.load != 0 || bus.pattern != Pattern::Full {
        return false;
    }
    if committed(state, route_id) - 1 < 2 {
        return false;
    }
    let replacement = state.vehicles.iter().any(|other| {
        other.vehicle_id != bus_id
            && other.route_id == Some(route_id)
            && other.pattern == Pattern::Full
            && other.phase == Phase::TerminalIdle
            && other.node == bus.node
            && other.load == 0
    });
    if !replacement {
        return false;
    }
    let key = (route_id, bus.direction, bus.node);
    let last_any = *state.last_departure_s.get(&key).unwrap_or(&-900);
    let last_full = *state.last_full_departure_s.get(&key).unwrap_or(&-900);
    let earliest = state.current_time_s.max(last_any + 120);
    earliest <= last_full + 1200
}

pub fn valid_action_mask(state: &WorldState, scenario: &Scenario) -> Vec<bool> {
    let mut mask = vec![false; ACTION_COUNT];
    mask[0] = true;
    let route_count = scenario.config.route_count;
    for (index, slot) in mask.iter_mut().enumerate().skip(1) {
        let Some(action) = action_at(index) else {
            continue;
        };
        *slot = match action {
            crate::actions::Action::Noop => true,
            crate::actions::Action::Dispatch { bus, route } => {
                let vehicle = state.vehicles.get(bus);
                vehicle.is_some()
                    && (route as usize) < route_count
                    && vehicle.map(|v| v.phase) == Some(Phase::DepotIdle)
                    && state.current_time_s >= vehicle.map(|v| v.cooldown_until_s).unwrap_or(0)
            }
            crate::actions::Action::Recall { bus } => {
                let vehicle = state.vehicles.get(bus);
                vehicle.is_some()
                    && state.current_time_s >= vehicle.map(|v| v.cooldown_until_s).unwrap_or(0)
                    && donor_guard(state, bus)
            }
            crate::actions::Action::Reassign { bus, route } => {
                let vehicle = state.vehicles.get(bus);
                scenario.enable_reassign
                    && vehicle.is_some()
                    && (route as usize) < route_count
                    && vehicle.map(|v| v.route_id) != Some(Some(route))
                    && state.current_time_s >= vehicle.map(|v| v.cooldown_until_s).unwrap_or(0)
                    && donor_guard(state, bus)
            }
            crate::actions::Action::ShortTurn { bus, route } => match state.vehicles.get(bus) {
                None => false,
                Some(vehicle) => {
                    if !(scenario.enable_short_turn
                        && (route as usize) < route_count
                        && state.current_time_s >= vehicle.cooldown_until_s
                        && vehicle.load == 0)
                    {
                        false
                    } else if vehicle.phase == Phase::DepotIdle {
                        true
                    } else {
                        let origin = scenario.route(route).stops[0];
                        vehicle.route_id == Some(route)
                            && vehicle.pattern == Pattern::Full
                            && vehicle.phase == Phase::TerminalIdle
                            && vehicle.node == origin
                            && donor_guard(state, bus)
                    }
                }
            },
            crate::actions::Action::SetHeadway { route, headway_s } => {
                (route as usize) < route_count
                    && state.headway_targets_s[route as usize] != headway_s
                    && state.current_time_s - state.headway_changed_at_s[route as usize] >= 600
            }
        };
    }
    mask
}
