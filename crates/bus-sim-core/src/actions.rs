//! The fixed 221-slot action table (`schema v2`), ported from
//! `src/bus_sim/oracle/actions.py`.

pub const F_MAX: usize = 16;
pub const R_MAX: usize = 4;
pub const HEADWAYS: [i64; 3] = [360, 600, 900];
pub const ACTION_COUNT: usize = 221;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Action {
    Noop,
    Dispatch { bus: usize, route: u32 },
    Reassign { bus: usize, route: u32 },
    ShortTurn { bus: usize, route: u32 },
    Recall { bus: usize },
    SetHeadway { route: u32, headway_s: i64 },
}

impl Action {
    pub fn kind(self) -> &'static str {
        match self {
            Action::Noop => "NOOP",
            Action::Dispatch { .. } => "DISPATCH",
            Action::Reassign { .. } => "REASSIGN",
            Action::ShortTurn { .. } => "SHORT_TURN",
            Action::Recall { .. } => "RECALL",
            Action::SetHeadway { .. } => "SET_HEADWAY",
        }
    }
}

/// Decode an action index into the frozen table layout.
pub fn action_at(index: usize) -> Option<Action> {
    if index == 0 {
        return Some(Action::Noop);
    }
    let mut cursor = index - 1;
    if cursor < F_MAX * R_MAX {
        return Some(Action::Dispatch {
            bus: cursor / R_MAX,
            route: (cursor % R_MAX) as u32,
        });
    }
    cursor -= F_MAX * R_MAX;
    if cursor < F_MAX * R_MAX {
        return Some(Action::Reassign {
            bus: cursor / R_MAX,
            route: (cursor % R_MAX) as u32,
        });
    }
    cursor -= F_MAX * R_MAX;
    if cursor < F_MAX * R_MAX {
        return Some(Action::ShortTurn {
            bus: cursor / R_MAX,
            route: (cursor % R_MAX) as u32,
        });
    }
    cursor -= F_MAX * R_MAX;
    if cursor < F_MAX {
        return Some(Action::Recall { bus: cursor });
    }
    cursor -= F_MAX;
    if cursor < R_MAX * HEADWAYS.len() {
        return Some(Action::SetHeadway {
            route: (cursor / HEADWAYS.len()) as u32,
            headway_s: HEADWAYS[cursor % HEADWAYS.len()],
        });
    }
    None
}
