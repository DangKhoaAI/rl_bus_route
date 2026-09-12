//! PyO3 bridge exposing the native kernel to the Python parity tests.
//!
//! Module name: `bus_sim`. The production Gym wrapper lives in Python (R3);
//! this bridge is deliberately small and debug-oriented.

use bus_sim_core::costs::{interval_cost, mean_waiting_minutes, RewardConfig};
use bus_sim_core::domain::{Scenario, StepCosts, WorldState};
use bus_sim_core::snapshot::{full_snapshot, StateSnapshot};
use bus_sim_core::{engine, guards, initial_state, scenario_from_json, KernelError};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyReadonlyArray5, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Cost field order must match `bus_rl.parity.snapshot.COST_FIELDS`.
const COST_FIELDS: [&str; 10] = [
    "waiting_pm",
    "onboard_pm",
    "crowding_pm",
    "active_bus_min",
    "deadhead_bus_min",
    "excessive_wait_pm",
    "first_denied_count",
    "abandoned_count",
    "mission_changes",
    "terminal_unfinished_count",
];

fn to_py(error: KernelError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn array2<'py>(
    py: Python<'py>,
    data: Vec<i64>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let array = ndarray::Array2::from_shape_vec((rows, cols), data)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(array.into_pyarray(py))
}

fn vehicle_matrix<'py>(
    py: Python<'py>,
    vehicles: &[[i64; 14]],
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let data: Vec<i64> = vehicles.iter().flatten().copied().collect();
    array2(py, data, vehicles.len(), 14)
}

fn cohort_matrix<'py>(
    py: Python<'py>,
    cohorts: &[[i64; 14]],
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let data: Vec<i64> = cohorts.iter().flatten().copied().collect();
    array2(py, data, cohorts.len(), 14)
}

fn costs_list(costs: &StepCosts) -> Vec<f64> {
    costs.as_array().to_vec()
}

fn state_dict<'py>(py: Python<'py>, snapshot: &StateSnapshot) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("time_s", snapshot.time_s)?;
    dict.set_item("counters", PyArray1::from_slice(py, &snapshot.counters))?;
    dict.set_item("vehicles", vehicle_matrix(py, &snapshot.vehicles)?)?;
    dict.set_item("cohorts", cohort_matrix(py, &snapshot.cohorts)?)?;
    dict.set_item(
        "cohort_kind",
        PyArray1::from_slice(py, &snapshot.cohort_kind),
    )?;
    dict.set_item(
        "headway_targets",
        PyArray1::from_slice(py, &snapshot.headway_targets),
    )?;
    dict.set_item(
        "headway_changed_at",
        PyArray1::from_slice(py, &snapshot.headway_changed_at),
    )?;
    let key_data: Vec<i64> = snapshot.departure_keys.iter().flatten().copied().collect();
    dict.set_item(
        "departure_keys",
        array2(py, key_data, snapshot.departure_keys.len(), 3)?,
    )?;
    dict.set_item(
        "last_departure",
        PyArray1::from_slice(py, &snapshot.last_departure),
    )?;
    dict.set_item(
        "last_full_departure",
        PyArray1::from_slice(py, &snapshot.last_full_departure),
    )?;
    dict.set_item("terminal_settled", snapshot.terminal_settled)?;
    Ok(dict)
}

#[pyclass]
struct Kernel {
    scenario: Scenario,
    state: Option<WorldState>,
    reward: RewardConfig,
    conservation_checks: bool,
}

#[pymethods]
impl Kernel {
    #[new]
    fn new(
        scenario_json: &str,
        arrivals: PyReadonlyArray5<i32>,
        traffic: PyReadonlyArray2<f32>,
    ) -> PyResult<Self> {
        let shape = arrivals.shape();
        let arrival_dims = [shape[0], shape[1], shape[2], shape[3], shape[4]];
        let arrival_tape = arrivals
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .to_vec();
        let tshape = traffic.shape();
        let traffic_dims = [tshape[0], tshape[1]];
        let traffic_tape = traffic
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .to_vec();
        let scenario = scenario_from_json(
            scenario_json,
            arrival_tape,
            arrival_dims,
            traffic_tape,
            traffic_dims,
        )
        .map_err(to_py)?;
        Ok(Self {
            scenario,
            state: None,
            reward: RewardConfig::default(),
            conservation_checks: false,
        })
    }

    fn reset(&mut self) -> PyResult<()> {
        let mut state = initial_state(&self.scenario).map_err(to_py)?;
        state.conservation_checks = self.conservation_checks;
        self.state = Some(state);
        Ok(())
    }

    fn set_conservation_checks(&mut self, enabled: bool) {
        self.conservation_checks = enabled;
        if let Some(state) = self.state.as_mut() {
            state.conservation_checks = enabled;
        }
    }

    fn action_mask<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let state = self.require_state()?;
        let mask = guards::valid_action_mask(state, &self.scenario);
        Ok(PyArray1::from_slice(py, &mask))
    }

    #[getter]
    fn current_time_s(&self) -> PyResult<i64> {
        Ok(self.require_state()?.current_time_s)
    }

    fn terminal(&self) -> PyResult<bool> {
        Ok(self.require_state()?.current_time_s >= self.scenario.config.horizon_s)
    }

    /// Step one control interval, returning `(costs, reward, terminated)`.
    fn step(&mut self, action_index: usize) -> PyResult<(Vec<f64>, f64, bool)> {
        let scenario = &self.scenario;
        let reward_config = self.reward;
        let state = self
            .state
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("reset() must be called first"))?;
        let costs = engine::advance_interval(state, scenario, action_index).map_err(to_py)?;
        let reward = -interval_cost(&costs, &reward_config) / reward_config.n_ref;
        let terminated = state.current_time_s >= scenario.config.horizon_s;
        Ok((costs_list(&costs), reward, terminated))
    }

    /// Full state snapshot for golden-test comparison.
    fn debug_snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let state = self.require_state()?;
        let snapshot = full_snapshot(state);
        let dict = state_dict(py, &snapshot)?;
        match mean_waiting_minutes(state) {
            Some(value) => dict.set_item("mean_wait", value)?,
            None => dict.set_item("mean_wait", py.None())?,
        }
        Ok(dict)
    }

    /// Step one interval with per-tick snapshots and event deltas (test-only).
    fn debug_step<'py>(
        &mut self,
        py: Python<'py>,
        action_index: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let scenario = &self.scenario;
        let reward_config = self.reward;
        let state = self
            .state
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("reset() must be called first"))?;
        let mask = guards::valid_action_mask(state, scenario);
        if action_index >= mask.len() || !mask[action_index] {
            return Err(PyValueError::new_err(format!(
                "invalid action index: {action_index}"
            )));
        }
        let before_departures = state.departures.len();
        let before_actions = state.accepted_actions.len();
        let trace =
            engine::advance_interval_traced(state, scenario, action_index).map_err(to_py)?;

        let ticks = PyList::empty(py);
        for tick in &trace.ticks {
            let tick_dict = PyDict::new(py);
            tick_dict.set_item("time_s", tick.state.time_s)?;
            tick_dict.set_item("counters", PyArray1::from_slice(py, &tick.state.counters))?;
            tick_dict.set_item("vehicles", vehicle_matrix(py, &tick.state.vehicles)?)?;
            tick_dict.set_item("costs", costs_list(&tick.costs))?;
            tick_dict.set_item(
                "event",
                [
                    tick.event.time_s,
                    tick.event.boarded,
                    tick.event.denied,
                    tick.event.abandoned,
                ],
            )?;
            ticks.append(tick_dict)?;
        }

        let departures = PyList::empty(py);
        for event in &state.departures[before_departures..] {
            let event_dict = PyDict::new(py);
            event_dict.set_item("time_s", event.time_s)?;
            event_dict.set_item("route_id", event.route_id)?;
            event_dict.set_item("direction", event.direction)?;
            event_dict.set_item("bus_id", event.bus_id)?;
            if event.pattern_extra {
                event_dict.set_item("pattern", event.pattern.name())?;
            } else {
                event_dict.set_item("pattern", event.pattern.as_i64())?;
            }
            departures.append(event_dict)?;
        }

        let actions = PyList::empty(py);
        for event in &state.accepted_actions[before_actions..] {
            let event_dict = PyDict::new(py);
            event_dict.set_item("time_s", event.time_s)?;
            event_dict.set_item("kind", event.kind)?;
            if event.kind == "SET_HEADWAY" {
                event_dict.set_item("route_id", event.route_id)?;
            } else {
                event_dict.set_item("bus_id", event.bus_id)?;
                event_dict.set_item("route_id", event.route_id)?;
            }
            actions.append(event_dict)?;
        }

        let costs = &trace.costs;
        let reward = -interval_cost(costs, &reward_config) / reward_config.n_ref;
        let terminated = state.current_time_s >= scenario.config.horizon_s;

        let result = PyDict::new(py);
        result.set_item("costs", costs_list(costs))?;
        result.set_item("reward", reward)?;
        result.set_item("terminated", terminated)?;
        result.set_item("ticks", ticks)?;
        result.set_item("departures", departures)?;
        result.set_item("accepted_actions", actions)?;
        Ok(result)
    }
}

impl Kernel {
    fn require_state(&self) -> PyResult<&WorldState> {
        self.state
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("reset() must be called first"))
    }
}

#[pyfunction]
fn cost_fields() -> Vec<&'static str> {
    COST_FIELDS.to_vec()
}

#[pyfunction]
fn observation_placeholder() -> &'static str {
    "observation and masks are implemented in R2; this bridge exposes the R1 kernel"
}

#[pymodule]
fn bus_sim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Kernel>()?;
    m.add_function(wrap_pyfunction!(cost_fields, m)?)?;
    m.add_function(wrap_pyfunction!(observation_placeholder, m)?)?;
    Ok(())
}
