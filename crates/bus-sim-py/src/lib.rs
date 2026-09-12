//! PyO3 bridge exposing the native kernel to the Python integration layer.
//!
//! Module name: `bus_sim`. R1/R2 expose state, observations and the mask cache;
//! R3 adds the per-decision contract (`reset_contract`/`step_contract`) plus the
//! evaluator summary/trace surfaces used by `bus_rl.env.native_bus_dispatch`.
//! The Gym wrapper itself lives in Python.

use bus_sim_core::costs::{interval_cost, mean_waiting_minutes, RewardConfig};
use bus_sim_core::domain::{Scenario, StepCosts, WorldState};
use bus_sim_core::observation::{self, Observation};
use bus_sim_core::snapshot::{full_snapshot, StateSnapshot};
use bus_sim_core::{engine, guards, initial_state, scenario_from_json, KernelError};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray5,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::sync::Arc;

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

/// Shaped observation tensors, matching `bus_rl.env.observation.observe`.
fn obs_dict<'py>(py: Python<'py>, obs: Observation) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(
        "stops",
        PyArray1::from_vec(py, obs.stops).reshape([
            observation::MAX_ROUTES,
            observation::MAX_DIRECTIONS,
            observation::MAX_STOPS,
            observation::STOP_CHANNELS,
        ])?,
    )?;
    dict.set_item(
        "arrival_history",
        PyArray1::from_vec(py, obs.arrival_history).reshape([
            observation::MAX_ROUTES,
            observation::MAX_DIRECTIONS,
            observation::MAX_STOPS,
            observation::HISTORY_LAGS,
        ])?,
    )?;
    dict.set_item(
        "forecast",
        PyArray1::from_vec(py, obs.forecast).reshape([
            observation::MAX_ROUTES,
            observation::MAX_DIRECTIONS,
            observation::MAX_STOPS,
        ])?,
    )?;
    dict.set_item(
        "vehicles",
        PyArray1::from_vec(py, obs.vehicles)
            .reshape([observation::MAX_VEHICLES, observation::VEHICLE_FEATURES])?,
    )?;
    dict.set_item(
        "routes",
        PyArray1::from_vec(py, obs.routes)
            .reshape([observation::MAX_ROUTES, observation::ROUTE_FEATURES])?,
    )?;
    dict.set_item(
        "stop_valid",
        PyArray1::from_vec(py, obs.stop_valid).reshape([
            observation::MAX_ROUTES,
            observation::MAX_DIRECTIONS,
            observation::MAX_STOPS,
        ])?,
    )?;
    dict.set_item("vehicle_valid", PyArray1::from_vec(py, obs.vehicle_valid))?;
    dict.set_item("route_valid", PyArray1::from_vec(py, obs.route_valid))?;
    dict.set_item("context", PyArray1::from_vec(py, obs.context))?;
    Ok(dict)
}

#[pyclass]
struct Kernel {
    scenario: Arc<Scenario>,
    state: Option<WorldState>,
    reward: RewardConfig,
    conservation_checks: bool,
    /// Guard-derived mask owned by the kernel; `action_mask()` returns a copy.
    mask_cache: Option<Vec<bool>>,
    mask_computations: u64,
}

/// Owns packed scenarios once so many kernels (envs) can share the immutable
/// tapes instead of copying them per env.
#[pyclass]
struct ScenarioStore {
    scenarios: Vec<Arc<Scenario>>,
}

fn build_scenario(
    scenario_json: &str,
    arrivals: &PyReadonlyArray5<i32>,
    traffic: &PyReadonlyArray2<f32>,
) -> PyResult<Scenario> {
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
    scenario_from_json(
        scenario_json,
        arrival_tape,
        arrival_dims,
        traffic_tape,
        traffic_dims,
    )
    .map_err(to_py)
}

#[pymethods]
impl ScenarioStore {
    #[new]
    fn new() -> Self {
        Self {
            scenarios: Vec::new(),
        }
    }

    /// Pack one scenario and return its reusable index. Tapes are copied once.
    fn add(
        &mut self,
        scenario_json: &str,
        arrivals: PyReadonlyArray5<i32>,
        traffic: PyReadonlyArray2<f32>,
    ) -> PyResult<usize> {
        let scenario = build_scenario(scenario_json, &arrivals, &traffic)?;
        self.scenarios.push(Arc::new(scenario));
        Ok(self.scenarios.len() - 1)
    }

    fn __len__(&self) -> usize {
        self.scenarios.len()
    }
}

#[pymethods]
impl Kernel {
    #[new]
    fn new(
        scenario_json: &str,
        arrivals: PyReadonlyArray5<i32>,
        traffic: PyReadonlyArray2<f32>,
    ) -> PyResult<Self> {
        Ok(Self {
            scenario: Arc::new(build_scenario(scenario_json, &arrivals, &traffic)?),
            state: None,
            reward: RewardConfig::default(),
            conservation_checks: false,
            mask_cache: None,
            mask_computations: 0,
        })
    }

    /// Create a kernel that shares a packed scenario from a store (env reuse).
    #[staticmethod]
    fn from_store(store: PyRef<'_, ScenarioStore>, index: usize) -> PyResult<Self> {
        let scenario = store
            .scenarios
            .get(index)
            .ok_or_else(|| PyValueError::new_err(format!("scenario index {index} out of range")))?
            .clone();
        Ok(Self {
            scenario,
            state: None,
            reward: RewardConfig::default(),
            conservation_checks: false,
            mask_cache: None,
            mask_computations: 0,
        })
    }

    fn reset(&mut self) -> PyResult<()> {
        let mut state = initial_state(&self.scenario).map_err(to_py)?;
        state.conservation_checks = self.conservation_checks;
        // Compute the initial mask once here; later reads reuse the cache.
        self.mask_cache = Some(guards::valid_action_mask(&state, &self.scenario));
        self.mask_computations += 1;
        self.state = Some(state);
        Ok(())
    }

    /// Contract reset: `{obs, mask}` for the first decision boundary.
    fn reset_contract<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.reset()?;
        let state = self.require_state()?;
        let obs = observation::observe(state, &self.scenario);
        let dict = PyDict::new(py);
        dict.set_item("obs", obs_dict(py, obs)?)?;
        dict.set_item(
            "mask",
            PyArray1::from_slice(py, self.mask_cache.as_ref().expect("mask set by reset")),
        )?;
        Ok(dict)
    }

    fn set_conservation_checks(&mut self, enabled: bool) {
        self.conservation_checks = enabled;
        if let Some(state) = self.state.as_mut() {
            state.conservation_checks = enabled;
        }
    }

    fn action_mask<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let mask = self
            .mask_cache
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("reset() must be called first"))?;
        // `from_slice` copies, so callers cannot mutate the native cache.
        Ok(PyArray1::from_slice(py, mask))
    }

    /// Number of times the mask has been (re)computed by a mutation. Repeated
    /// `action_mask()` reads do not change it.
    #[getter]
    fn mask_computations(&self) -> u64 {
        self.mask_computations
    }

    /// Recompute the mask from the current state and compare it to the cache.
    fn debug_validate_mask(&self) -> PyResult<bool> {
        let state = self.require_state()?;
        let fresh = guards::valid_action_mask(state, &self.scenario);
        Ok(self.mask_cache.as_ref() == Some(&fresh))
    }

    /// Observation tensors at the current boundary (R2 parity surface).
    fn observe<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let state = self.require_state()?;
        obs_dict(py, observation::observe(state, &self.scenario))
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
        self.validate_cached(action_index)?;
        let scenario = &self.scenario;
        let reward_config = self.reward;
        let state = self
            .state
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("reset() must be called first"))?;
        let costs = engine::advance_interval(state, scenario, action_index).map_err(to_py)?;
        let reward = -interval_cost(&costs, &reward_config) / reward_config.n_ref;
        let terminated = state.current_time_s >= scenario.config.horizon_s;
        let next_mask = guards::valid_action_mask(state, scenario);
        self.mask_cache = Some(next_mask);
        self.mask_computations += 1;
        Ok((costs_list(&costs), reward, terminated))
    }

    /// Contract step: `{obs, reward, terminated, truncated, mask, costs}`.
    fn step_contract<'py>(
        &mut self,
        py: Python<'py>,
        action_index: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.validate_cached(action_index)?;
        let scenario = &self.scenario;
        let reward_config = self.reward;
        let state = self
            .state
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("reset() must be called first"))?;
        let costs = engine::advance_interval(state, scenario, action_index).map_err(to_py)?;
        let reward = -interval_cost(&costs, &reward_config) / reward_config.n_ref;
        let terminated = state.current_time_s >= scenario.config.horizon_s;
        let next_mask = guards::valid_action_mask(state, scenario);
        let obs = observation::observe(state, scenario);
        self.mask_cache = Some(next_mask);
        self.mask_computations += 1;

        let dict = PyDict::new(py);
        dict.set_item("obs", obs_dict(py, obs)?)?;
        dict.set_item(
            "mask",
            PyArray1::from_slice(py, self.mask_cache.as_ref().expect("mask just set")),
        )?;
        dict.set_item("reward", reward)?;
        dict.set_item("terminated", terminated)?;
        dict.set_item("truncated", false)?;
        dict.set_item("costs", costs_list(&costs))?;
        Ok(dict)
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

    /// Episode-end inputs for the Python evaluator: counters, cohort table,
    /// departures and accepted action kinds. Built once at episode end.
    fn episode_summary_inputs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let state = self.require_state()?;
        let dict = state_dict(py, &full_snapshot(state))?;
        let departures = PyList::empty(py);
        for event in &state.departures {
            let row = PyDict::new(py);
            row.set_item("time_s", event.time_s)?;
            row.set_item("route_id", event.route_id)?;
            row.set_item("direction", event.direction)?;
            row.set_item("bus_id", event.bus_id)?;
            if event.pattern_extra {
                row.set_item("pattern", event.pattern.name())?;
            } else {
                row.set_item("pattern", event.pattern.as_i64())?;
            }
            departures.append(row)?;
        }
        dict.set_item("departures", departures)?;
        let kinds = PyList::empty(py);
        for event in &state.accepted_actions {
            kinds.append(event.kind)?;
        }
        dict.set_item("accepted_kinds", kinds)?;
        Ok(dict)
    }

    /// Per-decision trace row for `evaluation/runner.py`.
    fn trace_snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let state = self.require_state()?;
        let queues: Vec<i64> = (0..self.scenario.config.route_count)
            .map(|route| {
                state
                    .cohorts
                    .iter()
                    .filter(|cohort| cohort.route_id as usize == route)
                    .map(|cohort| cohort.count)
                    .sum()
            })
            .collect();
        let buses = PyList::empty(py);
        for vehicle in &state.vehicles {
            let row = PyDict::new(py);
            row.set_item("id", vehicle.vehicle_id)?;
            row.set_item("phase", vehicle.phase.name())?;
            row.set_item("route_id", vehicle.route_id)?;
            row.set_item("pattern", vehicle.pattern.name())?;
            row.set_item("load", vehicle.load)?;
            buses.append(row)?;
        }
        let dict = PyDict::new(py);
        dict.set_item("time_s", state.current_time_s)?;
        dict.set_item("queues", queues)?;
        dict.set_item(
            "headway_targets",
            PyArray1::from_slice(py, &state.headway_targets_s),
        )?;
        dict.set_item("buses", buses)?;
        dict.set_item("waiting", state.waiting_count())?;
        dict.set_item("onboard", state.onboard_count())?;
        dict.set_item("generated", state.generated_count())?;
        dict.set_item("abandoned", state.abandoned_count())?;
        dict.set_item("completed", state.completed_count())?;
        Ok(dict)
    }

    /// Step one interval with per-tick snapshots and event deltas (test-only).
    fn debug_step<'py>(
        &mut self,
        py: Python<'py>,
        action_index: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.validate_cached(action_index)?;
        let scenario = &self.scenario;
        let reward_config = self.reward;
        let state = self
            .state
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("reset() must be called first"))?;
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
        let next_mask = guards::valid_action_mask(state, scenario);

        let result = PyDict::new(py);
        result.set_item("costs", costs_list(costs))?;
        result.set_item("reward", reward)?;
        result.set_item("terminated", terminated)?;
        result.set_item("ticks", ticks)?;
        result.set_item("departures", departures)?;
        result.set_item("accepted_actions", actions)?;
        self.mask_cache = Some(next_mask);
        self.mask_computations += 1;
        Ok(result)
    }
}

impl Kernel {
    fn require_state(&self) -> PyResult<&WorldState> {
        self.state
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("reset() must be called first"))
    }

    /// Reject masked/out-of-range actions before any mutation.
    fn validate_cached(&self, action_index: usize) -> PyResult<()> {
        let mask = self
            .mask_cache
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("reset() must be called first"))?;
        if action_index >= mask.len() || !mask[action_index] {
            return Err(PyValueError::new_err(format!(
                "invalid action index: {action_index}"
            )));
        }
        Ok(())
    }
}

#[pyfunction]
fn cost_fields() -> Vec<&'static str> {
    COST_FIELDS.to_vec()
}

#[pymodule]
fn bus_sim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Kernel>()?;
    m.add_class::<ScenarioStore>()?;
    m.add_function(wrap_pyfunction!(cost_fields, m)?)?;
    Ok(())
}
