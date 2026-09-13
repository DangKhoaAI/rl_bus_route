"""R3.1 native Gym wrapper contract: shapes, ownership and env isolation.

Skipped when the `bus_sim_native` extension has not been built
(`python scripts/build_native.py`).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bus_rl.config import ControlConfig, RuntimeConfig, load_run_config
from bus_rl.execution.environments.factory import make_env_for_run
from bus_rl.execution.environments.python.environment import BusDispatchEnv
from bus_rl.execution.scenarios.generation import generate_scenario
from bus_sim.oracle.domain import SimConfig, StepCosts
from bus_sim.parity.fixtures import load_fixture
from bus_sim.parity.scenarios import CONTROL_FLAGS
from tests.support.compare import assert_obs, assert_value
from tests.support.paths import GOLDEN, ROOT
from tests.support.rust_bridge import cached_scenario

pytest.importorskip("bus_sim_native")

from bus_rl.execution.environments.rust.environment import NativeBusDispatchEnv

pytestmark = pytest.mark.native


def _control(spec: dict) -> ControlConfig:
    enable_reassign, enable_short_turn = CONTROL_FLAGS[spec.get("control", "M3")]
    return ControlConfig(enable_reassign=enable_reassign, enable_short_turn=enable_short_turn)


def test_native_env_contract_and_masked_rejection():
    import gymnasium as gym

    scenario = generate_scenario(2001)
    env = NativeBusDispatchEnv([scenario], scenario.config, control=ControlConfig())
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 221
    observation, info = env.reset(seed=5)
    assert info == {}
    assert env.observation_space.contains(observation)
    masked = int(np.flatnonzero(~env.action_masks())[0])
    with pytest.raises(ValueError):
        env.step(masked)
    terminated = False
    decisions = scenario.config.horizon_s // scenario.config.control_interval_s
    for _ in range(decisions):
        observation, reward, terminated, truncated, info = env.step(0)
        assert observation["stops"].shape == (4, 2, 8, 7)
        assert isinstance(info["costs"], StepCosts)
        assert np.isfinite(reward)
        assert not truncated
    assert terminated


@pytest.mark.parametrize(
    "name",
    ["zero_m3", "normal_m3", "capacity_m3", "abandon_m3", "burst_m3", "traffic_m3"],
)
def test_native_env_matches_python_env_at_every_boundary(name):
    payload, _, _ = load_fixture(GOLDEN / name)
    scenario = cached_scenario(payload["spec"])
    control = _control(payload["spec"])
    oracle = BusDispatchEnv([scenario], scenario.config, control=control)
    native = NativeBusDispatchEnv([scenario], scenario.config, control=control)

    observation_py, _ = oracle.reset(seed=0, options={"scenario_index": 0})
    observation_rs, _ = native.reset(seed=0, options={"scenario_index": 0})
    assert_obs(observation_py, observation_rs)
    assert np.array_equal(oracle.action_masks(), native.action_masks())

    for action in payload["actions"]:
        step_py = oracle.step(int(action))
        step_rs = native.step(int(action))
        assert_obs(step_py[0], step_rs[0])
        np.testing.assert_allclose(step_py[1], step_rs[1], rtol=1e-9, atol=1e-9)
        assert step_py[2] == step_rs[2]
        assert step_py[3] == step_rs[3]
        for field in StepCosts.__dataclass_fields__:
            assert_value(
                getattr(step_py[4]["costs"], field),
                getattr(step_rs[4]["costs"], field),
                field,
            )
        assert np.array_equal(oracle.action_masks(), native.action_masks())
        if step_py[2]:
            break


def test_returned_arrays_are_owned_and_survive_later_steps():
    scenario = generate_scenario(2001)
    env = NativeBusDispatchEnv([scenario], scenario.config, control=ControlConfig())
    observation, _ = env.reset(seed=0)
    saved_obs = {key: np.array(value, copy=True) for key, value in observation.items()}
    for _ in range(3):
        env.step(0)
    env.reset(seed=1)
    # Saved rollout inputs are untouched by later native operations.
    for key, value in saved_obs.items():
        np.testing.assert_array_equal(value, observation[key])
    # Mutating a returned mask cannot corrupt the native cache.
    scratch = env.action_masks()
    scratch[:] = False
    assert env.action_masks().any()


def test_interleaved_native_envs_do_not_share_state():
    scenario = generate_scenario(2001)
    first = NativeBusDispatchEnv([scenario], scenario.config, control=ControlConfig())
    second = NativeBusDispatchEnv([scenario], scenario.config, control=ControlConfig())
    solo = NativeBusDispatchEnv([scenario], scenario.config, control=ControlConfig())
    first.reset(seed=0)
    second.reset(seed=0)
    solo.reset(seed=0)
    for _ in range(5):
        left = first.step(0)
        right = second.step(0)
        reference = solo.step(0)
        assert_obs(left[0], reference[0])
        assert_obs(right[0], reference[0])
        assert left[1] == right[1] == reference[1]


def test_native_envs_share_one_scenario_store():
    scenarios = [generate_scenario(2001), generate_scenario(2002), generate_scenario(2003)]
    envs = [
        NativeBusDispatchEnv(scenarios, scenarios[0].config, control=ControlConfig())
        for _ in range(3)
    ]
    assert envs[0]._scenario_store is envs[1]._scenario_store is envs[2]._scenario_store
    assert len(envs[0]._scenario_store) == 3
    # Kernels are built lazily: one active kernel per env (independent episode
    # state) while every env shares the packed tapes.
    assert envs[0].kernel is None
    envs[0].reset(seed=0, options={"scenario_index": 0})
    envs[1].reset(seed=0, options={"scenario_index": 0})
    assert envs[0].kernel is not envs[1].kernel
    envs[0].step(0)
    assert envs[0].kernel.current_time_s != envs[1].kernel.current_time_s


def test_metadata_only_scenarios_load_tapes_from_disk(tmp_path):
    from bus_rl.execution.environments.rust.bridge import _scenario_tapes
    from bus_rl.execution.scenarios.io import load_split, save_manifest

    scenario = generate_scenario(2001)
    manifest = save_manifest(tmp_path / "data", {"train": (scenario,)}, scenario.config)
    full = load_split(manifest, "train")[0]
    meta = load_split(manifest, "train", with_tapes=False)[0]
    # Rust runs hold metadata only; the store loads the tape once from `path`.
    assert meta.arrival_tape.size == 0 and meta.traffic_tape.size == 0
    assert meta.path is not None
    assert meta.scenario_hash == full.scenario_hash
    arrivals, traffic = _scenario_tapes(meta)
    np.testing.assert_array_equal(arrivals, full.arrival_tape)
    np.testing.assert_array_equal(traffic, full.traffic_tape)


def test_action_masks_reuse_the_step_mask_without_recompute():
    scenario = generate_scenario(2001)
    env = NativeBusDispatchEnv([scenario], scenario.config, control=ControlConfig())
    env.reset(seed=0)
    before = env.kernel.mask_computations
    for _ in range(5):
        env.action_masks()
    assert env.kernel.mask_computations == before
    env.step(0)
    assert env.kernel.mask_computations == before + 1
    for _ in range(5):
        env.action_masks()
    assert env.kernel.mask_computations == before + 1


def test_disabling_validation_preserves_observations():
    scenario = generate_scenario(2001)
    run = replace(
        load_run_config(ROOT / "configs" / "experiments" / "core.toml", ROOT),
        control=ControlConfig(),
        runtime=RuntimeConfig(backend="rust", validate_observation=False),
    )
    env = make_env_for_run([scenario], run)
    observation, _ = env.reset(seed=0, options={"scenario_index": 0})
    for key, value in observation.items():
        assert np.isfinite(value).all(), key
    oracle = BusDispatchEnv([scenario], scenario.config, control=ControlConfig())
    reference, _ = oracle.reset(seed=0, options={"scenario_index": 0})
    for key in observation:
        np.testing.assert_allclose(observation[key], reference[key], rtol=1e-6, atol=1e-6)


def test_dummy_vec_env_auto_reset():
    from stable_baselines3.common.vec_env import DummyVecEnv

    config = SimConfig(horizon_s=480, demand_end_s=240)
    scenario = generate_scenario(2001, config)

    def make(seed: int):
        def _init():
            env = NativeBusDispatchEnv([scenario], config, control=ControlConfig())
            env.reset(seed=seed)
            return env

        return _init

    vec = DummyVecEnv([make(1), make(2)])
    try:
        observation = vec.reset()
        assert observation["stops"].shape == (2, 4, 2, 8, 7)
        for _ in range(6):
            observation, rewards, _, _ = vec.step(np.zeros(2, dtype=np.int64))
            assert np.isfinite(rewards).all()
            assert observation["stops"].shape == (2, 4, 2, 8, 7)
    finally:
        vec.close()
