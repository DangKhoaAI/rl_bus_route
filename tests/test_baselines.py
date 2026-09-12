import numpy as np

from bus_rl.baselines import (
    FixedController,
    ProportionalController,
    RandomValidController,
    ThresholdController,
)
from bus_rl.control.actions import action_id
from bus_rl.env.bus_dispatch import BusDispatchEnv
from bus_rl.env.observation import observe
from tests.fixtures import empty_scenario, waiting_state


def test_all_baselines_return_valid_actions():
    scenario = empty_scenario()
    env = BusDispatchEnv([scenario], scenario.config)
    observation, _ = env.reset(seed=7)
    mask = env.action_masks()
    for controller in (
        FixedController(),
        RandomValidController(7),
        ThresholdController(),
        ProportionalController(),
    ):
        assert mask[controller.act(observation, mask)]


def test_threshold_prefers_short_reserve_when_prior_share_is_high():
    observation = observe(waiting_state(50, destination=2, stage="M3"), empty_scenario("M3"))
    mask = np.zeros(221, dtype=bool)
    mask[0] = True
    short = action_id("SHORT_TURN", 9, 0)
    dispatch = action_id("DISPATCH", 9, 0)
    mask[short] = mask[dispatch] = True
    observation["vehicles"][9, 0] = 1
    assert ThresholdController().act(observation, mask) == short
