import numpy as np

from bus_rl.execution.environments.python.environment import BusDispatchEnv
from bus_rl.learning.baselines import (
    FixedController,
    ProportionalController,
    RandomValidController,
    ThresholdController,
)
from bus_sim.oracle.actions import action_id
from bus_sim.oracle.domain import initial_state
from bus_sim.oracle.observation import observe
from tests.fixtures import empty_scenario, waiting_state


def initial_state_empty():
    return initial_state(empty_scenario())


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


def test_threshold_follows_headway_recall_and_lowest_urgency_donor():
    observation = observe(waiting_state(90), empty_scenario())
    mask = np.zeros(221, dtype=bool)
    mask[0] = True
    six = action_id("SET_HEADWAY", route_id=0, headway_s=360)
    mask[six] = True
    assert ThresholdController().act(observation, mask) == six
    empty = observe(initial_state_empty(), empty_scenario())
    recall = action_id("RECALL", 0)
    mask = np.zeros(221, dtype=bool)
    mask[0] = True
    mask[recall] = True
    assert ThresholdController().act(empty, mask) == recall
    observation = observe(waiting_state(50, stage="M2"), empty_scenario("M2"))
    donor_low = action_id("REASSIGN", 6, 0)
    donor_high = action_id("REASSIGN", 3, 0)
    mask = np.zeros(221, dtype=bool)
    mask[0] = mask[donor_low] = mask[donor_high] = True
    observation["stops"][1, :, :, 2] = 0.1
    observation["stops"][2, :, :, 2] = 0.0
    assert ThresholdController().act(observation, mask) == donor_low


def test_proportional_dispatches_shortage_and_reassigns_surplus_via_mask():
    scenario = empty_scenario("M2")
    observation = observe(waiting_state(130, stage="M2"), scenario)
    observation["arrival_history"][0] = 0.4
    observation["arrival_history"][1] = 0.1
    observation["arrival_history"][2] = 0.1
    observation["routes"][0, 3] = 2 / 16
    observation["routes"][1, 3] = 5 / 16
    observation["routes"][2, 3] = 2 / 16
    mask = np.zeros(221, dtype=bool)
    mask[0] = True
    dispatch = action_id("DISPATCH", 9, 0)
    mask[dispatch] = True
    observation["vehicles"][9, 0] = 1
    assert ProportionalController().act(observation, mask) == dispatch
    mask[dispatch] = False
    reassign = action_id("REASSIGN", 3, 0)
    mask[reassign] = True
    observation["vehicles"][3, 0] = 0
    observation["vehicles"][3, 8] = 1
    assert ProportionalController().act(observation, mask) == reassign
