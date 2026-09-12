from bus_rl.baselines import (
    FixedController,
    ProportionalController,
    RandomValidController,
    ThresholdController,
)
from bus_rl.env.bus_dispatch import BusDispatchEnv
from tests.fixtures import empty_scenario


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
