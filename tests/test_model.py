from sb3_contrib import MaskablePPO

from bus_rl.env.bus_dispatch import BusDispatchEnv
from tests.fixtures import empty_scenario


def test_masked_ppo_save_load_preserves_valid_action(tmp_path):
    scenario = empty_scenario()
    env = BusDispatchEnv([scenario], scenario.config)
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        gamma=1.0,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        seed=11,
        device="cpu",
    )
    model.learn(total_timesteps=16)
    observation, _ = env.reset(seed=7)
    mask = env.action_masks()
    action, _ = model.predict(observation, action_masks=mask, deterministic=True)
    assert mask[int(action)]
    model.save(tmp_path / "smoke")
    restored = MaskablePPO.load(tmp_path / "smoke", env=env)
    other, _ = restored.predict(observation, action_masks=mask, deterministic=True)
    assert mask[int(other)]
