from sb3_contrib import MaskablePPO
from torch import nn

from bus_rl.env.bus_dispatch import BusDispatchEnv
from bus_rl.models.features import POLICY_KWARGS
from bus_rl.training.train import make_model
from tests.fixtures import empty_scenario


def test_policy_architecture_is_shared_mlp_with_actor_critic_heads():
    env = BusDispatchEnv([empty_scenario()], empty_scenario().config)
    model = make_model(env, seed=11)
    shared = [
        layer.out_features
        for layer in model.policy.features_extractor.mlp
        if isinstance(layer, nn.Linear)
    ]
    actor = [
        layer.out_features
        for layer in model.policy.mlp_extractor.policy_net
        if isinstance(layer, nn.Linear)
    ]
    critic = [
        layer.out_features
        for layer in model.policy.mlp_extractor.value_net
        if isinstance(layer, nn.Linear)
    ]
    assert shared == [256, 128]
    assert actor == [128]
    assert critic == [128]
    assert model.policy.net_arch == POLICY_KWARGS["net_arch"]


def test_masked_ppo_save_load_preserves_action(tmp_path):
    scenario = empty_scenario()
    env = BusDispatchEnv([scenario], scenario.config)
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        gamma=1.0,
        n_steps=16,
        batch_size=16,
        n_epochs=1,
        seed=11,
        device="cpu",
        verbose=0,
        policy_kwargs=POLICY_KWARGS,
    )
    model.learn(total_timesteps=32)
    observation, _ = env.reset(seed=7)
    mask = env.action_masks()
    action, _ = model.predict(observation, action_masks=mask, deterministic=True)
    assert mask[int(action)]
    model.save(tmp_path / "smoke")
    restored = MaskablePPO.load(tmp_path / "smoke", env=env)
    other, _ = restored.predict(observation, action_masks=mask, deterministic=True)
    assert int(action) == int(other)
