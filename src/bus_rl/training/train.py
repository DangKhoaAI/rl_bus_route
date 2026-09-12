from sb3_contrib import MaskablePPO

from bus_rl.models.features import POLICY_KWARGS


def make_model(env, seed: int):
    return MaskablePPO(
        "MultiInputPolicy", env, gamma=1.0, seed=seed, device="cpu", policy_kwargs=POLICY_KWARGS
    )
