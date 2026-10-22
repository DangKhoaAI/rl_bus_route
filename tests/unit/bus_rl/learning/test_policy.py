import numpy as np
import pytest
import torch
from gymnasium import spaces

from bus_rl.config import AlgorithmConfig
from bus_rl.learning.policy import arrival_rate_belief, policy_kwargs


def observation():
    return {
        "arrival_history": torch.zeros(2, 4, 2, 8, 5),
        "stop_valid": torch.ones(2, 4, 2, 8),
        "context": torch.zeros(2, 3),
    }


def test_belief_counts_only_elapsed_bins_and_masks_padding():
    obs = observation()
    # First row is reset: the posterior is still the Gamma(1,1) prior.
    # Second row has two observed bins with counts 2 and 4: Gamma(7,3).
    obs["context"][1, 0] = 2 / 120
    obs["arrival_history"][1, 0, 0, 0, :2] = torch.tensor([2, 4]) / 40
    obs["arrival_history"][..., 2:] = 100  # unavailable bins must not count
    obs["stop_valid"][:, 3] = 0
    belief = arrival_rate_belief(obs, 120)
    assert belief[0, 0].item() == pytest.approx(1 / 40)
    assert belief[0, 64].item() == pytest.approx(1 / 40)
    assert belief[1, 0].item() == pytest.approx(7 / 3 / 40)
    assert belief[1, 64].item() == pytest.approx(7**0.5 / 3 / 40)
    assert torch.count_nonzero(belief[:, 48:64]) == 0
    assert torch.count_nonzero(belief[:, 112:128]) == 0


def test_gru_reads_oldest_first_and_is_independent_of_batch_order():
    obs = observation()
    obs["arrival_history"][0, 0, 0, 0] = torch.arange(5)
    obs["arrival_history"][1, 0, 0, 0] = torch.arange(5) + 5
    space = spaces.Dict(
        {
            key: spaces.Box(-np.inf, np.inf, value.shape[1:], np.float32)
            for key, value in obs.items()
        }
    )
    kwargs = policy_kwargs("belief_gru")
    extractor = kwargs["features_extractor_class"](space, **kwargs["features_extractor_kwargs"])
    inputs = []
    hook = extractor.gru.register_forward_pre_hook(lambda module, args: inputs.append(args[0]))
    output = extractor(obs)
    assert torch.equal(inputs[0][0, :, 0], torch.tensor([4, 3, 2, 1, 0]))
    reordered = extractor({key: value.flip(0) for key, value in obs.items()})
    torch.testing.assert_close(output, reordered.flip(0))
    output.square().mean().backward()
    assert extractor.gru.weight_ih_l0.grad.abs().sum() > 0
    hook.remove()


def test_unknown_encoder_fails_instead_of_silently_using_core():
    with pytest.raises(ValueError, match="encoder"):
        AlgorithmConfig(encoder="lstm_typo")


def test_extended_belief_uses_long_history_prior_and_discounted_exposure():
    obs = observation()
    obs["context"][:, 0] = 1
    obs["arrival_memory"] = torch.zeros(2, 4, 2, 8, 10)
    obs["arrival_memory"][:, 0, 0, 0, 0] = 2 / 40
    obs["arrival_memory"][:, 0, 0, 0, 8] = 4 / 40
    belief = arrival_rate_belief(obs, 120, prior_shape=0.1, prior_rate=0.1, decay=0.8)
    alpha = 0.1 + 2 + 4 * 0.8**8
    beta = 0.1 + sum(0.8**lag for lag in range(10))
    assert belief[0, 0].item() == pytest.approx(alpha / beta / 40)
    assert belief[0, 64].item() == pytest.approx(alpha**0.5 / beta / 40)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"history_minutes": 0},
        {"history_minutes": 20.5},
        {"belief_decay": 1.1},
        {"belief_prior_rate": 0},
        {"belief_prior_shape": float("nan")},
    ],
)
def test_invalid_memory_belief_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        AlgorithmConfig(**kwargs)
