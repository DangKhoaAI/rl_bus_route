"""Process-level runtime settings shared by train, eval and benchmark.

Torch thread count is a global process setting; keeping it in one helper makes
train/eval/benchmark use the same value and lets it be recorded in metadata.
The same hook disables Torch distribution argument validation: validation only
checks inputs and raises, and the maskable categorical's simplex/``all``/``eq``
checks are a measurable part of every policy forward, while our action masks
are validated at the env/native boundary.
"""

from __future__ import annotations


def configure_torch_distributions() -> None:
    """Skip Torch distribution argument validation (bit-identical, faster).

    Validation does not change sampled values, log-probs or logits; it only
    raises on malformed inputs. Masks are validated by ``BatchKernel`` /
    ``valid_action_mask`` before they reach the policy.
    """
    import torch

    torch.distributions.Distribution.set_default_validate_args(False)


def apply_torch_threads(threads: int) -> int:
    """Set process-level Torch runtime settings and return the thread count."""
    import torch

    configure_torch_distributions()
    value = max(1, int(threads))
    torch.set_num_threads(value)
    return int(torch.get_num_threads())
