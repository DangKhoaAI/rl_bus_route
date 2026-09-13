"""Process-level runtime settings shared by train, eval and benchmark.

Torch thread count and Torch distribution argument validation are global
process settings; keeping them in one helper makes train/eval/benchmark use the
same effective values and lets them be recorded in metadata.

Validation is **on by default** (the historical behavior). Disabling it is a
bit-identical speed opt-in: validation only checks inputs and raises, but a
valid action mask does not by itself guarantee finite logits/probabilities, so
turning it off can surface arithmetic problems later. The accepted runtime
config sets it off explicitly and metadata records the effective value.
"""

from __future__ import annotations


def configure_torch_distributions(validate: bool) -> bool:
    """Set Torch distribution argument validation and return the effective value."""
    import torch

    torch.distributions.Distribution.set_default_validate_args(bool(validate))
    return bool(torch.distributions.Distribution._validate_args)


def distribution_validation_enabled() -> bool:
    """Return the process-wide Torch distribution validation setting."""
    import torch

    return bool(torch.distributions.Distribution._validate_args)


def apply_torch_threads(threads: int) -> int:
    """Set the Torch CPU thread count and return the value in effect."""
    import torch

    value = max(1, int(threads))
    torch.set_num_threads(value)
    return int(torch.get_num_threads())


def apply_runtime_settings(run) -> dict:
    """Apply ``run.runtime`` to the process and return requested/effective values."""
    import torch

    threads = apply_torch_threads(run.algorithm.torch_threads)
    requested = bool(run.runtime.validate_distributions)
    effective = configure_torch_distributions(requested)
    return {
        "torch_threads_requested": int(run.algorithm.torch_threads),
        "torch_threads_actual": threads,
        "validate_distributions_requested": requested,
        "validate_distributions_actual": effective,
        "torch_distribution_validate_args": bool(
            torch.distributions.Distribution._validate_args
        ),
    }
