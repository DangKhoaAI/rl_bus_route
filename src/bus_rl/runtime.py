"""Process-level runtime settings shared by train, eval and benchmark.

Torch thread count is a global process setting; keeping it in one helper makes
train/eval/benchmark use the same value and lets it be recorded in metadata.
"""

from __future__ import annotations


def apply_torch_threads(threads: int) -> int:
    """Set the Torch CPU thread count and return the value in effect."""
    import torch

    value = max(1, int(threads))
    torch.set_num_threads(value)
    return int(torch.get_num_threads())
