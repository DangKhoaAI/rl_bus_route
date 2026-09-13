"""Unit tests for the evaluation statistics helpers."""

from __future__ import annotations

import numpy as np

from bus_rl.evaluation.statistics import paired_bootstrap


def test_paired_bootstrap_constant_difference_is_exact():
    left = np.arange(10.0) + 2.0
    right = np.arange(10.0)
    mean, low, high = paired_bootstrap(left, right, n_resamples=2_000, seed=6_001)
    assert mean == 2.0
    assert low == 2.0
    assert high == 2.0
