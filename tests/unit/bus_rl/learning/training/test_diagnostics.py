import numpy as np
import pytest

from bus_rl.learning.training.diagnostics import (
    distribution_summary,
    finite_check,
    mask_metrics,
    probability_metrics,
)


def test_mask_metrics_marks_unavailable_family_as_null():
    mask = np.zeros(221, dtype=bool)
    mask[[0, 1, 2]] = True
    row = mask_metrics(mask, action=1)
    assert row["valid_action_count"] == 3
    assert row["families"]["NOOP"]["conditional_selection_rate"] == 0.0
    assert row["families"]["DISPATCH"]["conditional_selection_rate"] == 1.0
    assert row["families"]["REASSIGN"]["conditional_selection_rate"] is None


def test_probability_metrics_handles_k_one_and_masked_zero_mass():
    mask = np.zeros(221, dtype=bool)
    mask[0] = True
    probabilities = np.zeros(221, dtype=np.float64)
    probabilities[0] = 1.0
    row = probability_metrics(probabilities, mask, action=0)
    assert row["valid_action_count"] == 1
    assert row["entropy"] == 0.0
    assert row["normalized_entropy"] is None
    assert row["families"]["NOOP"]["probability_mass"] == 1.0


def test_invalid_masks_and_nonfinite_probabilities_fail_explicitly():
    with pytest.raises(ValueError, match="K=0"):
        mask_metrics(np.zeros(221, dtype=bool))
    mask = np.zeros(221, dtype=bool)
    mask[0] = True
    bad = np.zeros(221, dtype=np.float64)
    bad[0] = np.nan
    with pytest.raises(FloatingPointError, match="non-finite"):
        probability_metrics(bad, mask)


def test_finite_and_distribution_summaries_are_serializable():
    finite = finite_check("x", np.array([1.0, 2.0]))
    summary = distribution_summary("x", np.array([1.0, 2.0, 3.0]))
    assert finite["finite"] is True
    assert summary["q50"] == 2.0
    assert summary["nonfinite_count"] == 0
