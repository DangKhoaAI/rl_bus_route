import numpy as np
import pandas as pd
import pytest

from bus_rl.evaluation.report_data import compare_baselines, summarize_training, validate_results


def _paired_rows(multiplier=0.8):
    rows = []
    for day, cost in enumerate([100.0, 200.0, 300.0, 400.0]):
        common = {
            "split": "test_id",
            "scenario_seed": str(2**63 + day),
            "scenario_hash": f"day-{day}",
            "mean_wait": 4.0,
        }
        rows.append({**common, "method": "fixed", "model_seed": None, "total_cost_core": cost})
        for seed, offset in [(11, -5), (22, 5)]:
            rows.append(
                {
                    **common,
                    "method": "ppo",
                    "model_seed": str(seed),
                    "total_cost_core": multiplier * cost + offset,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    "multiplier,saving,status", [(0.8, 20.0, "rl_lower_cost"), (1.2, -20.0, "rl_higher_cost")]
)
def test_paired_reductions_average_seeds_and_bootstrap_the_ratio(multiplier, saving, status):
    frame = validate_results(_paired_rows(multiplier))
    result = compare_baselines(frame, "ppo").iloc[0]
    assert result.cost_saved_pct == pytest.approx(saving)
    assert result.cost_saved_pct_ci95_low == pytest.approx(saving)
    assert result.cost_saved_pct_ci95_high == pytest.approx(saving)
    assert result.paired_days == 4  # 8 RL episodes represent four days, not eight independent days.
    assert result.rl_model_seeds == 2
    assert result.status == status


def test_same_seed_with_different_tapes_is_rejected():
    frame = _paired_rows()
    frame.loc[0, "scenario_hash"] = "different-tape"
    with pytest.raises(ValueError, match="scenario hash mismatch"):
        validate_results(frame)


def test_incomplete_seed_coverage_and_duplicate_inputs_are_rejected():
    frame = _paired_rows().iloc[1:]
    with pytest.raises(ValueError, match="unpaired scenarios"):
        compare_baselines(validate_results(frame), "ppo")
    duplicate = pd.concat([_paired_rows(), _paired_rows().iloc[:1]])
    with pytest.raises(ValueError, match="duplicate evaluation"):
        validate_results(duplicate)


def test_missing_identity_or_single_day_cannot_produce_a_significance_claim():
    missing = validate_results(_paired_rows().drop(columns="scenario_hash"))
    result = compare_baselines(missing, "ppo").iloc[0]
    assert result.status == "missing_scenario_identity"
    assert np.isnan(result.cost_saved_pct)
    one_day = validate_results(_paired_rows().iloc[:3])
    result = compare_baselines(one_day, "ppo").iloc[0]
    assert result.status == "insufficient_days"
    assert result.cost_saved_pct == pytest.approx(20)
    assert np.isnan(result.cost_saved_pct_ci95_low)


def test_zero_baseline_cost_has_no_relative_percentage():
    frame = _paired_rows()
    frame.loc[frame.method == "fixed", "total_cost_core"] = 0
    result = compare_baselines(validate_results(frame), "ppo").iloc[0]
    assert np.isnan(result.cost_saved_pct)
    assert result.status == "rl_higher_cost"


def test_historical_training_reference_is_not_labelled_untrained():
    history = pd.DataFrame(
        {
            "run_id": ["a"] * 3,
            "seed": [11] * 3,
            "encoder": ["gru"] * 3,
            "history_minutes": [10] * 3,
            "timesteps": [12000, 24000, 36000],
            "val_cost": [100.0, 80.0, 90.0],
        }
    )
    row = summarize_training(history).iloc[0]
    assert row.reference_kind == "first_logged"
    assert row.reference_timestep == 12000
    assert row.best_timesteps == 24000
    assert row.last_val_cost == 90
    assert row.validation_reduction_pct == 20


def test_step_zero_is_reference_only_and_training_regression_stays_visible():
    history = pd.DataFrame(
        {
            "run_id": ["a"] * 3,
            "seed": [11] * 3,
            "encoder": ["belief"] * 3,
            "history_minutes": [10] * 3,
            "timesteps": [0, 100, 200],
            "val_cost": [80.0, 100.0, 120.0],
            "checkpoint_eligible": [False, True, True],
        }
    )
    row = summarize_training(history).iloc[0]
    assert row.reference_kind == "untrained"
    assert row.best_timesteps == 100
    assert row.best_val_cost == 100
    assert row.validation_reduction_pct == -25
