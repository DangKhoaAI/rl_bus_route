"""Comparison helpers shared by the cross-backend parity tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tests.support.contract import OBS_KEYS


def assert_value(x, y, label: str) -> None:
    if x is None or y is None:
        assert x is None and y is None, (label, x, y)
    elif isinstance(x, float) or isinstance(y, float):
        if np.isnan(x) or np.isnan(y):
            assert np.isnan(x) and np.isnan(y), (label, x, y)
        else:
            np.testing.assert_allclose(float(x), float(y), rtol=1e-9, atol=1e-9, err_msg=label)
    else:
        assert x == y, (label, x, y)


def assert_frames(left: pd.DataFrame, right: pd.DataFrame) -> None:
    assert list(left.columns) == list(right.columns)
    assert len(left) == len(right)
    for column in left.columns:
        if column == "wall_s":
            continue
        for index, (x, y) in enumerate(zip(left[column], right[column], strict=True)):
            assert_value(x, y, f"{column}[{index}]")


def assert_obs(left: dict, right: dict, *, rtol: float = 1e-6, atol: float = 1e-6) -> None:
    assert set(left) == set(right)
    for key, value in left.items():
        np.testing.assert_allclose(value, right[key], rtol=rtol, atol=atol, err_msg=key)


def assert_obs_exact(left, right, label: str) -> None:
    for key in OBS_KEYS:
        np.testing.assert_array_equal(
            np.asarray(left[key]), np.asarray(right[key]), err_msg=f"{label}:{key}"
        )


def assert_single_result(batched_row: dict, scalar: dict, label: str) -> None:
    assert_obs_exact(
        {key: value[0] for key, value in batched_row["obs"].items()},
        scalar["obs"],
        label,
    )
    np.testing.assert_array_equal(batched_row["mask"][0], scalar["mask"], err_msg=label)
    assert float(batched_row["reward"][0]) == float(scalar["reward"]), label
    assert bool(batched_row["terminated"][0]) == bool(scalar["terminated"]), label
    np.testing.assert_array_equal(batched_row["costs"][0], scalar["costs"], err_msg=label)


def first_action_divergence(left_traces: dict, right_traces: dict) -> dict | None:
    for index in sorted(set(left_traces) | set(right_traces)):
        left_rows = left_traces.get(index, [])
        right_rows = right_traces.get(index, [])
        if len(left_rows) != len(right_rows):
            return {
                "field": "trace_length",
                "scenario_index": index,
                "expected": len(left_rows),
                "actual": len(right_rows),
            }
        for step, (left, right) in enumerate(zip(left_rows, right_rows, strict=True)):
            if left.get("action_index") != right.get("action_index"):
                return {
                    "field": "action_index",
                    "scenario_index": index,
                    "step": step,
                    "time_s": left.get("time_s"),
                    "expected_action": left.get("action_index"),
                    "actual_action": right.get("action_index"),
                    "expected_kind": left.get("action"),
                    "actual_kind": right.get("action"),
                    "reproducer": (
                        "evaluate_scenarios(..., trace_all=True) on "
                        f"fixtures/reference/diagnose-after day {index} step {step}"
                    ),
                }
    return None
