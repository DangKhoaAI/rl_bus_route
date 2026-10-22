import importlib
import json

import pandas as pd
import pytest

from bus_rl.provenance import file_hash
from scripts.build_study_report import load_algorithm_extension, validate_decision_example
from tests.support.paths import ROOT


def test_incomplete_masking_matrix_cannot_be_reported_as_complete(tmp_path):
    extension = tmp_path / "extensions/masking"
    extension.mkdir(parents=True)
    (extension / "protocol.json").write_text(
        json.dumps(
            {"plan": {"spec": {"policies": {"ppo_apply": "config.toml"}, "seeds": [11, 22, 33]}}}
        )
    )
    with pytest.raises(ValueError, match="incomplete"):
        load_algorithm_extension(tmp_path)


def test_append_rejects_changed_or_missing_operating_metrics(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    extend = importlib.import_module("extend_standard_study")
    old, new = tmp_path / "old", tmp_path / "new"
    frame = pd.DataFrame(
        {"method": ["fixed"], "scenario_seed": [3001], "total_cost_core": [100.0], "wall_s": [1.0]}
    )
    for split in ("test_id", "test_ood_burst", "test_ood_traffic"):
        for root in (old, new):
            (root / split).mkdir(parents=True)
        frame.to_csv(old / split / "results.csv", index=False)
        frame.assign(wall_s=9.0, invalid_request_rate=0.0).to_csv(
            new / split / "results.csv", index=False
        )
    extend.validate_shared_baselines(old, new)
    frame.assign(total_cost_core=101.0).to_csv(new / "test_id/results.csv", index=False)
    with pytest.raises(AssertionError):
        extend.validate_shared_baselines(old, new)
    frame.drop(columns="total_cost_core").to_csv(new / "test_id/results.csv", index=False)
    with pytest.raises(ValueError, match="missing columns"):
        extend.validate_shared_baselines(old, new)


def test_decision_example_must_match_episode_counters(tmp_path):
    path = tmp_path / "eval/test_id/decisions.csv"
    path.parent.mkdir(parents=True)
    decisions = pd.DataFrame(
        {
            "step": range(1, 121),
            "requested_action_index": [0] + [1] * 119,
            "executed_action_index": [0] * 120,
            "requested_action_valid": [True] + [False] * 119,
            "invalid_action_fallback": ["none"] + ["noop"] * 119,
        }
    )
    decisions.to_csv(path, index=False)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "scenario_seed": 1,
                "scenario_hash": "scenario",
                "checkpoint_sha256": "checkpoint",
                "decisions_sha256": file_hash(path),
                "repeated_200_day_evaluation_matches": True,
            }
        )
    )
    results = pd.DataFrame(
        {
            "scenario_seed": ["1"],
            "scenario_hash": ["scenario"],
            "invalid_request_count": [119],
            "requested_noop_count": [1],
            "executed_noop_count": [120],
        }
    )
    metadata = {"best_checkpoint_sha256": "checkpoint"}
    validate_decision_example(tmp_path, results, metadata)
    with pytest.raises(ValueError, match="counters"):
        validate_decision_example(tmp_path, results.assign(invalid_request_count=118), metadata)
