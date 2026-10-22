import importlib
import json

import pandas as pd
import pytest

from scripts.build_study_report import load_algorithm_extension
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
