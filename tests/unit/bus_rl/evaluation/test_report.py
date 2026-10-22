import pandas as pd

from bus_rl.evaluation.report import _history


def test_legacy_histories_keep_real_seed_identity_when_encoder_is_missing(tmp_path):
    path = tmp_path / "evaluations.csv"
    pd.DataFrame({"timesteps": [100, 200], "val_cost": [10.0, 8.0]}).to_csv(path, index=False)
    first = _history(path, {"seed": 11}, 0)
    second = _history(path, {"seed": 22}, 1)
    assert first.run_id.notna().all()
    assert first.run_id.iloc[0] != second.run_id.iloc[0]
    assert first.seed.iloc[0] == "11"
    assert first.encoder.iloc[0] == "unknown"
    assert first.timesteps.min() == 100
