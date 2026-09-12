from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bus_rl.cli import main as cli_main
from bus_rl.evaluation.plots import render_plots
from bus_rl.evaluation.statistics import paired_bootstrap
from bus_rl.provenance import require_fresh_output

ROOT = Path(__file__).resolve().parents[1]


def test_paired_bootstrap_constant_difference_is_exact():
    left = np.arange(10.0) + 2.0
    right = np.arange(10.0)
    mean, low, high = paired_bootstrap(left, right, n_resamples=2_000, seed=6_001)
    assert mean == 2.0
    assert low == 2.0
    assert high == 2.0


def test_cli_smoke_writes_csv_and_rejects_existing_output(tmp_path):
    data = tmp_path / "data"
    command = [
        sys.executable,
        "-m",
        "bus_rl.cli",
        "generate",
        "--config",
        str(ROOT / "configs/base.toml"),
        "--output",
        str(data),
        "--counts",
        "train=2,validation=1,test_id=1",
    ]
    # One real subprocess checks the installed entry point; the rest run
    # in-process so torch/sb3 are imported once.
    subprocess.run(command, check=True, cwd=ROOT)
    assert (data / "manifest.json").exists()
    with pytest.raises(SystemExit) as repeated:
        cli_main(command[3:])
    assert repeated.value.code != 0

    train_out = tmp_path / "train"
    cli_main(
        [
            "train",
            "--config",
            str(ROOT / "configs/pilot.toml"),
            "--manifest",
            str(data / "manifest.json"),
            "--seed",
            "11",
            "--output",
            str(train_out),
            "--timesteps",
            "32",
            "--n-envs",
            "1",
            "--eval-limit",
            "1",
        ]
    )
    assert (train_out / "best.zip").exists()
    assert (train_out / "metadata.json").exists()
    metadata = json.loads((train_out / "metadata.json").read_text())
    assert metadata["obs_version"] == 2
    assert metadata["total_timesteps_actual"] >= 32

    eval_out = tmp_path / "eval"
    cli_main(
        [
            "evaluate",
            "--config",
            str(ROOT / "configs/eval.toml"),
            "--manifest",
            str(data / "manifest.json"),
            "--checkpoint",
            str(train_out / "best.zip"),
            "--split",
            "validation",
            "--output",
            str(eval_out),
            "--trace",
        ]
    )
    results = pd.read_csv(eval_out / "results.csv")
    assert not results.empty
    assert "total_cost_core" in results.columns
    assert (eval_out / "events.jsonl").exists()

    baseline_out = tmp_path / "baselines"
    cli_main(
        [
            "baseline",
            "--config",
            str(ROOT / "configs/eval.toml"),
            "--manifest",
            str(data / "manifest.json"),
            "--split",
            "validation",
            "--methods",
            "fixed,threshold,proportional",
            "--output",
            str(baseline_out),
        ]
    )
    baseline = pd.read_csv(baseline_out / "results.csv")
    assert set(baseline["method"]) == {"fixed", "threshold", "proportional"}


def test_require_fresh_output_and_plots_handle_none(tmp_path):
    path = tmp_path / "already"
    path.mkdir()
    with pytest.raises(FileExistsError):
        require_fresh_output(path)
    frame = pd.DataFrame(
        {
            "method": ["fixed", "ppo"],
            "model_seed": [np.nan, 11],
            "scenario_seed": [1, 1],
            "mean_wait": [np.nan, np.nan],
            "generated": [0, 0],
            "total_cost_core": [10.0, 12.0],
            "abandoned_share": [np.nan, np.nan],
        }
    )
    render_plots(frame, tmp_path / "plots")
    assert (tmp_path / "plots" / "waiting_cdf.png").exists()
    assert (tmp_path / "plots" / "cost_bars.png").exists()
    assert (tmp_path / "plots" / "fleet_gantt.png").exists()
