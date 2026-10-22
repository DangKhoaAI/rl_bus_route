from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from bus_rl.cli import main as cli_main
from bus_rl.evaluation.plots import render_plots
from bus_rl.provenance import require_fresh_output
from tests.support.paths import ROOT


def test_cli_smoke_writes_csv_and_rejects_existing_output(tmp_path):
    data = tmp_path / "data"
    command = [
        sys.executable,
        "-m",
        "bus_rl.cli",
        "data",
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
            "model",
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
    assert metadata["algorithm"]["encoder"] == "gru"
    assert metadata["algorithm"]["history_minutes"] == 10

    eval_out = tmp_path / "eval"
    cli_main(
        [
            "model",
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
    first_event = json.loads((eval_out / "events.jsonl").read_text().splitlines()[0])
    assert isinstance(first_event["headway_targets"], list)
    assert all(isinstance(value, (int, float)) for value in first_event["headway_targets"])

    baseline_out = tmp_path / "baselines"
    cli_main(
        [
            "baseline",
            "run",
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

    report_out = tmp_path / "report"
    cli_main(
        [
            "report",
            "generate",
            "--results",
            str(eval_out / "results.csv"),
            "--history",
            str(train_out / "evaluations.csv"),
            "--output",
            str(report_out),
        ]
    )
    assert (report_out / "summary.json").exists()
    assert (report_out / "learning_curves.png").exists()
    assert (report_out / "queue_heatmap.png").exists()
    assert (report_out / "headway.png").exists()

    diagnose_out = tmp_path / "diagnose"
    cli_main(
        [
            "test",
            "diagnose",
            "--manifest",
            str(data / "manifest.json"),
            "--output",
            str(diagnose_out),
            "--timesteps",
            "32",
            "--n-envs",
            "1",
            "--eval-limit",
            "1",
            "--torch-threads",
            "2",
        ]
    )
    assert (diagnose_out / "diagnosis.json").exists()
    assert (diagnose_out / "cprofile.txt").exists()
    assert (diagnose_out / "last.zip").exists()


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


@pytest.mark.parametrize(
    "flags,encoder,minutes,prior,decay",
    [
        ([], "gru", 10, 1.0, 1.0),
        (["--encoder", "mlp"], "mlp", 10, 1.0, 1.0),
        (["--config", "configs/policies/gru30.toml"], "gru", 30, 1.0, 1.0),
        (
            [
                "--config",
                "configs/policies/belief20.toml",
                "--encoder",
                "belief_gru",
                "--belief-prior-shape",
                "0.1",
                "--belief-prior-rate",
                "0.1",
                "--belief-decay",
                "0.8",
            ],
            "belief_gru",
            20,
            0.1,
            0.8,
        ),
    ],
)
def test_cli_policy_defaults_config_flag_precedence_and_checkpoint(
    tmp_path, flags, encoder, minutes, prior, decay
):
    from bus_rl.config import load_run_config
    from bus_rl.learning.training.checkpoint import load_model

    data = tmp_path / "data"
    cli_main(
        [
            "data",
            "generate",
            "--config",
            "configs/base.toml",
            "--output",
            str(data),
            "--counts",
            "train=1,validation=1",
        ]
    )
    train = tmp_path / "train"
    cli_main(
        [
            "model",
            "train",
            "--manifest",
            str(data / "manifest.json"),
            "--output",
            str(train),
            "--timesteps",
            "32",
            "--n-envs",
            "1",
            "--eval-limit",
            "1",
            "--torch-threads",
            "2",
            *flags,
        ]
    )
    meta = json.loads((train / "metadata.json").read_text())
    assert meta["algorithm"]["encoder"] == encoder
    assert meta["algorithm"]["history_minutes"] == minutes
    assert meta["algorithm"]["belief_prior_shape"] == prior
    assert meta["algorithm"]["belief_prior_rate"] == prior
    assert meta["algorithm"]["belief_decay"] == decay
    config = load_run_config(ROOT / "configs/base.toml")
    model, _ = load_model(train / "best.zip", None, config.physical, "rust")
    extractor = model.policy.features_extractor
    assert getattr(extractor, "use_gru", False) == (encoder in {"gru", "belief_gru"})
    assert getattr(extractor, "use_belief", False) == (encoder in {"belief", "belief_gru"})
    output = tmp_path / "eval"
    cli_main(
        [
            "model",
            "evaluate",
            "--manifest",
            str(data / "manifest.json"),
            "--checkpoint",
            str(train / "best.zip"),
            "--output",
            str(output),
            "--torch-threads",
            "2",
        ]
    )
    result = pd.read_csv(output / "results.csv")
    assert result.total_cost_core.mean() == pytest.approx(meta["best_val_cost"])


@pytest.mark.parametrize(
    "flags",
    [
        ["--encoder", "mlp", "--history-minutes", "20"],
        ["--belief-decay", "0"],
        ["--encoder", "gru", "--history-minutes", "21"],
    ],
)
def test_cli_invalid_policy_settings_fail_before_output_creation(tmp_path, flags):
    output = tmp_path / "invalid"
    with pytest.raises(SystemExit) as error:
        cli_main(["model", "train", "--manifest", "unused.json", "--output", str(output), *flags])
    assert error.value.code == 1
    assert not output.exists()


def test_flat_generate_alias_preserves_archived_command(tmp_path):
    output = tmp_path / "old-command"
    cli_main(
        [
            "generate",
            "--config",
            "configs/base.toml",
            "--output",
            str(output),
            "--counts",
            "train=1",
        ]
    )
    assert (output / "manifest.json").exists()


def test_grouped_profile_routes_to_existing_handler(tmp_path, monkeypatch):
    from bus_rl.evaluation import profile

    calls = []

    # Only verify CLI dispatch; the historical microprofiler's Python engine is deprecated.
    def fake_profile(output, config, decisions):
        calls.append((output, config, decisions))
        return []

    monkeypatch.setattr(profile, "run_profile", fake_profile)
    cli_main(
        [
            "test",
            "profile",
            "--config",
            "configs/pilot.toml",
            "--decisions",
            "2",
            "--output",
            str(tmp_path / "profile"),
        ]
    )
    assert len(calls) == 1
    assert calls[0][0] == tmp_path / "profile"
    assert calls[0][2] == 2


@pytest.mark.parametrize(
    "group,flags,action",
    [
        ("baseline", ["--config", "configs/eval.toml", "--manifest", "unused.json"], "run"),
        ("report", ["--results", "unused.csv"], "generate"),
    ],
)
def test_flat_aliases_with_names_shared_by_groups(tmp_path, monkeypatch, group, flags, action):
    from bus_rl import cli

    calls = []
    monkeypatch.setattr(cli, f"cmd_{group}", calls.append)
    cli.main([group, *flags, "--output", str(tmp_path / "output")])
    assert len(calls) == 1
    assert (calls[0].group, calls[0].action) == (group, action)
