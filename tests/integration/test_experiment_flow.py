import json

import pandas as pd
import pytest

from bus_rl.artifacts import verify_artifacts
from bus_rl.cli import main


def test_fixed_matrix_exports_reusable_evidence_and_resumes_completed_stages(tmp_path):
    data = tmp_path / "data"
    main(
        [
            "data",
            "generate",
            "--config",
            "configs/base.toml",
            "--output",
            str(data),
            "--counts",
            "train=2,validation=1,test_id=2",
        ]
    )
    policy = tmp_path / "policy.toml"
    policy.write_text("""[algorithm]
encoder = "gru"
n_envs = 1
n_steps = 16
batch_size = 16
n_epochs = 1
total_timesteps = 32
eval_freq = 16
torch_threads = 2
diagnostics_enabled = false
""")
    baseline = tmp_path / "baseline.toml"
    baseline.write_text("[algorithm]\ntorch_threads = 2\n")
    study = tmp_path / "study.toml"
    study.write_text(f'''schema_version = 1
name = "smoke"
jobs = 1
seeds = [11]
splits = ["test_id"]
baselines = ["fixed"]
baseline_config = "{baseline}"
[counts]
train = 2
validation = 1
test_id = 2
[policies]
gru10 = "{policy}"
''')
    local, artifacts = tmp_path / "experiment", tmp_path / "evidence"
    arguments = [
        "experiment",
        "run",
        "--config",
        str(study),
        "--manifest",
        str(data / "manifest.json"),
        "--output",
        str(local),
    ]
    main([*arguments, "--artifacts", str(artifacts)])
    exported = verify_artifacts(artifacts)
    assert not any(name.endswith((".zip", ".log", ".jsonl")) for name in exported["files"])
    assert not (local / "gru10/11/train/last.zip").exists()
    assert (local / "gru10/11/train/best.zip").exists()
    history = pd.read_csv(artifacts / "gru10/11/train/evaluations.csv")
    assert history.timesteps.tolist() == [0, 16, 32]
    assert (artifacts / "gru10/report/report.html").exists()
    assert len(pd.read_csv(artifacts / "comparisons.csv")) == 1
    metadata = (local / "gru10/11/train/metadata.json").read_bytes()
    main([*arguments, "--resume"])
    assert (local / "gru10/11/train/metadata.json").read_bytes() == metadata
    assert json.loads((local / "status.json").read_text())["status"] == "complete"
    policy.write_text(policy.read_text().replace("32", "64"))
    with pytest.raises(SystemExit) as error:
        main([*arguments, "--resume"])
    assert error.value.code == 1
