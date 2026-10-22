import json

import pytest

from bus_rl.artifacts import export_artifacts, verify_artifacts


def _completed_source(tmp_path):
    source = tmp_path / "local"
    train = source / "gru10" / "11" / "train"
    train.mkdir(parents=True)
    (source / "status.json").write_text('{"status":"complete"}')
    (source / "protocol.json").write_text('{"schema_version":1}')
    (train / "metadata.json").write_text('{"seed":11}')
    (train / "evaluations.csv").write_text("timesteps,val_cost\n0,100\n32,80\n")
    for name in (
        "best.zip",
        "last.zip",
        "training_diagnostics.jsonl",
        "validation_metrics.csv",
        "train.log",
    ):
        (train / name).write_text("local state")
    return source


def test_export_preserves_evidence_and_excludes_training_state(tmp_path):
    source = _completed_source(tmp_path)
    output = tmp_path / "git-evidence"
    manifest = export_artifacts(source, output)
    assert set(manifest["files"]) == {
        "status.json",
        "protocol.json",
        "gru10/11/train/metadata.json",
        "gru10/11/train/evaluations.csv",
    }
    assert len(manifest["local_only"]) == 5
    assert (source / "gru10/11/train/best.zip").exists()
    assert verify_artifacts(output)["inventory_sha256"] == manifest["inventory_sha256"]
    (output / "gru10/11/train/evaluations.csv").write_text("tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_artifacts(output)


def test_export_requires_completion_and_separate_fresh_output(tmp_path):
    source = _completed_source(tmp_path)
    (source / "status.json").write_text(json.dumps({"status": "training"}))
    with pytest.raises(ValueError, match="completed experiment"):
        export_artifacts(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    (source / "status.json").write_text('{"status":"complete"}')
    with pytest.raises(ValueError, match="separate directory"):
        export_artifacts(source, source / "export")


def test_git_ignore_separates_local_state_from_run_evidence():
    import subprocess

    checks = subprocess.run(
        ["git", "check-ignore", "--stdin"],
        input="experiments/standard-v1/x/train/best.zip\nruns/standard-v1/x/train/best.zip\nruns/standard-v1/x/train/metadata.json\n",
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    assert checks == [
        "experiments/standard-v1/x/train/best.zip",
        "runs/standard-v1/x/train/best.zip",
    ]


def test_verify_bundle_with_nested_manifest_checks_report_and_evidence(tmp_path):
    import shutil

    from bus_rl.provenance import canonical_hash, file_hash

    source = _completed_source(tmp_path)
    bundle = tmp_path / "bundle"
    export_artifacts(source, bundle)
    old = json.loads((bundle / "artifacts.json").read_text())
    evidence = bundle / "evidence"
    evidence.mkdir()
    for path in list(bundle.iterdir()):
        if path != evidence:
            shutil.move(str(path), evidence / path.name)
    report = bundle / "report.pdf"
    report.write_bytes(b"PDF fixture")
    old["files"] = {
        path.relative_to(bundle).as_posix(): {
            "sha256": file_hash(path),
            "bytes": path.stat().st_size,
        }
        for path in bundle.rglob("*")
        if path.is_file() and path != evidence / "artifacts.json"
    }
    old["inventory_sha256"] = canonical_hash(old["files"])
    (evidence / "artifacts.json").write_text(json.dumps(old))
    assert verify_artifacts(bundle)["files"]["report.pdf"]["bytes"] == 11
    report.write_bytes(b"tampered PDF")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_artifacts(bundle)
