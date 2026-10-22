#!/usr/bin/env python
"""Append a completed masking study without overwriting original policy evidence."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd
from build_study_report import build_bundle, load_algorithm_extension

from bus_rl.artifacts import export_artifacts, verify_artifacts
from bus_rl.provenance import canonical_hash, file_hash


def refresh_inventory(study, manifest):
    index = study / "evidence/artifacts.json"
    files = {
        p.relative_to(study).as_posix(): {"sha256": file_hash(p), "bytes": p.stat().st_size}
        for p in sorted(study.rglob("*"))
        if p.is_file() and p != index
    }
    manifest = {
        **manifest,
        "files": files,
        "inventory_sha256": canonical_hash(files),
        "total_bytes": sum(item["bytes"] for item in files.values()),
    }
    index.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    verify_artifacts(study)


def validate_shared_baselines(original, additional):
    """Every operating metric and scenario identity must agree, not timing telemetry."""
    for split in ("test_id", "test_ood_burst", "test_ood_traffic"):
        left = pd.read_csv(original / split / "results.csv")
        right = pd.read_csv(additional / split / "results.csv")
        if missing := set(left) - set(right):
            raise ValueError(f"extension baseline is missing columns: {sorted(missing)}")
        common = sorted((set(left) & set(right)) - {"wall_s"})
        keys = ["method", "scenario_seed"]
        pd.testing.assert_frame_equal(
            left[common].sort_values(keys).reset_index(drop=True),
            right[common].sort_values(keys).reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1e-10,
            atol=1e-8,
        )


def extend_study(study, source, screening):
    original_manifest = verify_artifacts(study)
    original_files = {
        name: data["sha256"]
        for name, data in original_manifest["files"].items()
        if name.startswith("evidence/")
    }
    if json.loads((source / "data/manifest.json").read_text()) != json.loads(
        (study / "evidence/data/manifest.json").read_text()
    ):
        raise ValueError("extension dataset differs from standard-v1")
    validate_shared_baselines(
        study / "evidence/baselines/shared/eval", source / "baselines/shared/eval"
    )
    with tempfile.TemporaryDirectory(prefix="bus-algorithm-extension-") as temporary:
        root = Path(temporary)
        exported = root / "export"
        exported_manifest = export_artifacts(source, exported)
        stage = root / study.name
        shutil.copytree(study, stage)
        evidence = stage / "evidence"
        extension = evidence / "extensions/masking"
        extension.mkdir(parents=True, exist_ok=False)
        for name in ("protocol.json", "status.json", "summary.csv", "comparisons.csv"):
            shutil.copyfile(exported / name, extension / name)
        shutil.copytree(exported / "configs", extension / "configs")
        for algo in ("reinforce", "a2c", "ppo"):
            for mode in ("apply", "free"):
                origin = exported / f"{algo}_{mode}"
                target = evidence / algo / mode
                if target.exists():
                    raise ValueError(f"extension arm already exists: {target}")
                target.mkdir(parents=True)
                for seed in (11, 22, 33):
                    shutil.copytree(origin / str(seed), target / str(seed))
                screen_origin = screening / algo / mode / "11/train"
                screen_target = extension / "screening" / algo / mode / "11"
                screen_target.mkdir(parents=True)
                for filename in ("metadata.json", "evaluations.csv"):
                    shutil.copyfile(screen_origin / filename, screen_target / filename)
        shutil.copyfile(screening / "screening.json", extension / "screening.json")
        # Check all original evidence hashes before regenerating any presentation file.
        for name, digest in original_files.items():
            if file_hash(stage / name) != digest:
                raise ValueError(f"original evidence was changed: {name}")
        load_algorithm_extension(evidence)
        manifest = dict(original_manifest)
        manifest["extensions"] = [
            *manifest.get("extensions", []),
            {
                "kind": "algorithm-masking-ablation",
                "source": str(source.resolve()),
                "protocol_sha256": file_hash(extension / "protocol.json"),
                "source_inventory_sha256": exported_manifest["inventory_sha256"],
                "layout": "evidence/<algorithm>/<mode>/<seed>/{train,eval}",
            },
        ]
        refresh_inventory(stage, manifest)
        build_bundle(stage)
        # Report regeneration must preserve every pre-existing evidence file.
        for name, digest in original_files.items():
            if file_hash(stage / name) != digest:
                raise ValueError(f"report build changed original evidence: {name}")
        verify_artifacts(stage)
        backup = Path(tempfile.mkdtemp(prefix="bus-study-before-extension-")) / study.name
        shutil.move(str(study), backup)
        try:
            shutil.copytree(stage, study)
            verify_artifacts(study)
        except BaseException:
            shutil.rmtree(study, ignore_errors=True)
            shutil.move(str(backup), study)
            raise
    print(f"Extended {study}; original bundle preserved at {backup}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--screening", type=Path, required=True)
    args = parser.parse_args()
    extend_study(args.study, args.source, args.screening)
