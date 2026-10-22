"""Fixed experiment matrices: local training state and a separate Git evidence export."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from bus_rl.artifacts import export_artifacts
from bus_rl.config import load_run_config
from bus_rl.evaluation.report import generate_report
from bus_rl.execution.scenarios.io import load_manifest
from bus_rl.provenance import file_hash, git_status, physical_config_hash, require_fresh_output


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _status(output: Path, stage: str, **extra) -> None:
    _write(
        output / "status.json",
        {"status": stage, "updated_at": datetime.now(UTC).isoformat(), **extra},
    )


def run_experiment(
    config: Path, manifest: Path, output: Path, artifacts: Path | None = None, resume: bool = False
) -> None:
    from bus_rl.cli import main

    spec = tomllib.loads(config.read_text())
    if spec.get("schema_version") != 1:
        raise ValueError("unsupported experiment schema")
    seeds, splits, baselines = spec["seeds"], spec["splits"], spec["baselines"]
    policies = spec["policies"]
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(seed) is not int or seed < 0 for seed in seeds)
    ):
        raise ValueError("experiment requires unique non-negative integer seeds")
    if (
        not splits
        or len(set(splits)) != len(splits)
        or not baselines
        or len(set(baselines)) != len(baselines)
    ):
        raise ValueError("experiment requires unique splits and baselines")
    if set(baselines) - {"fixed", "proportional", "threshold", "random"}:
        raise ValueError("unsupported experiment baseline")
    if not policies or any(
        not re.fullmatch(r"[a-z][a-z0-9_-]*", name) or name in {"baselines", "configs", "data"}
        for name in policies
    ):
        raise ValueError("invalid experiment policy names")
    if any(not re.fullmatch(r"[a-z][a-z0-9_-]*", name) for name in splits):
        raise ValueError("invalid experiment split names")
    if artifacts is not None:
        require_fresh_output(artifacts)
        if (
            output.resolve() in artifacts.resolve().parents
            or artifacts.resolve() in output.resolve().parents
            or output.resolve() == artifacts.resolve()
        ):
            raise ValueError("local experiment and Git artifacts require separate trees")
    runs = {name: load_run_config(Path(path)) for name, path in policies.items()}
    baseline_config = spec["baseline_config"]
    baseline_run = load_run_config(Path(baseline_config))
    jobs = spec.get("jobs", 1)
    if type(jobs) is not int or not 1 <= jobs <= 8:
        raise ValueError("experiment jobs must be an integer in [1, 8]")
    if baseline_run.runtime.eval_batch_size != 1 or baseline_run.runtime.native_batch:
        raise ValueError("baseline config must select scalar evaluation")
    budgets = {run.algorithm.total_timesteps for run in runs.values()}
    if len(budgets) != 1:
        raise ValueError("standard policies must have the same training budget")
    dataset = load_manifest(manifest)
    for split, count in spec["counts"].items():
        if len(dataset["splits"].get(split, [])) != count:
            raise ValueError(f"dataset count mismatch for {split}: expected {count}")
    if {"train", "validation", *splits} - set(dataset["splits"]):
        raise ValueError("dataset does not contain every requested split")
    for run in [*runs.values(), baseline_run]:
        if physical_config_hash(run.physical) != dataset["physical_config_hash"]:
            raise ValueError("experiment config does not match dataset physics")
        if run.forecast.enabled:
            raise ValueError("standard matrix currently requires forecast-disabled policies")
    sha, dirty = git_status()
    source_hashes = {str(path): file_hash(path) for path in sorted(Path("src").rglob("*.py"))}
    source_hashes.update(
        {str(path): file_hash(path) for path in sorted(Path("crates").rglob("*.rs"))}
    )
    source_hashes["uv.lock"] = file_hash(Path("uv.lock"))
    native = Path("src/bus_sim_native.so")
    if native.exists():
        source_hashes[str(native)] = file_hash(native)
    plan = {
        "schema_version": 1,
        "spec": spec,
        "manifest_sha256": file_hash(manifest),
        "config_sha256": {name: file_hash(Path(path)) for name, path in policies.items()},
        "baseline_config_sha256": file_hash(Path(baseline_config)),
        "base_config_sha256": file_hash(Path("configs/base.toml")),
        "source_sha256": source_hashes,
    }
    if resume:
        previous = json.loads((output / "protocol.json").read_text())
        if previous["plan"] != plan:
            raise ValueError(
                "resume plan differs in config, dataset or implementation; use a fresh output"
            )
    else:
        require_fresh_output(output)
        output.mkdir(parents=True)
        (output / "configs").mkdir()
        (output / "data").mkdir()
        shutil.copyfile(manifest, output / "data" / "manifest.json")
        shutil.copyfile(config, output / "configs" / "study.toml")
        for name, path in policies.items():
            shutil.copyfile(path, output / "configs" / f"{name}.toml")
        shutil.copyfile("configs/base.toml", output / "configs" / "base.toml")
        shutil.copyfile(baseline_config, output / "configs" / "baselines.toml")
        _write(
            output / "protocol.json",
            {
                "plan": plan,
                "git_sha": sha,
                "git_dirty": dirty,
                "created_at": datetime.now(UTC).isoformat(),
                "resolved_configs": {name: asdict(run) for name, run in runs.items()},
                "selection": "Minimum post-training validation cost; earliest tie; step zero is reference only",
                "evaluation": "Train all registered policies/seeds before evaluating test splits; no test-driven tuning",
                "data": "manifest.json identifies regenerable scenario tapes; tapes remain local",
            },
        )
    _status(output, "training")
    training_tasks = []
    for name, run in runs.items():
        for seed in seeds:
            destination = output / name / str(seed) / "train"
            if resume and (destination / "metadata.json").exists():
                meta = json.loads((destination / "metadata.json").read_text())
                expected = asdict(replace(run.algorithm, seed=seed))
                if meta["algorithm"] != expected or meta.get("best_checkpoint_sha256") != file_hash(
                    destination / "best.zip"
                ):
                    raise ValueError(f"inconsistent completed training: {destination}")
                continue
            training_tasks.append(
                [
                    "model",
                    "train",
                    "--config",
                    policies[name],
                    "--manifest",
                    str(manifest),
                    "--seed",
                    str(seed),
                    "--output",
                    str(destination),
                ]
            )

    def train_task(arguments):
        print(f"[experiment] train {' '.join(arguments)}", flush=True)
        subprocess.run(
            [sys.executable, "-m", "bus_rl.cli", *arguments], check=True, stdout=subprocess.DEVNULL
        )

    # Models run in separate processes so global Torch/NumPy RNG state is never shared.
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        list(executor.map(train_task, training_tasks))
    _status(output, "evaluation")
    baseline_paths = []
    for split in splits:
        destination = output / "baselines" / "shared" / "eval" / split
        baseline_paths.append(destination / "results.csv")
        if not (resume and _evaluation_complete(destination)):
            print(f"[experiment] baselines split={split}", flush=True)
            main(
                [
                    "baseline",
                    "run",
                    "--config",
                    baseline_config,
                    "--manifest",
                    str(manifest),
                    "--split",
                    split,
                    "--methods",
                    ",".join(baselines),
                    "--output",
                    str(destination),
                ]
            )
    all_summary, all_comparisons = [], []
    for name in policies:
        evaluations = []
        for seed in seeds:
            for split in splits:
                destination = output / name / str(seed) / "eval" / split
                evaluations.append(destination / "results.csv")
                if not (resume and _evaluation_complete(destination)):
                    print(f"[experiment] evaluate {name} seed={seed} split={split}", flush=True)
                    main(
                        [
                            "model",
                            "evaluate",
                            "--config",
                            policies[name],
                            "--manifest",
                            str(manifest),
                            "--checkpoint",
                            str(output / name / str(seed) / "train" / "best.zip"),
                            "--split",
                            split,
                            "--output",
                            str(destination),
                        ]
                    )
        report = output / name / "report"
        if not (resume and (report / "provenance" / "verification.json").exists()):
            print(f"[experiment] report {name}", flush=True)
            generate_report([*evaluations, *baseline_paths], report)
        for table, rows in [("summary", all_summary), ("comparisons", all_comparisons)]:
            frame = pd.read_csv(report / "tables" / f"{table}.csv")
            frame.insert(0, "policy", name)
            rows.append(frame)
    pd.concat(all_summary, ignore_index=True).to_csv(output / "summary.csv", index=False)
    pd.concat(all_comparisons, ignore_index=True).to_csv(output / "comparisons.csv", index=False)
    _write_index(output, spec)
    _status(output, "complete", policies=list(policies), seeds=seeds, splits=splits)
    if artifacts is not None:
        exported = export_artifacts(output, artifacts)
        print(
            f"[experiment] exported {exported['total_bytes'] / 1024**2:.1f} MiB to {artifacts}",
            flush=True,
        )
    # Best checkpoints remain available for reevaluation; final optimizer snapshots are redundant here.
    for name in policies:
        for seed in seeds:
            (output / name / str(seed) / "train" / "last.zip").unlink(missing_ok=True)


def _evaluation_complete(output: Path) -> bool:
    path = output / "evaluation.json"
    if not path.exists():
        return False
    meta = json.loads(path.read_text())
    if meta["results_sha256"] != file_hash(output / "results.csv"):
        raise ValueError(f"evaluation checksum mismatch: {output}")
    return True


def _write_index(output: Path, spec: dict) -> None:
    text = "# Standard Run Evidence\n\n"
    text += f"Study: `{spec['name']}`. Seeds: {', '.join(map(str, spec['seeds']))}. Evaluation conditions: {', '.join(spec['splits'])}.\n\n"
    text += "| Configuration | Evidence report |\n|---|---|\n"
    text += "".join(
        f"| `{name}` | [{name}/report/README.md]({name}/report/README.md) |\n"
        for name in spec["policies"]
    )
    text += "\n`summary.csv` and `comparisons.csv` consolidate every registered policy, including unsuccessful configurations. `protocol.json` freezes budgets, configs, scenario identities and implementation hashes.\n\n"
    text += "Each policy/seed contains train metadata and test evaluations. Policy-level reports aggregate all seeds. Shared baseline evaluations are under `baselines/shared/eval/`. These artifacts support the human-authored final study in `reports/`.\n\n"
    text += "Local checkpoint weights, validation episode telemetry and diagnostic logs are excluded from Git evidence. Regenerate tapes with `bus-rl data generate` using configs/base.toml and the split counts in configs/study.toml; compare the resulting manifest against data/manifest.json.\n"
    (output / "README.md").write_text(text)
