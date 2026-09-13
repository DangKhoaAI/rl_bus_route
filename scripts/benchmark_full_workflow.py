#!/usr/bin/env python
"""O5 multi-repetition full-workflow benchmark (paired, interleaved).

Runs `bus_rl.cli train` as separate processes for each variant, rotating the
variant order between repetitions so host drift cannot favour one variant.
Reports median/min/max wall and RSS, and checks that every run is bit-identical
to the reference baseline (validation curve + policy/optimizer tensors).

`baseline` uses the shipped defaults (distribution validation on). The opt-in
variants use `--no-validate-distributions`; `o2b16-on` reruns the accepted O2
config with validation on to isolate that toggle from the native batch step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from stable_baselines3.common.save_util import load_from_zip_file

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "core-threads2.toml"
MANIFEST = ROOT / "data" / "generated" / "base" / "manifest.json"
RUNS = ROOT / "runs" / "runtime-optimization" / "full-reps"
REPORT = ROOT / "reports" / "runtime-optimization" / "full-workflow-reps.json"

VARIANTS: dict[str, list[str]] = {
    "baseline": [],
    "o2b16": [
        "--native-batch",
        "--eval-batch-size",
        "16",
        "--reuse-eval-pool",
        "--no-validate-distributions",
    ],
    "o2b32": [
        "--native-batch",
        "--eval-batch-size",
        "32",
        "--reuse-eval-pool",
        "--no-validate-distributions",
    ],
    "o2b16-on": [
        "--native-batch",
        "--eval-batch-size",
        "16",
        "--reuse-eval-pool",
        "--validate-distributions",
    ],
}

# `o2b16-on` is a parity probe, not a timing candidate; run it once.
VARIANT_REPS = {"baseline": None, "o2b16": None, "o2b32": None, "o2b16-on": 1}


def _run_once(variant: str, repetition: int) -> dict:
    output = RUNS / f"{variant}-r{repetition}"
    if output.exists():
        shutil.rmtree(output)
    command = [
        "/usr/bin/time",
        "-v",
        sys.executable,
        "-m",
        "bus_rl.cli",
        "train",
        "--config",
        str(CONFIG),
        "--manifest",
        str(MANIFEST),
        "--seed",
        "11",
        "--output",
        str(output),
        "--backend",
        "rust",
        "--torch-threads",
        "2",
        "--eval-limit",
        "100",
        *VARIANTS[variant],
    ]
    started = perf_counter()
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    wall = perf_counter() - started
    if process.returncode != 0:
        sys.stderr.write(process.stdout[-2000:])
        sys.stderr.write(process.stderr[-4000:])
        raise SystemExit(f"{variant} r{repetition} failed with {process.returncode}")
    match = re.search(r"Maximum resident set size \(kbytes\): (\d+)", process.stderr)
    if match is None:
        raise SystemExit(f"could not read RSS for {variant} r{repetition}")
    metadata = json.loads((output / "metadata.json").read_text())
    return {
        "variant": variant,
        "repetition": repetition,
        "wall_s": wall,
        "learn_wall_s": metadata["wall_time_s"],
        "peak_rss_kb": int(match.group(1)),
        "best_val_cost": metadata["best_val_cost"],
        "validate_distributions": metadata["validate_distributions"],
        "torch_distribution_validate_args": metadata["torch_distribution_validate_args"],
        "native_library_sha256": metadata["native_build"]["library_sha256"],
        "build_record_matches_runtime": metadata["native_build"]["build_record_matches_runtime"],
        "output": output,
        "evaluations": pd.read_csv(output / "evaluations.csv"),
        "checkpoints": {
            which: hashlib.sha256((output / f"{which}.zip").read_bytes()).hexdigest()
            for which in ("best", "last")
        },
    }


def _leaves(payload, prefix: str = ""):
    if isinstance(payload, dict):
        for key, value in payload.items():
            yield from _leaves(value, f"{prefix}{key}.")
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            yield from _leaves(value, f"{prefix}{index}.")
    else:
        yield prefix.rstrip("."), payload


def _tensor_state(run: dict, which: str) -> dict:
    _, params, _ = load_from_zip_file(
        str(run["output"] / f"{which}.zip"), device="cpu", load_data=False
    )
    return dict(_leaves(params))


def _max_tensor_diff(left: dict, right: dict) -> float:
    diff = 0.0
    for key in set(left) | set(right):
        a, b = left.get(key), right.get(key)
        if hasattr(a, "detach"):
            a = a.cpu().numpy()
        if hasattr(b, "detach"):
            b = b.cpu().numpy()
        if isinstance(a, (str, type(None), bool)) or isinstance(b, (str, type(None), bool)):
            assert a == b, (key, a, b)
            continue
        diff = max(
            diff, float(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)).max())
        )
    return diff


def _stats(values: list[float]) -> dict:
    return {
        "values": [round(value, 3) for value in values],
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, default=REPORT)
    args = parser.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)
    variants = list(VARIANTS)
    per_variant_reps = {
        name: (VARIANT_REPS[name] if VARIANT_REPS.get(name) is not None else args.repetitions)
        for name in variants
    }
    max_reps = max(per_variant_reps.values())
    records: list[dict] = []
    for repetition in range(max_reps):
        order = variants[repetition % len(variants) :] + variants[: repetition % len(variants)]
        for variant in order:
            if repetition >= per_variant_reps[variant]:
                continue
            print(f"[fw] rep {repetition} {variant}", flush=True)
            record = _run_once(variant, repetition)
            print(
                f"[fw]   wall={record['wall_s']:.2f}s rss={record['peak_rss_kb']} "
                f"best={record['best_val_cost']}",
                flush=True,
            )
            records.append(record)

    baseline = next(record for record in records if record["variant"] == "baseline")
    reference_curve = baseline["evaluations"]
    reference_state = {which: _tensor_state(baseline, which) for which in ("last", "best")}
    parity = {}
    for record in records:
        key = f"{record['variant']}-r{record['repetition']}"
        curve = record["evaluations"]
        entry = {
            "schedule_identical": list(curve["timesteps"]) == list(reference_curve["timesteps"]),
            "max_abs_val_cost_diff": float(
                np.abs(curve["val_cost"].to_numpy() - reference_curve["val_cost"].to_numpy()).max()
            ),
        }
        for which in ("last", "best"):
            entry[f"{which}_tensor_max_abs_diff"] = _max_tensor_diff(
                reference_state[which], _tensor_state(record, which)
            )
        parity[key] = entry

    results = {}
    for variant in per_variant_reps:
        rows = [record for record in records if record["variant"] == variant]
        results[variant] = {
            "flags": VARIANTS[variant],
            "repetitions": len(rows),
            "wall_s": _stats([row["wall_s"] for row in rows]),
            "learn_wall_s": _stats([row["learn_wall_s"] for row in rows]),
            "peak_rss_kb": _stats([float(row["peak_rss_kb"]) for row in rows]),
            "best_val_cost": [row["best_val_cost"] for row in rows],
            "validate_distributions": sorted({row["validate_distributions"] for row in rows}),
            "native_library_sha256": sorted({row["native_library_sha256"] for row in rows}),
            "build_record_matches_runtime": sorted(
                {tuple(row["build_record_matches_runtime"]) for row in rows}
            ),
        }
    median = {name: results[name]["wall_s"]["median"] for name in results}
    speedup = {
        f"{name}_vs_baseline": round(median["baseline"] / median[name], 4)
        for name in results
        if name != "baseline" and median[name]
    }
    memory = {
        f"{name}_over_baseline": round(
            results[name]["peak_rss_kb"]["median"] / results["baseline"]["peak_rss_kb"]["median"], 4
        )
        for name in results
        if name != "baseline"
    }
    payload = {
        "protocol": {
            "variants": VARIANTS,
            "variant_reps": per_variant_reps,
            "interleaved": True,
            "seed": 11,
            "torch_threads": 2,
            "execution_note": "variant order rotates each repetition; separate processes",
        },
        "results": results,
        "speedup_median": speedup,
        "memory_median_ratio": memory,
        "parity_vs_baseline_run": parity,
        "per_run": [
            {
                key: value
                for key, value in record.items()
                if key not in ("evaluations", "output", "checkpoints")
            }
            | {"checkpoints": record["checkpoints"]}
            for record in records
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "speedup_median": speedup,
                "memory_median_ratio": memory,
                "wall_medians": {k: results[k]["wall_s"]["median"] for k in results},
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"[fw] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
