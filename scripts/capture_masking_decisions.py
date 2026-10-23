"""Capture compact, predetermined seed-11/test-ID decision examples for each arm."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from bus_rl.cli import main
from bus_rl.provenance import file_hash


def capture(source: Path):
    if json.loads((source / "status.json").read_text()).get("status") != "complete":
        raise ValueError("capture examples only after the complete registered matrix")
    for algo in ("reinforce", "a2c", "ppo"):
        for mode in ("apply", "free"):
            arm = source / f"{algo}_{mode}" / "11"
            original = arm / "eval/test_id"
            with tempfile.TemporaryDirectory(prefix="bus-mask-trace-") as temporary:
                output = Path(temporary) / "evaluation"
                main(
                    [
                        "model",
                        "evaluate",
                        "--config",
                        f"configs/studies/masking/{algo}-{mode}.toml",
                        "--manifest",
                        "data/generated/base/manifest.json",
                        "--checkpoint",
                        str(arm / "train/best.zip"),
                        "--split",
                        "test_id",
                        "--trace",
                        "--output",
                        str(output),
                    ]
                )
                previous = pd.read_csv(original / "results.csv")
                repeated = pd.read_csv(output / "results.csv")
                for metric in ("total_cost_core", "mean_wait", "p95_wait", "invalid_request_rate"):
                    np.testing.assert_array_equal(previous[metric], repeated[metric])
                trace = [
                    json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()
                ]
                rows = [
                    {
                        "step": index + 1,
                        **{
                            key: row[key]
                            for key in (
                                "requested_action_index",
                                "executed_action_index",
                                "requested_action_valid",
                                "valid_action_count",
                                "invalid_action_fallback",
                            )
                        },
                    }
                    for index, row in enumerate(trace)
                ]
                if len(rows) != 120:
                    raise ValueError("decision example must cover one complete 120-step episode")
                target = original / "decisions.csv"
                pd.DataFrame(rows).to_csv(target, index=False)
                example = previous.iloc[0]
                (original / "decisions.json").write_text(
                    json.dumps(
                        {
                            "selection": "seed 11, test_id, manifest scenario index 0; fixed for all six arms",
                            "algorithm": algo,
                            "mask_mode": mode,
                            "scenario_seed": int(example.scenario_seed),
                            "scenario_hash": example.scenario_hash,
                            "steps": 120,
                            "checkpoint_sha256": file_hash(arm / "train/best.zip"),
                            "decisions_sha256": file_hash(target),
                            "repeated_200_day_evaluation_matches": True,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                )
                print(f"Verified compact decision trace: {algo}/{mode}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    capture(parser.parse_args().source)
