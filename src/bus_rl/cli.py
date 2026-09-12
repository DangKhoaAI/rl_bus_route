"""Reproducible CLI: generate, baseline, profile, train, evaluate, report."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from bus_rl.config import ControlConfig, RuntimeConfig, load_run_config, parse_counts
from bus_rl.data.io import load_manifest, load_split, save_manifest
from bus_rl.data.scenario import generate_manifest
from bus_rl.env.factory import make_env_for_run
from bus_rl.evaluation.plots import render_plots
from bus_rl.evaluation.profile import run_profile
from bus_rl.evaluation.runner import evaluate_scenarios, write_results
from bus_rl.forecasting.historical import HistoricalForecaster
from bus_rl.provenance import physical_config_hash, require_fresh_output
from bus_rl.rewards.costs import RewardConfig
from bus_rl.training.checkpoint import load_metadata, load_model
from bus_rl.training.diagnose import diagnose_train
from bus_rl.training.train import fit_algorithm, train_run


def _run_config(args) -> object:
    run = load_run_config(Path(args.config))
    backend = getattr(args, "backend", None)
    if backend and backend != run.runtime.backend:
        run = replace(run, runtime=RuntimeConfig(backend=backend))
    return run


def _maybe_forecaster(args, run, train_scenarios=None):
    if not run.forecast.enabled:
        return None
    logs = [scenario.arrival_tape for scenario in train_scenarios or []]
    if not logs:
        if not getattr(args, "manifest", None):
            raise ValueError("forecast training requires a manifest with a train split")
        logs = [scenario.arrival_tape for scenario in load_split(Path(args.manifest), "train")]
    forecaster = HistoricalForecaster(tick_s=run.physical.tick_s)
    forecaster.fit(logs, run.physical)
    return forecaster


def cmd_generate(args) -> None:
    run = _run_config(args)
    output = Path(args.output)
    require_fresh_output(output)
    counts = parse_counts(args.counts)
    splits = {
        split: generate_manifest(split, count, run.physical) for split, count in counts.items()
    }
    save_manifest(output, splits, run.physical)


def cmd_baseline(args) -> None:
    run = _run_config(args)
    output = Path(args.output)
    require_fresh_output(output)
    methods = [name.strip() for name in args.methods.split(",") if name.strip()]
    scenarios = load_split(Path(args.manifest), args.split)
    if args.limit:
        scenarios = scenarios[: args.limit]
    frames = []
    traces = {}
    for method in methods:
        frame, method_traces = evaluate_scenarios(
            scenarios,
            run,
            method,
            split=args.split,
            trace_index=0 if args.trace else None,
        )
        frames.append(frame)
        traces.update({(method, index): rows for index, rows in method_traces.items()})
    import pandas as pd

    write_results(pd.concat(frames, ignore_index=True), output, traces or None)


def cmd_profile(args) -> None:
    run = _run_config(args)
    output = Path(args.output)
    require_fresh_output(output)
    rows = run_profile(output, run.physical, decisions=args.decisions)
    print(json.dumps(rows, indent=2))


def cmd_train(args) -> None:
    run = _run_config(args)
    algorithm = fit_algorithm(
        run.algorithm,
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
    )
    run = replace(run, algorithm=algorithm)
    train_scenarios = load_split(Path(args.manifest), "train")
    try:
        val_scenarios = load_split(Path(args.manifest), "validation")
    except ValueError:
        val_scenarios = train_scenarios[:1]
    forecaster = _maybe_forecaster(args, run, train_scenarios)
    train_run(
        run,
        train_scenarios,
        val_scenarios,
        Path(args.output),
        forecaster=forecaster,
        eval_limit=args.eval_limit,
    )


def cmd_diagnose(args) -> None:
    run = _run_config(args)
    algorithm = fit_algorithm(
        run.algorithm,
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
    )
    run = replace(run, algorithm=algorithm)
    started = perf_counter()
    train_scenarios = load_split(Path(args.manifest), "train")
    try:
        val_scenarios = load_split(Path(args.manifest), "validation")
    except ValueError:
        val_scenarios = train_scenarios[:1]
    load_s = perf_counter() - started
    print(
        f"[diagnose] load_manifest {load_s:.3f}s  train_days={len(train_scenarios)}",
        flush=True,
    )
    summary = diagnose_train(
        run,
        train_scenarios,
        val_scenarios,
        Path(args.output),
        eval_limit=args.eval_limit or 10,
    )
    print(
        json.dumps({k: summary[k] for k in ("setup_s", "learn_s", "eval_s", "projected")}, indent=2)
    )


def _reward_from_metadata(payload: dict) -> RewardConfig:
    allowed = RewardConfig.__dataclass_fields__
    return RewardConfig(**{key: payload[key] for key in allowed if key in payload})


def cmd_evaluate(args) -> None:
    run = _run_config(args)
    output = Path(args.output)
    require_fresh_output(output)
    scenarios = load_split(Path(args.manifest), args.split)
    if args.limit:
        scenarios = scenarios[: args.limit]
    manifest = load_manifest(Path(args.manifest))
    if manifest.get("physical_config_hash") != physical_config_hash(run.physical):
        raise ValueError("physical config does not match the dataset manifest")
    model = None
    model_seed = args.seed
    method = args.method
    forecaster = None
    if args.checkpoint:
        metadata = load_metadata(Path(args.checkpoint))
        run = replace(
            run,
            control=ControlConfig(**metadata["control"]),
            reward=_reward_from_metadata(metadata["reward"]),
            forecast=replace(run.forecast, enabled=bool(metadata.get("forecast_enabled"))),
        )
        frozen = Path(args.checkpoint).parent / "forecaster.npz"
        if frozen.exists():
            forecaster = HistoricalForecaster.load(frozen)
        elif run.forecast.enabled:
            forecaster = _maybe_forecaster(args, run)
        env = make_env_for_run(scenarios[:1], run, forecaster=forecaster)
        model, metadata = load_model(
            Path(args.checkpoint), env, run.physical, backend=run.runtime.backend
        )
        model_seed = metadata.get("seed", model_seed)
        method = "ppo"
    else:
        forecaster = _maybe_forecaster(args, run)
    frame, traces = evaluate_scenarios(
        scenarios,
        run,
        method,
        model=model,
        model_seed=model_seed,
        split=args.split,
        forecaster=forecaster,
        trace_index=0 if args.trace else None,
    )
    write_results(frame, output, traces or None)


def cmd_report(args) -> None:
    import pandas as pd

    output = Path(args.output)
    require_fresh_output(output)
    frame = pd.read_csv(args.results)
    traces = []
    events = Path(args.results).parent / "events.jsonl"
    if events.exists():
        traces = [json.loads(line) for line in events.read_text().splitlines() if line]
    history = None
    if args.history:
        history = pd.read_csv(args.history)
        if "seed" not in history.columns:
            history["seed"] = 0
    render_plots(frame, output, traces=traces, history=history)
    summary = frame.groupby("method", dropna=False)[
        [
            column
            for column in ("total_cost_core", "mean_wait", "abandoned_share")
            if column in frame
        ]
    ].mean(numeric_only=True)
    (output / "summary.json").write_text(summary.to_json(indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bus-rl")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate")
    generate.add_argument("--config", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--counts")
    generate.set_defaults(func=cmd_generate)

    baseline = sub.add_parser("baseline")
    baseline.add_argument("--config", required=True)
    baseline.add_argument("--manifest", required=True)
    baseline.add_argument("--split", default="validation")
    baseline.add_argument("--methods", default="fixed,threshold,proportional")
    baseline.add_argument("--output", required=True)
    baseline.add_argument("--trace", action="store_true")
    baseline.add_argument("--limit", type=int)
    baseline.add_argument("--backend", choices=("python", "rust"))
    baseline.set_defaults(func=cmd_baseline)

    profile = sub.add_parser("profile")
    profile.add_argument("--config", required=True)
    profile.add_argument("--decisions", type=int, default=120)
    profile.add_argument("--output", required=True)
    profile.set_defaults(func=cmd_profile)

    train = sub.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--manifest", required=True)
    train.add_argument("--seed", type=int)
    train.add_argument("--output", required=True)
    train.add_argument("--timesteps", type=int)
    train.add_argument("--n-envs", type=int, dest="n_envs")
    train.add_argument("--eval-limit", type=int, dest="eval_limit")
    train.add_argument("--backend", choices=("python", "rust"))
    train.set_defaults(func=cmd_train)

    diagnose = sub.add_parser("diagnose")
    diagnose.add_argument("--config", required=True)
    diagnose.add_argument("--manifest", required=True)
    diagnose.add_argument("--seed", type=int, default=11)
    diagnose.add_argument("--output", required=True)
    diagnose.add_argument("--timesteps", type=int, default=2048)
    diagnose.add_argument("--n-envs", type=int, dest="n_envs")
    diagnose.add_argument("--eval-limit", type=int, dest="eval_limit", default=10)
    diagnose.add_argument("--backend", choices=("python", "rust"))
    diagnose.set_defaults(func=cmd_diagnose)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--split", default="validation")
    evaluate.add_argument("--checkpoint")
    evaluate.add_argument("--method", default="ppo")
    evaluate.add_argument("--seed", type=int)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--trace", action="store_true")
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--backend", choices=("python", "rust"))
    evaluate.set_defaults(func=cmd_evaluate)

    report = sub.add_parser("report")
    report.add_argument("--results", required=True)
    report.add_argument("--output", required=True)
    report.add_argument("--history")
    report.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except FileExistsError as error:
        print(error, file=sys.stderr)
        raise SystemExit(2) from error
    except ValueError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
