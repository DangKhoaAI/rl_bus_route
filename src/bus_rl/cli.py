"""Noun/action CLI for data, models, baselines, reports and performance tests."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from bus_rl.config import ControlConfig, load_run_config, parse_counts
from bus_rl.execution.scenarios.generation import generate_manifest
from bus_rl.execution.scenarios.io import load_manifest, load_split, save_manifest
from bus_rl.provenance import physical_config_hash, require_fresh_output
from bus_sim.oracle.costs import RewardConfig


def _run_config(args) -> object:
    run = load_run_config(Path(args.config))
    algorithm_updates = {
        name: value
        for name in (
            "name",
            "mask_mode",
            "encoder",
            "history_minutes",
            "belief_prior_shape",
            "belief_prior_rate",
            "belief_decay",
        )
        if (value := getattr(args, name, None)) is not None
    }
    if algorithm_updates:
        run = replace(run, algorithm=replace(run.algorithm, **algorithm_updates))
    backend = getattr(args, "backend", None)
    legacy_python = bool(getattr(args, "legacy_python", False))
    if backend:
        run = replace(
            run,
            runtime=replace(run.runtime, backend=backend, legacy_python=legacy_python),
        )
    elif legacy_python:
        run = replace(run, runtime=replace(run.runtime, legacy_python=True))
    torch_threads = getattr(args, "torch_threads", None)
    if torch_threads is not None:
        run = replace(run, algorithm=replace(run.algorithm, torch_threads=torch_threads))
    runtime_updates: dict = {}
    if getattr(args, "eval_batch_size", None) is not None:
        runtime_updates["eval_batch_size"] = int(args.eval_batch_size)
    if getattr(args, "reuse_eval_pool", False):
        runtime_updates["reuse_eval_pool"] = True
    if getattr(args, "native_batch", False):
        runtime_updates["native_batch"] = True
    if getattr(args, "validate_distributions", None) is not None:
        runtime_updates["validate_distributions"] = bool(args.validate_distributions)
    if runtime_updates:
        run = replace(run, runtime=replace(run.runtime, **runtime_updates))
    _ = run.arrival_memory_bins  # Reject incompatible encoder/history before creating output.
    return run


def _maybe_forecaster(args, run, train_scenarios=None):
    if not run.forecast.enabled:
        return None
    from bus_rl.learning.forecasting import HistoricalForecaster

    logs = [scenario.arrival_tape for scenario in train_scenarios or []]
    if not logs:
        if not getattr(args, "manifest", None):
            raise ValueError("forecast training requires a manifest with a train split")
        logs = [scenario.arrival_tape for scenario in load_split(Path(args.manifest), "train")]
    forecaster = HistoricalForecaster(tick_s=run.physical.tick_s)
    forecaster.fit(logs, run.physical)
    return forecaster


def _load_scenarios(args, run, split: str):
    """Load a split for a run.

    Rust runs skip the dense tapes: the native store owns them and reads each
    scenario once from disk, so the run never holds all 600 tapes in Python.
    Forecast needs the tapes, so it keeps the full scenarios.
    """
    meta_only = run.runtime.backend == "rust" and not run.forecast.enabled
    return load_split(Path(args.manifest), split, with_tapes=not meta_only)


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
    from bus_rl.evaluation.report import write_evaluation_metadata
    from bus_rl.evaluation.runner import evaluate_scenarios, write_results

    run = _run_config(args)
    output = Path(args.output)
    require_fresh_output(output)
    methods = [name.strip() for name in args.methods.split(",") if name.strip()]
    scenarios = _load_scenarios(args, run, args.split)
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

    frame = pd.concat(frames, ignore_index=True)
    write_results(frame, output, traces or None)
    write_evaluation_metadata(output, frame, run, Path(args.manifest))


def cmd_profile(args) -> None:
    from bus_rl.evaluation.profile import run_profile

    run = _run_config(args)
    output = Path(args.output)
    require_fresh_output(output)
    rows = run_profile(output, run.physical, decisions=args.decisions)
    print(json.dumps(rows, indent=2))


def cmd_train(args) -> None:
    from bus_rl.learning.training.train import fit_algorithm, train_run

    run = _run_config(args)
    algorithm = fit_algorithm(
        run.algorithm,
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
    )
    run = replace(run, algorithm=algorithm)
    train_scenarios = _load_scenarios(args, run, "train")
    try:
        val_scenarios = _load_scenarios(args, run, "validation")
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
    from bus_rl.learning.training.diagnose import diagnose_train
    from bus_rl.learning.training.train import fit_algorithm

    run = _run_config(args)
    algorithm = fit_algorithm(
        run.algorithm,
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
    )
    run = replace(run, algorithm=algorithm)
    started = perf_counter()
    train_scenarios = _load_scenarios(args, run, "train")
    try:
        val_scenarios = _load_scenarios(args, run, "validation")
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
    from bus_rl.evaluation.report import write_evaluation_metadata
    from bus_rl.evaluation.runner import evaluate_scenarios, write_results
    from bus_rl.execution.environments.factory import make_env_for_run
    from bus_rl.learning.forecasting import HistoricalForecaster
    from bus_rl.learning.training.algorithms import learned_method
    from bus_rl.learning.training.checkpoint import load_metadata, load_model

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
    method = args.method or "ppo"
    forecaster = None
    metadata = None
    if args.checkpoint:
        metadata = load_metadata(Path(args.checkpoint))
        run = replace(
            run,
            control=ControlConfig(**metadata["control"]),
            reward=_reward_from_metadata(metadata["reward"]),
            forecast=replace(run.forecast, enabled=bool(metadata.get("forecast_enabled"))),
            algorithm=replace(
                run.algorithm,
                name=metadata.get("algorithm", {}).get("name", "ppo"),
                mask_mode=metadata.get("mask_mode", "apply"),
                encoder=metadata.get("algorithm", {}).get("encoder", "mlp"),
                history_minutes=metadata.get("algorithm", {}).get("history_minutes", 10),
                belief_prior_shape=metadata.get("algorithm", {}).get("belief_prior_shape", 1.0),
                belief_prior_rate=metadata.get("algorithm", {}).get("belief_prior_rate", 1.0),
                belief_decay=metadata.get("algorithm", {}).get("belief_decay", 1.0),
            ),
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
        method = learned_method(run.algorithm.name, run.algorithm.mask_mode)
        if args.method is not None and args.method not in {
            method,
            "maskable_ppo" if method == "ppo" else method,
        }:
            raise ValueError("explicit method conflicts with checkpoint algorithm")
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
    write_evaluation_metadata(
        output,
        frame,
        run,
        Path(args.manifest),
        Path(args.checkpoint) if args.checkpoint else None,
        metadata,
    )


def cmd_report(args) -> None:
    from bus_rl.evaluation.report import generate_report

    generate_report(args.results, Path(args.output), args.history, rl_method=args.rl_method)


def cmd_experiment(args) -> None:
    from bus_rl.experiments import run_experiment

    run_experiment(
        Path(args.config),
        Path(args.manifest),
        Path(args.output),
        Path(args.artifacts) if args.artifacts else None,
        args.resume,
    )


def cmd_artifact_export(args) -> None:
    from bus_rl.artifacts import export_artifacts

    result = export_artifacts(Path(args.source), Path(args.output))
    print(f"Exported {len(result['files'])} files ({result['total_bytes'] / 1024**2:.1f} MiB)")


def cmd_artifact_verify(args) -> None:
    from bus_rl.artifacts import verify_artifacts

    result = verify_artifacts(Path(args.path))
    print(f"Verified {len(result['files'])} files ({result['total_bytes'] / 1024**2:.1f} MiB)")


def _add_runtime_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        dest="eval_batch_size",
        help="O1: evaluate this many scenarios per batched predict (default from config)",
    )
    parser.add_argument(
        "--reuse-eval-pool",
        action="store_true",
        help="O1: reuse one bounded evaluator pool across validation calls",
    )
    parser.add_argument(
        "--native-batch",
        action="store_true",
        help="O2: step eval/train envs through one native BatchKernel call (rust only)",
    )
    parser.add_argument(
        "--legacy-python",
        action="store_true",
        help="opt in to the deprecated Python oracle backend (use with `--backend python`)",
    )
    parser.add_argument(
        "--validate-distributions",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="validate_distributions",
        help="Torch distribution argument validation (default from config: on)",
    )


def _add_encoder_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--encoder",
        choices=("mlp", "gru", "belief", "belief_gru"),
        help="policy encoder; default GRU10, explicit flags override TOML",
    )
    parser.add_argument(
        "--history-minutes",
        type=int,
        help="arrival history in minutes (>=10, multiple of control interval; default 10)",
    )
    parser.add_argument("--belief-prior-shape", type=float, help="Gamma prior shape (default 1)")
    parser.add_argument("--belief-prior-rate", type=float, help="Gamma prior rate (default 1)")
    parser.add_argument(
        "--belief-decay", type=float, help="old-bin weight multiplier in (0,1] (default 1)"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bus-rl")
    groups = parser.add_subparsers(dest="group", required=True)
    data_actions = groups.add_parser("data", help="manage scenario data").add_subparsers(
        dest="action", required=True
    )
    model_actions = groups.add_parser("model", help="train and evaluate policies").add_subparsers(
        dest="action", required=True
    )
    baseline_actions = groups.add_parser(
        "baseline", help="run programmed controllers"
    ).add_subparsers(dest="action", required=True)
    report_actions = groups.add_parser(
        "report", help="generate plots and summaries"
    ).add_subparsers(dest="action", required=True)
    test_actions = groups.add_parser(
        "test", help="diagnose and profile performance"
    ).add_subparsers(dest="action", required=True)
    experiment_actions = groups.add_parser(
        "experiment", help="run fixed experiment matrices"
    ).add_subparsers(dest="action", required=True)
    artifact_actions = groups.add_parser(
        "artifact", help="export and verify Git evidence"
    ).add_subparsers(dest="action", required=True)

    experiment = experiment_actions.add_parser(
        "run", help="train, evaluate and export a fixed matrix"
    )
    experiment.add_argument("--config", default="configs/studies/standard.toml")
    experiment.add_argument("--manifest", default="data/generated/base/manifest.json")
    experiment.add_argument("--output", required=True)
    experiment.add_argument("--artifacts", help="fresh Git evidence path, normally under runs/")
    experiment.add_argument(
        "--resume",
        action="store_true",
        help="reuse completed stages only when the frozen plan matches",
    )
    experiment.set_defaults(func=cmd_experiment)
    export = artifact_actions.add_parser(
        "export", help="export evidence without weights or diagnostic logs"
    )
    export.add_argument("--source", required=True)
    export.add_argument("--output", required=True)
    export.set_defaults(func=cmd_artifact_export)
    verify = artifact_actions.add_parser("verify", help="verify all exported evidence checksums")
    verify.add_argument("--path", required=True)
    verify.set_defaults(func=cmd_artifact_verify)

    generate = data_actions.add_parser("generate", help="generate a scenario manifest and tapes")
    generate.add_argument("--config", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--counts")
    generate.set_defaults(func=cmd_generate)

    baseline = baseline_actions.add_parser("run", help="evaluate fixed and heuristic controllers")
    baseline.add_argument("--config", required=True)
    baseline.add_argument("--manifest", required=True)
    baseline.add_argument("--split", default="validation")
    baseline.add_argument("--methods", default="fixed,threshold,proportional")
    baseline.add_argument("--output", required=True)
    baseline.add_argument("--trace", action="store_true")
    baseline.add_argument("--limit", type=int)
    baseline.add_argument(
        "--backend", choices=("python", "rust"), help="default rust; python is deprecated"
    )
    baseline.add_argument("--torch-threads", type=int, dest="torch_threads")
    _add_runtime_flags(baseline)
    baseline.set_defaults(func=cmd_baseline)

    profile = test_actions.add_parser("profile", help="microprofile simulator execution")
    profile.add_argument("--config", required=True)
    profile.add_argument("--decisions", type=int, default=120)
    profile.add_argument("--output", required=True)
    profile.set_defaults(func=cmd_profile)

    train = model_actions.add_parser(
        "train", help="train masked PPO, A2C or REINFORCE (default PPO + GRU10)"
    )
    train.add_argument(
        "--config", default="configs/train.toml", help="TOML config (default: configs/train.toml)"
    )
    train.add_argument("--manifest", required=True)
    train.add_argument("--seed", type=int)
    train.add_argument("--output", required=True)
    train.add_argument("--timesteps", type=int)
    train.add_argument("--n-envs", type=int, dest="n_envs")
    train.add_argument("--eval-limit", type=int, dest="eval_limit")
    train.add_argument(
        "--backend", choices=("python", "rust"), help="default rust; python is deprecated"
    )
    train.add_argument("--torch-threads", type=int, dest="torch_threads")
    _add_runtime_flags(train)
    _add_encoder_flags(train)
    train.add_argument("--algo", choices=("ppo", "a2c", "reinforce"), dest="name")
    train.set_defaults(func=cmd_train)

    diagnose = test_actions.add_parser("diagnose", help="diagnose training and evaluation timings")
    diagnose.add_argument(
        "--config", default="configs/train.toml", help="TOML config (default: configs/train.toml)"
    )
    diagnose.add_argument("--manifest", required=True)
    diagnose.add_argument("--seed", type=int, default=11)
    diagnose.add_argument("--output", required=True)
    diagnose.add_argument("--timesteps", type=int, default=2048)
    diagnose.add_argument("--n-envs", type=int, dest="n_envs")
    diagnose.add_argument("--eval-limit", type=int, dest="eval_limit", default=10)
    diagnose.add_argument(
        "--backend", choices=("python", "rust"), help="default rust; python is deprecated"
    )
    diagnose.add_argument("--torch-threads", type=int, dest="torch_threads")
    _add_runtime_flags(diagnose)
    _add_encoder_flags(diagnose)
    diagnose.add_argument("--algo", choices=("ppo", "a2c", "reinforce"), dest="name")
    diagnose.set_defaults(func=cmd_diagnose)

    evaluate = model_actions.add_parser("evaluate", help="evaluate a saved checkpoint")
    evaluate.add_argument(
        "--config", default="configs/eval.toml", help="TOML config (default: configs/eval.toml)"
    )
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--split", default="validation")
    evaluate.add_argument("--checkpoint")
    evaluate.add_argument("--method")
    evaluate.add_argument("--seed", type=int)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--trace", action="store_true")
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument(
        "--backend", choices=("python", "rust"), help="default rust; python is deprecated"
    )
    evaluate.add_argument("--torch-threads", type=int, dest="torch_threads")
    _add_runtime_flags(evaluate)
    evaluate.set_defaults(func=cmd_evaluate)

    report = report_actions.add_parser(
        "generate", help="generate tables, figures and provenance from results"
    )
    report.add_argument("--results", required=True, nargs="+", help="RL and baseline result CSVs")
    report.add_argument("--output", required=True)
    report.add_argument(
        "--history",
        nargs="+",
        help="training histories; otherwise loaded from evaluation artifacts",
    )
    report.add_argument(
        "--rl-method", help="method treated as RL (default: infer ppo/maskable_ppo/core)"
    )
    report.set_defaults(func=cmd_report)
    return parser


LEGACY_COMMANDS = {
    "generate": ("data", "generate"),
    "train": ("model", "train"),
    "evaluate": ("model", "evaluate"),
    "baseline": ("baseline", "run"),
    "report": ("report", "generate"),
    "diagnose": ("test", "diagnose"),
    "profile": ("test", "profile"),
}


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Keep archived commands runnable, while help only advertises noun/action syntax.
    # baseline/report are also new group names, so only rewrite their flat flag form.
    if argv and argv[0] in LEGACY_COMMANDS:
        flat = argv[0] not in {"baseline", "report"} or (
            len(argv) > 1 and argv[1].startswith("--") and argv[1] != "--help"
        )
        if flat:
            argv = [*LEGACY_COMMANDS[argv[0]], *argv[1:]]
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
