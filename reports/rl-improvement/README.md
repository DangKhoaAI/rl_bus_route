# RL improvement artifacts

This directory contains the frozen L0/L1/L2/L3 protocol and curated evidence. Raw
checkpoints, per-day validation rows and telemetry are under the gitignored
`runs/rl-improvement/<experiment>/<seed>/<run-id>/` tree.

## Frozen protocol

- [`protocol.json`](protocol.json) records the Rust binary/config/data hashes,
  budgets, splits, diagnostics contract, checkpoint rule and acceptance gates.
- [`l3_protocol.json`](l3_protocol.json) freezes the L3 T9 matrix, held-out
  splits, service limits, analysis rules and fixed-wall-clock accounting before
  any held-out split was opened.
- [`experiments.csv`](experiments.csv) is the pre-registered experiment ledger;
  every L1/L2 candidate must have a complete card before its first run. It now
  contains the L2.1 causal-forecast, L2.3 PBRS and L3 T9 cards with their
  decisions.
- [`baseline.md`](baseline.md) contains the validation-only L0 diagnosis and
  next-step decision.
- `tables/verification.json` is the post-run structural audit for L0/L1 steps,
  episode counts, paired day counts, native hashes and finite-check failures.
- `tables/l2_forecast_*` and `tables/l2_pbrs_*` are the curated L2 results,
  decisions, audits and the PBRS train-only scale probe.
- `tables/l3_config_audit.json` is the frozen-matrix single-field diff audit;
  `tables/l3_held_out_summary.csv`, `l3_contrasts.csv`, `l3_paired_days.csv`,
  `l3_baselines_summary.csv` and `l3_verification.json` are the paired held-out
  results, statistics, baseline summary and structural audit.
- `plots/l3_validation_curves.png`, `plots/l3_held_out_costs.png` and
  `plots/l3_service_tradeoffs.png` are the L3 figures; failure traces live in
  `evidence/l3_failure_traces.json` (compact digest; raw decision rows under
  the gitignored `runs/rl-improvement/l3-failure-traces/`).
- `uv run python scripts/l3_analysis.py` regenerates every L3 table and plot
  from the raw `runs/rl-improvement/l3-*` tree;
  `uv run python scripts/l3_failure_traces.py` regenerates the failure traces.

## L3 reward-wiring fix

The L3 audit found the native `Kernel`/`BatchKernel` ignored `run.reward`. The
fix ( `set_reward` in `crates/bus-sim-python/src/lib.rs` plus the Python bridge)
is in the same commit as the L3 results; the earlier invalid `fairness_zero` runs
are retained under `runs/rl-improvement/t9-fairness_zero/` and are excluded from
the matrix. L3 uses native build `d6bf1551…` for all 12 arms.

The R4/runtime reports remain historical infrastructure evidence. Their seed-11
runs are not reused as L0 quality evidence because they predate the frozen L0
telemetry and use different runtime/instrumentation settings.
