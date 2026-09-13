# RL improvement artifacts

This directory contains the frozen L0/L1/L2 protocol and curated evidence. Raw
checkpoints, per-day validation rows and telemetry are under the gitignored
`runs/rl-improvement/<experiment>/<seed>/<run-id>/` tree.

## Frozen protocol

- [`protocol.json`](protocol.json) records the Rust binary/config/data hashes,
  budgets, splits, diagnostics contract, checkpoint rule and acceptance gates.
- [`experiments.csv`](experiments.csv) is the pre-registered experiment ledger;
  every L1/L2 candidate must have a complete card before its first run. It now
  contains the L2.1 causal-forecast card and its REJECT decision.
- [`baseline.md`](baseline.md) contains the validation-only L0 diagnosis and
  next-step decision.
- `tables/verification.json` is the post-run structural audit for L0/L1 steps,
  episode counts, paired day counts, native hashes and finite-check failures.
- `tables/l2_forecast_summary.csv`, `tables/l2_forecast_contrast.json` and
  `tables/l2_forecast_verification.json` are the curated L2.1 results, the
  paired contrast/acceptance decision and the L2 structural audit.

The R4/runtime reports remain historical infrastructure evidence. Their seed-11
runs are not reused as L0 quality evidence because they predate the frozen L0
telemetry and use different runtime/instrumentation settings.
