# Archive

Everything that is **not part of the final claim**, kept for provenance. Folder
names say what was run; open a file only if you need that history.

| Path | What it is | Status |
|---|---|---|
| [`pilot-12k-pipeline-gate/`](pilot-12k-pipeline-gate/) | 12,288-transition pilot: checks the pipeline, not policy quality | `HISTORICAL` |
| [`forecast-error-baseline.md`](forecast-error-baseline.md) | Historical forecast MAE/bias only; the policy test was rejected | `HISTORICAL` |
| [`rejected-experiments/`](rejected-experiments/) | L1 GAE, L2 causal forecast, L2 reward shaping: cards + tables | `REJECTED` |
| [`python-training-speed-before-after/`](python-training-speed-before-after/) | Python training-speed measurements before/after hot-path work | `HISTORICAL` |
| [`infrastructure/`](infrastructure/) | Rust migration, runtime batching, memory notes | `ACCEPTED` (infra) |
| [`l0-l3-full-report.md`](l0-l3-full-report.md) | Long L0–L3 report (historical; the root README is the short final) | `HISTORICAL` |

## Reading guide

- Final result: [`../README.md`](../README.md).
- Rejected changes: [`rejected-experiments/README.md`](rejected-experiments/README.md).
- Rust/runtime: [`infrastructure/README.md`](infrastructure/README.md).
- Long study history: [`l0-l3-full-report.md`](l0-l3-full-report.md).

Raw artifacts for the historical runs live in the git-ignored `runs/diagnose/`,
`runs/diagnose-after/`, `runs/rust-migration/` and `runs/rl-improvement/` trees.
