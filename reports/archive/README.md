# Archive

Everything that is **not part of the final claim**, kept for provenance. Names
are descriptive so you can tell what a file is without opening it.

| Path | What it is | Status |
|---|---|---|
| [`pilot-study/`](pilot-study/) | 12,288-transition pilot: pipeline gate, not a quality claim | `HISTORICAL` |
| [`forecast-diagnostic/`](forecast-diagnostic/) | Forecast error only; the policy test is in `rejected-experiments/` | `HISTORICAL` |
| [`rejected-experiments/`](rejected-experiments/) | L1 GAE, L2 causal forecast, L2 reward shaping: cards + tables | `REJECTED` |
| [`training-diagnosis/`](training-diagnosis/) | Python training-time measurements before/after hot-path work | `HISTORICAL` |
| [`infrastructure/`](infrastructure/) | Rust migration, runtime batching, memory notes | `ACCEPTED` (infra) |
| [`full-study-narrative.md`](full-study-narrative.md) | The long L0–L3 study report (superseded by `FINAL_REPORT.md`) | `HISTORICAL` |
| [`artifact-index.md`](artifact-index.md) | Old artifact index and reproduction commands | `HISTORICAL` |

## Reading guide

- For the final result: [`../FINAL_REPORT.md`](../FINAL_REPORT.md).
- For rejected changes: start with
  [`rejected-experiments/README.md`](rejected-experiments/README.md).
- For Rust/runtime: [`infrastructure/README.md`](infrastructure/README.md).
- For the old full narrative: [`full-study-narrative.md`](full-study-narrative.md).

Raw artifacts for the historical runs live in the git-ignored `runs/diagnose/`,
`runs/diagnose-after/`, `runs/rust-migration/` and `runs/rl-improvement/` trees.
