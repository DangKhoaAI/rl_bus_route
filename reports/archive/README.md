# Archive

Everything that is **not part of the final claim**, kept for provenance. Each
folder says what was run; open a file only if you need that history.

| Path | What it is | Status |
|---|---|---|
| [`pilot-12k-pipeline-gate/`](pilot-12k-pipeline-gate/) | 12,288-transition pilot: checks the pipeline, not policy quality | `HISTORICAL` |
| [`rejected-changes/`](rejected-changes/) | L1 GAE, L2 causal forecast, L2 reward shaping: cards + tables | `REJECTED` |
| [`python-training-speed-before-after/`](python-training-speed-before-after/) | Python training-speed measurements before/after hot-path work | `HISTORICAL` |
| [`infrastructure/`](infrastructure/) | Rust migration, runtime batching, memory notes | `ACCEPTED` (infra) |

## Reading guide

- Final result: [`../README.md`](../README.md).
- Rejected changes: [`rejected-changes/README.md`](rejected-changes/README.md).
- Rust/runtime: [`infrastructure/README.md`](infrastructure/README.md).

Raw artifacts for the historical runs live in the git-ignored `runs/diagnose/`,
`runs/diagnose-after/`, `runs/rust-migration/` and `runs/rl-improvement/` trees.
