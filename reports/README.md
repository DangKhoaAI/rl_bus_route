# Reports index

`reports/` holds committed, curated, human-facing outputs: analysis write-ups,
aggregated result tables, figures, and the frozen migration contract. Raw
per-invocation artifacts (checkpoints, event logs, profiler dumps) belong in
`runs/`, which is git-ignored. When a raw artifact must be committed as evidence
(because `runs/` is not tracked), it goes under `<study>/evidence/` and the
report must say why; do not keep duplicate copies in two places.

## Contents

| Path | Role |
|---|---|
| `rust-migration.md` | Rust migration analysis, parity matrix, benchmarks, R0–R4 checklist (frozen contract narrative). |
| `rust-migration/` | Frozen oracle contract (manifest/hashes) + provenance + benchmark JSON. |
| `pilot.md` | Pilot training/evaluation analysis (Task 8). |
| `pilot/results.csv` | Aggregated validation results for fixed/threshold/proportional/PPO. |
| `pilot/plots/` | Rendered pilot figures and `summary.json`. |
| `forecast.md` | Forecast diagnostic (Task 10). |
| `training-diagnosis-v1.0.md` | Python training-time diagnosis before hot-path work. |
| `training-diagnosis-v1.1-python-improve.md` | Python training-time diagnosis after hot-path work. |
| `rl-improvement.md` | L0/L1/L2 study status, decisions and reproduction evidence links. |
| `rl-improvement/` | Frozen L0/L1 protocol, experiment ledger and curated RL-improvement evidence (L0/L1 and the rejected L2.1 forecast round). |

The diagnosis runs' raw artifacts live in the git-ignored `runs/diagnose/`
and `runs/diagnose-after/`; the two `training-diagnosis-*.md` summaries keep the
derived before/after comparison.

## Hash scope

`reports/rust-migration/oracle-manifest.json` freezes only the migration
contract narrative (`reports/rust-migration.md`) plus the contract/provenance
files under `reports/rust-migration/`. Analysis reports, figures and aggregated
tables are intentionally **not** hashed: regenerating a plot must not drift the
frozen oracle.
