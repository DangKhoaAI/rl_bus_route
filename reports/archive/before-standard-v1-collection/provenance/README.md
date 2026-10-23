# Provenance

Machine-facing evidence for the final claim. You normally do not need to read
this directory; use [`../README.md`](../README.md) and
[`../tables/`](../tables/) instead.

| File | What it records |
|---|---|
| [`protocol.json`](protocol.json) | Frozen L0 protocol: runtime, config, data, budget, evaluation, diagnostics |
| [`l3_protocol.json`](l3_protocol.json) | Frozen L3 matrix, held-out splits, service limits, analysis rules |
| [`l0-diagnosis.md`](l0-diagnosis.md) | L0 validation diagnosis and next-step decision |
| [`l0-baseline-card.md`](l0-baseline-card.md) | Short L0 control card |
| [`l0-baseline-summary.json`](l0-baseline-summary.json) | L0 core/heuristic summary |
| [`l0-validation-summary.csv`](l0-validation-summary.csv) | Per-seed L0 validation metrics |
| [`l0-verification.json`](l0-verification.json) | L0 structural audit |
| [`l3-verification.json`](l3-verification.json) | L3 structural audit: 200 rows/day, paired hashes, native hash |
| [`l3-config-audit.json`](l3-config-audit.json) | Confirms each ablation differs from core in one field |
| [`l3-failure-traces.json`](l3-failure-traces.json) | Digest of the highest-cost core held-out days |
| [`experiments.csv`](experiments.csv) | Pre-registered experiment ledger (machine-readable) |
| [`experiments-ledger.md`](experiments-ledger.md) | Human-readable change ledger with status and result |
| [`core-learning-curves.csv`](core-learning-curves.csv) | L0 core learning-curve points |
| [`core_learning_curves.png`](core_learning_curves.png) | L0 core learning-curve plot |
| [`intro-sources.md`](intro-sources.md) | Source and verification status for each number in the Vietnamese intro page |

## Trust chain

1. `README.md` claims are computed from `tables/*.csv`.
2. `tables/*.csv` are produced by `scripts/l3_analysis.py` and
   `scripts/build_benchmark_tables.py`.
3. `l3-verification.json` and `l3-config-audit.json` check the matrix structure.
4. `l3_protocol.json` was frozen before the held-out splits were opened.
5. Raw runs are under git-ignored `runs/rl-improvement/`.

## Conventions

- `delta = candidate − core`; positive means the candidate is more expensive.
- Cost units are `total_cost_core` (normalized by `N_ref = 3000`).
- Wait units are minutes.
- Shares are fractions in `[0, 1]`.
- CIs are paired 200-day day-bootstrap, 95%, 2,000 resamples, seed 6001.

## Frozen migration contract

The Rust migration evidence is under
[`../archive/infrastructure/rust-migration/`](../archive/infrastructure/rust-migration/).
The frozen `oracle-manifest.json` records the report path at
`../archive/infrastructure/rust-migration/README.md`.
