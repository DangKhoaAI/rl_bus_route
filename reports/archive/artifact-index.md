# Artifact index (historical)

This was the index for the old `reports/rl-improvement/` tree. The files now live
in different places:

| Old artifact | New location |
|---|---|
| `protocol.json` | [`../provenance/protocol.json`](../provenance/protocol.json) |
| `l3_protocol.json` | [`../provenance/l3_protocol.json`](../provenance/l3_protocol.json) |
| `baseline.md` | [`../provenance/l0-diagnosis.md`](../provenance/l0-diagnosis.md) |
| `experiments.csv` | [`../provenance/experiments.csv`](../provenance/experiments.csv) |
| `tables/verification.json` | [`../provenance/l0-verification.json`](../provenance/l0-verification.json) |
| `tables/l3_*` | [`../tables/`](../tables/) |
| `plots/l3_*` | [`../figures/`](../figures/) |
| `evidence/l3_failure_traces.json` | [`../provenance/l3-failure-traces.json`](../provenance/l3-failure-traces.json) |
| `tables/l2_forecast_*`, `tables/l2_pbrs_*` | [`rejected-experiments/`](rejected-experiments/) |

## Reproduction

```bash
.venv/bin/python scripts/l3_analysis.py              # raw L3 tables + figures
.venv/bin/python scripts/build_benchmark_tables.py   # final benchmark tables
.venv/bin/python scripts/l3_failure_traces.py        # failure-trace digest
```

Raw checkpoints, per-day rows and telemetry are under the git-ignored
`runs/rl-improvement/<experiment>/<seed>/<run-id>/` tree.
