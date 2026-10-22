# Reproducible Run Evidence

This tracked directory contains compact outputs from fixed experiment matrices.
Full local state is under the ignored `experiments/` directory. Final study reports
are authored by humans under `reports/` using the evidence here.

| Study | Configuration | Evidence |
|---|---|---|
| `standard-v1` | MLP10, GRU10, Belief20-fade; three seeds; 245,760 steps/seed | [Index](standard-v1/README.md) |

Each study records config snapshots, scenario identities, training histories,
per-day evaluations, comparison tables, figures and artifact SHA256 checksums.
Checkpoints, optimizer state, raw validation telemetry and diagnostic logs are
excluded. Verify with `uv run bus-rl artifact verify --path runs/standard-v1`.

The standard matrix reuses the established reference dataset. It is reproducible
run evidence; it does not automatically update the accepted historical claims in
`reports/`. See [the directory contract](../docs/run-artifacts.md).
