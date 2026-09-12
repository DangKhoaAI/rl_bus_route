# Fixed reference checkpoint

Committed copy of the historical `runs/diagnose-after/last.zip` (2,048
transitions, seed 11, `configs/experiments/core.toml`). It is the paired
evaluation anchor for the Rust migration (R0.2/R0.3): evaluating it on the first
10 validation days reproduces the historical mean `total_cost = 15471.25`.

Provenance and hashes are in `../reference.json`; the full oracle manifest is
`reports/rust-migration/oracle-manifest.json`.

Reproduce:

```text
python scripts/export_oracle.py verify
```
