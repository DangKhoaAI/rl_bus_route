# Infrastructure index

These reports are about the simulator/runtime, not about RL policy quality. They
are evidence that the Rust backend is correct enough and fast enough to run the
study.

| Topic | Report | Raw/curated evidence | Status |
|---|---|---|---|
| Rust kernel R0–R4 | [`../rust-migration.md`](../rust-migration.md) | [`../rust-migration/`](../rust-migration/) | accepted |
| Runtime batching/caching O0–O2 | [`../runtime-optimization.md`](../runtime-optimization.md) | [`../runtime-optimization/`](../runtime-optimization/) | accepted, opt-in |
| Memory M1/M3 | [`../../docs/spec/memory_optimize.md`](../../docs/spec/memory_optimize.md) | measurements embedded in `rust-migration/` | accepted |
| Training-time diagnosis (legacy) | [`../archive/`](../archive/) | git-ignored `runs/diagnose*` | historical |

## Why these paths are at the reports root

`reports/rust-migration/oracle-manifest.json` hashes the migration narrative
`reports/rust-migration.md` and records paths under `reports/rust-migration/`.
Moving them would invalidate the frozen contract, so they stay at the root and
this directory is an index only.

## How to verify

```bash
python scripts/build_native.py       # rebuild src/bus_sim_native.so + provenance
cargo test -p bus-sim-core           # kernel unit tests
uv run pytest -m native -q           # Python↔Rust interface contract
```

Benchmarks (not quality gates):

```bash
python scripts/benchmark_backends.py
python scripts/profile_native_training.py
python scripts/benchmark_full_workflow.py
```

## What not to claim

- Rust speedup is infrastructure evidence; it is not multiplied into any policy
  improvement claim.
- A faster or leaner runtime does not prove a better policy on the same number of
  transitions.
