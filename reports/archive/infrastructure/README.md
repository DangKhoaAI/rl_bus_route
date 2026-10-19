# Infrastructure archive

These reports are about the simulator/runtime, not about RL policy quality. They
are evidence that the Rust backend is correct enough and fast enough to run the
study.

| Topic | Report | Raw/curated evidence | Status |
|---|---|---|---|
| Rust kernel R0–R4 | [`rust-migration/README.md`](rust-migration/README.md) | [`rust-migration/artifacts/`](rust-migration/artifacts/) | accepted |
| Runtime batching/caching O0–O2 | [`runtime-optimization/README.md`](runtime-optimization/README.md) | [`runtime-optimization/artifacts/`](runtime-optimization/artifacts/) | accepted, opt-in |
| Memory M1/M3 | [`../../../docs/spec/memory_optimize.md`](../../../docs/spec/memory_optimize.md) | measurements embedded in `rust-migration/artifacts/` | accepted |
| Training-time diagnosis (legacy) | [`../training-diagnosis/`](../training-diagnosis/) | git-ignored `runs/diagnose*` | historical |

## Why there are symlinks at the reports root

`rust-migration/artifacts/oracle-manifest.json` hashes the migration narrative
`reports/rust-migration.md` and records paths under `reports/rust-migration/`.
The archive keeps the real files; the two symlinks at the reports root keep the
frozen paths resolvable:

```text
reports/rust-migration.md -> archive/infrastructure/rust-migration/README.md
reports/rust-migration/   -> archive/infrastructure/rust-migration/artifacts
```

Do not use the symlinks as reading material. Use this directory.

The frozen `rust-migration/README.md` keeps its original content and historical
relative links for hash compatibility; some of those links are stale. Use the
artifact files in `rust-migration/artifacts/` directly.

## How to verify

```bash
.venv/bin/python scripts/build_native.py   # rebuild + provenance
cargo test -p bus-sim-core                 # kernel unit tests
.venv/bin/python -m pytest -m native -q    # Python↔Rust interface contract
```

Benchmarks (not quality gates):

```bash
.venv/bin/python scripts/benchmark_backends.py
.venv/bin/python scripts/profile_native_training.py
.venv/bin/python scripts/benchmark_full_workflow.py
```

## What not to claim

- Rust speedup is infrastructure evidence; it is not multiplied into any policy
  improvement claim.
- A faster or leaner runtime does not prove a better policy on the same number of
  transitions.
