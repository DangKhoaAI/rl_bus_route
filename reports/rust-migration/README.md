# Rust migration artifacts

| File | Role | Regenerable |
|---|---|---|
| `oracle-manifest.json` | Frozen R0 oracle contract: schema/config hashes, seeds, discrepancies, checkpoint/report inventory. | `python scripts/export_oracle.py build` |
| `fixtures-summary.json` | Golden fixture hashes/shape summary. | `python scripts/export_oracle.py build` |
| `native-build.json` | Native toolchain, git revision, `.so` sha256, import check. | `python scripts/build_native.py` |
| `python-benchmark.json` | R0.3 unprofiled Python baseline (simulation/learn/eval). | `python scripts/benchmark_python.py` |
| `speed-acceptance.json` | R4.2 interleaved Python-vs-Rust raw timings. | `python scripts/benchmark_backends.py` |
| `profile-native.json` | R4 optimization profile: Rust training segments, torch-thread sweep, eval segments, shared-store memory. | `python scripts/profile_native_training.py` |
| `runtime-tuning.json` | Same-condition torch-thread matrix and observation-validation cost for both backends. | `python scripts/tune_runtime.py` |

This directory is excluded from the report hash inventory on purpose; it is
indexed by `reports/rust-migration.md`. Large traces/checkpoints stay in
`runs/rust-migration/` (git-ignored) and are linked by hash from the report.

## Fast local loop

- Report-only edit: `python scripts/export_oracle.py build --skip-fixtures` (~1 s)
  then `python scripts/export_oracle.py verify --hashes-only` (~1 s).
- Fixture + reference check (no 1,200-scenario regeneration): `verify` (~14 s).
- R0 acceptance / generator change: `verify --deep` (~35 s) or
  `BUS_RL_DEEP=1 python -m pytest tests/backend_parity/test_manifest.py`.

The expensive 1,200-scenario regeneration is opt-in so routine report/test loops
stay fast.
