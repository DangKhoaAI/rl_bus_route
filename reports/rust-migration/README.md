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

Default commands are the fast path; the full checks are opt-in flags.

- Default report-only edit: `python scripts/export_oracle.py build` (~1 s) then
  `python scripts/export_oracle.py verify --hashes-only` (~1 s).
- Full fixture + reference check: `python scripts/export_oracle.py build --fixtures`
  (~8 s) then `verify` (~13 s).
- R0 acceptance / generator or physical-config change: `verify --deep` (~35 s)
  or `BUS_RL_DEEP=1 python -m pytest tests/backend_parity/test_manifest.py`.
- `pytest -q` runs 4 xdist workers by default (2 torch threads each, catalog
  scenarios cached per worker) and skips the 1,200-scenario regeneration test
  unless `BUS_RL_DEEP=1`; use `-n 0` for a serial run when debugging.

The expensive 1,200-scenario regeneration is opt-in so routine report/test loops
stay fast.
