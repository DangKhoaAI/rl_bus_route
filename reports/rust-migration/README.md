# Rust migration artifacts

Frozen evidence for the (now accepted) R0-R4 migration. `runs/` is git-ignored,
so the curated JSON summaries below are committed while raw traces and
checkpoints stay in `runs/rust-migration/`.

| File | Role | Regenerable |
|---|---|---|
| `oracle-manifest.json` | Frozen R0 oracle contract: schema/config hashes, seeds, discrepancies, checkpoint/report inventory. | archived (the export tooling was retired after acceptance) |
| `native-build.json` | Native toolchain, git revision, `.so` sha256, import check. | `python scripts/build_native.py` |
| `python-benchmark.json` | R0.3 unprofiled Python baseline (simulation/learn/eval). | `python scripts/benchmark_python.py` |
| `speed-acceptance.json` | R4.2 interleaved Python-vs-Rust raw timings. | `python scripts/benchmark_backends.py` |
| `profile-native.json` | R4 optimization profile: Rust training segments, torch-thread sweep, eval segments, shared-store memory. | `python scripts/profile_native_training.py` |
| `runtime-tuning.json` | Same-condition torch-thread matrix and observation-validation cost for both backends. | `python scripts/tune_runtime.py` |
| `full-workflow.json`, `full-workflow-memory.json` | R4.3 full-seed wall times, parity, validation-cost ablation and memory breakdown. | `python scripts/benchmark_full_workflow.py` |

The one-off R0 golden fixtures and the Python-level native parity suite were
retired after acceptance (see the top-level README). This directory is a frozen
snapshot, indexed by `reports/rust-migration.md`.
