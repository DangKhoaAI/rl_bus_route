# Infrastructure archive

These reports are about the simulator/runtime, not about RL policy quality. They
are evidence that the Rust backend is correct enough and fast enough to run the
study.

| Topic | Report | Raw/curated evidence | Status |
|---|---|---|---|
| Rust kernel R0–R4 | [`rust-migration/README.md`](rust-migration/README.md) | [`rust-migration/artifacts/`](rust-migration/artifacts/) | accepted |
| Runtime batching/caching O0–O2 | [`runtime-optimization/README.md`](runtime-optimization/README.md) | [`runtime-optimization/artifacts/`](runtime-optimization/artifacts/) | accepted, opt-in |
| Memory M1/M3 | [`../../../docs/spec/memory_optimize.md`](../../../docs/spec/memory_optimize.md) | measurements embedded in `rust-migration/artifacts/` | accepted |

## Frozen manifest path

`rust-migration/artifacts/oracle-manifest.json` originally recorded the report at
`reports/rust-migration.md`. That path was rebased to
`reports/archive/infrastructure/rust-migration/README.md` when the archive was
created, so the manifest points at the real file. No compatibility symlinks are
kept at the reports root.

The frozen `rust-migration/README.md` keeps its original content and historical
relative links; some of those links are stale. Use the artifact files in
`rust-migration/artifacts/` directly.

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

Các script benchmark/phân tích thí nghiệm cũ đã được dọn khỏi `scripts/`; các lệnh lịch sử có thể dùng lại với source ở Git commit `fbb1c83`. Các lệnh hiện tại nằm trong `scripts/README.md` và CLI `uv run bus-rl`.
