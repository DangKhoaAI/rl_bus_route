# Native simulator crates

The Rust-native simulator consists of two Cargo workspace members:

```text
crates/bus-sim-core/    # optimized simulator kernel
crates/bus-sim-python/  # PyO3 bridge exposing `bus_sim_native`
```

Its reference implementation is the Python package at `src/bus_sim/oracle/`.
That oracle is **deprecated**: the native kernel is the default and maintained
backend, and the two implementations are no longer kept in numerical lockstep.
Selecting the oracle requires `runtime.backend = "python"` together with
`runtime.legacy_python = true` (CLI: `--backend python --legacy-python`), and it
warns on use. What is still enforced is the **FFI interface**: the suite under
`tests/integration/bus_rl/execution/environments/rust/` pins the payload schema,
array shapes/dtypes, error mapping, ownership and lifecycle at the
`bus_sim_native` gate. Physics parity lives in the frozen evidence under
`reports/rust-migration/`, and the kernel's own invariants in `cargo test`.

## Build and test

```bash
python scripts/build_native.py       # build and install src/bus_sim_native.so
cargo test -p bus-sim-core           # 12 kernel unit tests
uv run pytest -m native -q           # Python↔Rust interface contract
```

The build installs the ignored extension at `src/bus_sim_native.so` and writes
a provenance record under `reports/rust-migration/` by default. `backend`
defaults to `rust`; running without the extension fails explicitly. The
deprecated oracle tests are skipped unless `pytest --legacy-python` is passed.

The Gym adapters are grouped by implementation:

```text
src/bus_rl/execution/environments/python/  # calls `bus_sim.oracle` (deprecated)
src/bus_rl/execution/environments/rust/    # calls `bus_sim_native` (default)
```

`Kernel.observe`, `Kernel.debug_snapshot`, and `Kernel.debug_step` remain the
debug/inspection surfaces; the Rust FFI suite treats `reset_contract`,
`step_contract`, `episode_summary_inputs` and the `BatchKernel` batch methods
as the contract the Python integration layer depends on.
