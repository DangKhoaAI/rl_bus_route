# Native simulator crates

The Rust-native simulator consists of two Cargo workspace members:

```text
crates/bus-sim-core/    # optimized simulator kernel
crates/bus-sim-python/  # PyO3 bridge exposing `bus_sim_native`
```

Its reference implementation is the Python package at `src/bus_sim/oracle/`.
Golden fixtures and differential tests in `tests/parity/` ensure that both
implementations remain behaviorally equivalent.

## Build and test

```bash
python scripts/build_native.py
cargo test -p bus-sim-core
python -m pytest tests/parity -q
```

The build installs the ignored extension at `src/bus_sim_native.so` and writes
a provenance record under `reports/rust-migration/` by default. Requesting the
Rust backend without that extension fails explicitly; Python remains the
default selected by `runtime.backend = "python"`.

The Gym adapters are grouped by implementation:

```text
src/bus_rl/execution/environments/python/  # calls `bus_sim.oracle`
src/bus_rl/execution/environments/rust/    # calls `bus_sim_native`
```

`Kernel.observe`, `Kernel.debug_snapshot`, and `Kernel.debug_step` remain the
golden-test surfaces.
