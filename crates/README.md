# Native kernel (`bus-sim`)

Rust port of the Python oracle in `src/bus_rl`. Layout follows the migration
spec section 3.1:

```text
crates/bus-sim/       # rlib `bus_sim_core`: domain, engine, passengers,
                      # vehicles, dispatcher, travel, guards, costs, snapshots
crates/bus-sim-py/    # cdylib `bus_sim`: PyO3 bridge (debug/parity surface)
```

## Build and install

```text
python scripts/build_native.py     # cargo build --release + install src/bus_sim.so
cargo test -p bus-sim              # native unit tests
python -m pytest tests/backend_parity -q
```

The build also writes `reports/rust-migration/native-build.json` (toolchain,
git revision, `.so` hash, import check). `target/` and `src/bus_sim.so` are
git-ignored; the release build is not part of the Python package install.

## Scope

R1 implements the domain, lifecycle, actions/guards, the fixed tick order and
cost/reward integration. Observation tensors, mask caching and the Gym wrapper
are R2/R3; the PyO3 surface here is deliberately a thin debug bridge
(`Kernel.debug_snapshot`, `Kernel.debug_step`, `Kernel.action_mask`).
