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
cost/reward integration. R2 adds the incremental observation tensors
(`observation.rs`) and the kernel-owned mask cache. R3 adds the per-decision
contract (`reset_contract`/`step_contract`), the evaluator summary/trace
surfaces and the Python Gym wrapper (`src/bus_rl/env/native_bus_dispatch.py`)
selected via `runtime.backend`. R4 (speed/parity acceptance) is still open.
`Kernel.observe`, `Kernel.debug_snapshot` and `Kernel.debug_step` remain the
golden-test surfaces.
