# Test conventions

`pytest` collects `tests/` (`-n 4` by default, 2 torch threads per worker).
Two levels, two naming rules.

## Unit — `tests/unit/<src path>/test_<module>.py`

- One file per `src` module, path mirrors `src` exactly.
  `src/bus_sim/oracle/engine.py` → `tests/unit/bus_sim/oracle/test_engine.py`.
- The module under test is the unit; other imports must be pure data/utilities,
  not subsystems under test.
- `tests/unit/bus_sim/oracle/` is a 10/10 mirror of `src/bus_sim/oracle/`; keep it
  that way. `tests/unit/bus_rl/**` mirrors its modules the same way.

## Integration — `tests/integration/<src path of the entry point>/test_<subject>[_<aspect>].py`

Integration has no single module to mirror, so:

- **Directory** mirrors the `src` location of the *entry point* (the caller or the
  boundary), so editing that file leads you to the tests.
- **Filename** names the interaction, not a module: `test_<subject>[_<aspect>].py`.
  - `subject` = the facade/boundary exercised: `cli`, `environment`, `kernel`,
    `batch`, `evaluator`, `baselines`, `training`.
  - `aspect` = optional, from the closed set
    `contract` · `payload` · `lifecycle` · `mapping` · `determinism` · `flow`.
  - Add `aspect` only when one subject needs several files. If a file needs two
    aspects, split the file.
- Never reuse a bare `src` module name at this level: an integration
  `test_engine.py` falsely implies a 1:1 unit and is wrong.

Examples:

```text
tests/integration/bus_rl/execution/environments/rust/
├── conftest.py                  # fixtures for this directory
├── contract.py                  # frozen interface constants (not a test module)
├── test_scenario_payload.py
├── test_kernel_contract.py
├── test_batch_contract.py
└── test_evaluator_payloads.py
```

## Deciding the level

If the test must import **two or more first-party top-level packages** and cannot
be satisfied by one module's public API plus stubs, it is integration — place it
under `tests/integration/` and name it by subject.

## Shared helpers

`tests/support/` holds fixtures/builders (no `test_` prefix). Test-scoped
`conftest.py` supplies fixtures and collection hooks only.

## Markers

| Marker | Meaning | How it runs |
|---|---|---|
| `native` | needs the `bus_sim_native` extension | skips when `.so` is absent; `pytest -m native` |
| `legacy_python` | exercises the deprecated Python oracle | skipped by default; `pytest --legacy-python` |

Legacy tests are auto-marked by path in `tests/conftest.py`
(`LEGACY_TEST_GLOBS`); update that list when such a file moves.

## Running

```bash
uv run pytest -q                     # native default; legacy skipped
uv run pytest -q --legacy-python     # include the deprecated-oracle tests
uv run pytest -q -n 0                # serial, for debugging
uv run pytest -m native -q           # Python↔Rust FFI contract only
```

## Do not

- Add tests for `src/bus_sim/oracle/` or `src/bus_rl/execution/environments/python/`
  expecting bug fixes: that code is deprecated and receives no further fixes.
- Put oracle-only checks into `integration/` or cross-module checks into `unit/`.
- Commit test data: `tests/` holds code only (fixtures live in `reports/` or are
  regenerated).
