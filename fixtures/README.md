# Frozen test fixtures

Data only — no code lives here. `tests/` holds test scripts; the frozen
reference data they compare against lives at the repository root so the test
tree stays pure code and the large immutable artifacts are easy to spot.

- `golden/` — frozen oracle scenarios (`input.json`, `expected.npz`,
  `expected_events.json`). Replayed by `tests/parity/oracle/test_golden.py`,
  `tests/parity/oracle/test_semantics.py` and the native parity tests.
- `reference/` — committed copy of the historical `runs/diagnose-after`
  checkpoint plus provenance (`reference.json`). It is the paired-evaluation
  anchor for the Rust migration. See `reference/README.md`.

Regenerate / verify (from the repository root):

```text
python scripts/export_oracle.py build --fixtures --force   # rewrite golden/
python scripts/export_oracle.py verify                     # re-hash everything
```
