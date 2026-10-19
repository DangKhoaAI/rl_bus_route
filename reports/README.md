# Reports

Curated evidence for the bus fleet-control study. This directory answers one
question: **what is the final claim, and what proves it?** Raw run data lives in
the git-ignored `runs/` tree; code and specs live outside `reports/`.

## Start here

1. [`FINAL_REPORT.md`](FINAL_REPORT.md) — config, final result, limitations (1–2 pages).
2. [`tables/`](tables/) — the exact benchmark and ablation tables.
3. [`figures/`](figures/) — the final three figures.
4. [`provenance/`](provenance/) — protocols, hashes, verification, ledger.
5. [`archive/`](archive/) — everything historical or rejected.

If you only have one minute: read `FINAL_REPORT.md`.

## Structure

```text
reports/
├── README.md                  # this file
├── FINAL_REPORT.md            # final answer, config, result, limits
├── figures/                   # final figures only
├── tables/                    # final tables + column guide
├── provenance/                # protocols, verification, ledger, L0 detail
├── archive/                   # pilot, forecast, rejected L1/L2, infra, old diagnosis
├── rust-migration.md ->       # compatibility symlink for the frozen oracle manifest
└── rust-migration/ ->         # compatibility symlink for the frozen oracle manifest
```

The two symlinks point into `archive/infrastructure/rust-migration/`. They exist
only because the frozen `oracle-manifest.json` records the old paths. Do not use
them as reading material; use [`archive/infrastructure/`](archive/infrastructure/).

## Status legend

- `ACCEPTED` — frozen, retained, or used as the control.
- `REJECTED` — tested and not adopted; the card records why.
- `HISTORICAL` — kept for provenance; not part of the final claim.
- `INVALID` — wiring/audit failure; excluded from the matrix.

The per-change table is [`provenance/experiments-ledger.md`](provenance/experiments-ledger.md).

## What belongs where

| Kind of file | Location |
|---|---|
| Final claim, config, main numbers | `FINAL_REPORT.md` |
| Human-facing benchmark/ablation tables | `tables/*.csv` |
| Final figures | `figures/*.png` |
| Protocols, hashes, verification JSON, ledger, L0 detail | `provenance/` |
| Rejected experiments, pilot, forecast diagnostic, old diagnosis, infrastructure | `archive/` |
| Raw checkpoints, telemetry, full traces | git-ignored `runs/` |

## How to add something

1. If it changes the final claim, update `FINAL_REPORT.md` and `tables/`.
2. If it is a new experiment, put the card and its tables under
   `archive/rejected-experiments/` (unless it is adopted, then update the final
   tables and add it to `provenance/experiments-ledger.md`).
3. If it is provenance, put it under `provenance/` and describe it in
   `provenance/README.md`.
4. Do not edit accepted benchmark tables or the frozen `rust-migration*` files;
   add a new analysis file instead.

## Frozen hash scope

`reports/archive/infrastructure/rust-migration/artifacts/oracle-manifest.json`
hashes `reports/rust-migration.md` and records paths under
`reports/rust-migration/`. The symlinks above keep those paths resolvable. Do
not move the archive copy without updating the manifest.
