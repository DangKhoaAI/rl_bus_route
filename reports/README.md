# Reports

Curated, human-facing evidence for the bus fleet-control study. Raw per-run
artifacts (checkpoints, event logs, telemetry, profiler dumps) live in the
git-ignored `runs/` tree; this directory keeps only what a reader needs.

**Start here:** [`FINAL.md`](FINAL.md) — one to two pages with the final claim,
main numbers and limitations.

## Reading paths

| Time | Read |
|---|---|
| 1 minute | [`FINAL.md`](FINAL.md) |
| 5 minutes | [`FINAL.md`](FINAL.md) → [`LEDGER.md`](LEDGER.md) |
| 30 minutes | [`FINAL.md`](FINAL.md) → [`LEDGER.md`](LEDGER.md) → [`studies/03-rl-improvement/l3-matrix/README.md`](studies/03-rl-improvement/l3-matrix/README.md) → [`infra/README.md`](infra/README.md) |
| Full study | [`studies/03-rl-improvement/README.md`](studies/03-rl-improvement/README.md) |
| Problem and RL formulation | [`../docs/problem_solution_brief.md`](../docs/problem_solution_brief.md), [`../docs/spec/spec_v1.0.md`](../docs/spec/spec_v1.0.md) |

## Status legend

- `ACCEPTED` — frozen, retained, or used as the control.
- `REJECTED` — tested and not adopted; the card records why.
- `HISTORICAL` — kept for provenance; not part of the final claim.
- `INVALID` — wiring/audit failure; excluded from the matrix.

The authoritative per-change table is [`LEDGER.md`](LEDGER.md).

## Directory map

```text
reports/
├── README.md                  # this reading guide
├── FINAL.md                   # final claim, main numbers, limitations
├── LEDGER.md                  # one row per change/experiment + status
├── studies/
│   ├── 01-pilot/              # historical plumbing gate (12,288 transitions)
│   ├── 02-forecast/           # historical forecast diagnostic
│   └── 03-rl-improvement/     # L0–L3 study
│       ├── README.md          # full study narrative
│       ├── ARTIFACTS.md       # artifact index + reproduction commands
│       ├── l0-baseline/       # decision card: core control
│       ├── l1-gae/            # decision card: GAE lambda (rejected)
│       ├── l2-forecast/       # decision card: causal forecast (rejected)
│       ├── l2-pbrs/           # decision card: reward shaping (rejected)
│       ├── l3-matrix/         # decision card: frozen T9 held-out matrix
│       ├── protocol.json      # frozen L0 protocol
│       ├── l3_protocol.json   # frozen L3 matrix/held-out protocol
│       ├── baseline.md        # L0 diagnosis
│       ├── experiments.csv    # pre-registered experiment ledger
│       ├── tables/            # aggregated results and statistics
│       ├── plots/             # selected figures
│       └── evidence/          # committed evidence (failure-trace digest)
├── infra/                     # infrastructure index (frozen paths live at root)
├── archive/                   # historical diagnosis reports
├── rust-migration.md          # FROZEN: R0–R4 migration narrative
├── rust-migration/            # FROZEN: oracle manifest + provenance JSON
├── runtime-optimization.md    # runtime batching/caching report
└── runtime-optimization/      # runtime benchmark JSON
```

## Studies

| Path | Question | Status |
|---|---|---|
| [`studies/01-pilot/`](studies/01-pilot/) | Does the full pipeline run end to end? | `HISTORICAL` |
| [`studies/02-forecast/`](studies/02-forecast/) | Is a causal forecast usable? | `HISTORICAL` |
| [`studies/03-rl-improvement/l0-baseline/`](studies/03-rl-improvement/l0-baseline/) | Is the core baseline reproducible? | `ACCEPTED` |
| [`studies/03-rl-improvement/l1-gae/`](studies/03-rl-improvement/l1-gae/) | Does a less-decayed GAE trace help? | `REJECTED` |
| [`studies/03-rl-improvement/l2-forecast/`](studies/03-rl-improvement/l2-forecast/) | Does a 15-min forecast help the policy? | `REJECTED` |
| [`studies/03-rl-improvement/l2-pbrs/`](studies/03-rl-improvement/l2-pbrs/) | Does potential-based shaping help? | `REJECTED` |
| [`studies/03-rl-improvement/l3-matrix/`](studies/03-rl-improvement/l3-matrix/) | Core vs ablations vs baselines on held-out days | `ACCEPTED` |

## Infrastructure

Frozen paths are intentionally kept at the reports root because the oracle
manifest hashes them:

- [`rust-migration.md`](rust-migration.md) and [`rust-migration/`](rust-migration/) —
  R0–R4 Rust migration, parity and speed evidence.
- [`runtime-optimization.md`](runtime-optimization.md) and
  [`runtime-optimization/`](runtime-optimization/) — batching/caching benchmarks.
- Memory optimization is specified in
  [`../docs/spec/memory_optimize.md`](../docs/spec/memory_optimize.md).

See [`infra/README.md`](infra/README.md) for the one-page index.

## Archive

[`archive/`](archive/) holds the two Python training-diagnosis reports. They are
historical measurements from before the Rust backend became the default; they are
kept because the specs cite them, not as current evidence.

## Adding a report

1. One folder = one question/change, under `studies/<NN>-<slug>/`.
2. Use the fixed card shape: Question / Setup / Result / Decision / Evidence / Limits.
3. Add one row to [`LEDGER.md`](LEDGER.md) with status and main result.
4. Only update [`FINAL.md`](FINAL.md) if the **final** claim changes; otherwise
   link to the card.
5. Do not edit accepted L3 tables or the frozen `rust-migration*` files. Add a new
   analysis file instead.
6. Raw artifacts stay in `runs/`; if one must be committed, put it under the
   study's `evidence/` and say why.

## Hash scope

`reports/rust-migration/oracle-manifest.json` freezes only the migration contract
narrative (`reports/rust-migration.md`) plus the contract/provenance files under
`reports/rust-migration/`. Analysis reports, figures and aggregated tables are
intentionally **not** hashed: regenerating a plot must not drift the frozen
oracle. For that reason, `rust-migration*` and `runtime-optimization*` stay at
the reports root; do not move them without updating the frozen manifest.
