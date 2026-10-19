# Experiment ledger

One row per change/experiment. Use this table to find the decision and the
evidence path; open the linked card only if you need detail.

Status legend:

- `ACCEPTED` — frozen, retained, or used as the control.
- `REJECTED` — tested and not adopted; all numbers are in the linked card.
- `HISTORICAL` — kept for provenance; not part of the final claim.
- `INVALID` — wiring/audit failure; excluded from the matrix.

## Research changes

| ID | Change | Status | Main result | Evidence |
|---|---|---|---|---|
| E01 | Pilot: 12,288 PPO transitions | `HISTORICAL` | PPO 18,207 vs fixed 16,017 validation cost; pipeline gate only | [`pilot-study/`](../archive/pilot-study/) |
| E02 | Forecast diagnostic | `HISTORICAL` | Forecast error reported; no policy-quality claim | [`forecast-diagnostic/`](../archive/forecast-diagnostic/) |
| E03 | L0 core baseline | `ACCEPTED` (control) | 3 seeds; best validation cost 13,199.1 / 13,114.0 / 13,213.5 | [`l0-baseline-card.md`](l0-baseline-card.md) |
| E04 | L1: GAE `lambda` 0.98 / 1.0 | `REJECTED` | +0.54% / +1.78% validation cost at seed-11 screening | [`l1-gae-lambda.md`](../archive/rejected-experiments/l1-gae-lambda.md) |
| E05 | L2.1: causal arrival forecast | `REJECTED` | +2.03% cost, CI [240.38, 295.72] above 0, 1/3 seeds lower | [`l2-causal-forecast.md`](../archive/rejected-experiments/l2-causal-forecast.md) |
| E06 | L2.3: potential-based reward shaping | `REJECTED` | +3.59% cost, CI [447.66, 498.57] above 0, 0/3 seeds lower | [`l2-reward-shaping.md`](../archive/rejected-experiments/l2-reward-shaping.md) |
| E07 | L3: frozen T9 held-out matrix | `ACCEPTED` (core retained) | core lowest on ID/burst/traffic; all 3 ablations worse | [`l3-details.md`](../tables/l3-details.md) |
| — | `fairness_zero` first run | `INVALID` | native kernel ignored `run.reward`; re-run on fixed build `d6bf1551…` | [`l3-details.md`](../tables/l3-details.md) |

## Infrastructure changes

| ID | Change | Status | Main result | Evidence |
|---|---|---|---|---|
| I01 | Rust simulator kernel R0–R4 | `ACCEPTED` | native backend is default; ~3.7x simulation speedup | [`rust-migration/`](../archive/infrastructure/rust-migration/) |
| I02 | Runtime batching/caching O0–O2 | `ACCEPTED` (opt-in) | ~2x paired speedup vs Rust scalar in the measured session | [`runtime-optimization/`](../archive/infrastructure/runtime-optimization/) |
| I03 | Memory M1/M3: sparse store, lazy kernel | `ACCEPTED` | peak RSS 1,236 MB → 690 MB for a full seed | [`memory_optimize.md`](../../docs/spec/memory_optimize.md) |
| I04 | Training-time diagnosis v1.0 / v1.1 | `HISTORICAL` | Python hot-path before/after measurements | [`training-diagnosis/`](../archive/training-diagnosis/) |

## How to read a row

1. Check `Status`. `REJECTED` means “do not adopt”, not “the run failed”.
2. Open the linked card for the registered rule, setup and limits.
3. For final numbers, always use [`../FINAL_REPORT.md`](../FINAL_REPORT.md) and
   [`../tables/`](../tables/); do not copy numbers from a card that predates L3.
