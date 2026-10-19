# L1 — GAE lambda round (rejected)

**Status:** `REJECTED` at screening. Control retained.
**Final report:** [`../../README.md`](../../README.md)
**Ledger rows:** `L1-GAE-098`, `L1-GAE-100` in [`../../provenance/experiments.csv`](../../provenance/experiments.csv)

## Question

Does a less-decayed GAE trace (`lambda=0.98` or `1.0` instead of `0.95`) improve
validation cost at a fixed PPO budget, because dispatch/cooldown effects are
delayed?

## Setup

- Control: L0 core, `gae_lambda=0.95`.
- Candidates: `0.98`, `1.00`; one parameter changed, everything else fixed.
- Budget: 245,760 transitions; seed 11 screening; confirmation on seeds 22/33
  only if a candidate reached the pre-registered 2% reduction.
- Evidence: [`l1-screening.csv`](l1-screening.csv),
  [`../../provenance/core-learning-curves.csv`](../../provenance/core-learning-curves.csv).

## Result (seed 11, 100 validation days)

| Arm | Lambda | Best validation cost | Cost vs control |
|---|---:|---:|---:|
| core control | 0.95 | 13,199.1325 | — |
| GAE-098 | 0.98 | 13,270.0425 | +0.54% |
| GAE-100 | 1.00 | 13,434.0263 | +1.78% |

## Decision

Neither candidate reached the registered 2% reduction, so neither was promoted to
seeds 22/33. This is a valid no-change result; no candidate is presented as an
improvement. L1 service limits were registered before screening and were not used
to rescue a candidate.

## Limits

Validation-only screening on one seed. It supports “do not adopt”, not a claim
about held-out performance.
