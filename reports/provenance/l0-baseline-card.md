# L0 — Core baseline freeze (accepted control)

**Status:** `ACCEPTED` — this is the control for every later change.
**Full narrative:** [`../archive/l0-l3-full-report.md`](../archive/l0-l3-full-report.md)
**Diagnosis:** [`l0-diagnosis.md`](l0-diagnosis.md)
**Protocol:** [`protocol.json`](protocol.json)

## Question

Can the Rust-backed MaskablePPO core be trained reproducibly for three seeds,
with valid diagnostics and a fixed checkpoint rule, before any algorithm change
is tested?

## Setup

- Rust backend, 245,760 transitions per seed, seeds 11/22/33.
- 100 validation days for checkpoint selection; held-out splits not read.
- `total_cost_core` is the selection metric; tie → earliest checkpoint.
- Evidence: [`l0-verification.json`](l0-verification.json),
  [`core-learning-curves.csv`](core-learning-curves.csv),
  [`core_learning_curves.png`](core_learning_curves.png).

## Result

| Seed | Best validation cost |
|---:|---:|
| 11 | 13,199.1325 |
| 22 | 13,113.9725 |
| 33 | 13,213.4563 |

All runs reached 245,760 transitions and 2,048 episodes, with 20 evaluation
points, 2,000 paired per-day validation rows, and finite diagnostics
(logits/values/returns/losses/gradients).

## Decision

Frozen as L0-CORE. L1 and L2 candidates are compared against it; L3 keeps it as
the incumbent unless a registered candidate passes the adoption rules.

## Limits

Validation-only baseline. It says nothing about held-out quality by itself; the
L3 matrix provides that evidence.
