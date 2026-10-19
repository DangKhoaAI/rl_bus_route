# L2.3 — Potential-based reward shaping (rejected)

**Status:** `REJECTED` after three-seed confirmation. Control retained.
**Full narrative:** [`../README.md`](../README.md)
**Ledger row:** `L2-PBRS` in [`../experiments.csv`](../experiments.csv)

## Question

Does a fixed causal potential on observed queue and excessive-wait counts densify
credit assignment and lower `total_cost_core` at a fixed budget, without changing
physics, masks or reward weights?

## Setup

- Single-group diff vs L0 core: `[shaping] enabled = false -> true`,
  `queue_weight=2.0`, `excess_weight=5.0`.
- Potential `Phi(o) = -(2*W + 5*E)/3000`, forced to 0 at terminal states.
- Training-only wrapper `r + gamma*Phi_next - Phi_now`; evaluation never wrapped.
- Weights fixed before trials from a train-only scale probe
  (std(Phi)=0.0216 vs std(reward)=0.0249).
- Budget 245,760 transitions; seeds 11/22/33; same 100 validation days as core.
- Evidence: [`../tables/l2_pbrs_summary.csv`](../tables/l2_pbrs_summary.csv),
  [`../tables/l2_pbrs_contrast.json`](../tables/l2_pbrs_contrast.json),
  [`../tables/l2_pbrs_verification.json`](../tables/l2_pbrs_verification.json),
  [`../tables/l2_pbrs_scale_probe.json`](../tables/l2_pbrs_scale_probe.json).

## Result

| Contrast (candidate − core) | Value |
|---|---:|
| Mean cost delta | +473.24 (+3.592%) |
| Paired 100-day bootstrap 95% CI | [447.66, 498.57] |
| Seeds with lower cost | 0/3 |
| Seed-11 / seed-33 P95 wait delta | +2.43 / +2.10 min |
| Seed-11 / seed-33 worst-route wait delta | +0.88 / +1.65 min |

## Decision

The CI is entirely above zero, no seed improved, and the P95/worst-route
increases exceed the registered +1.0 min limits on at least one split. Rejected by
every registered quality/service rule.

## Limits

One shaping scale was registered, not tuned on validation. The result does not
rule out other potentials or architectures.
