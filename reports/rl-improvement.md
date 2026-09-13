# RL improvement study status

Status: **L0 and L1 accepted with core retained; both selected L2 directions
(causal forecast, PBRS) rejected on three-seed confirmation.**

The active study follows [`docs/spec/improve_RL.md`](../docs/spec/improve_RL.md)
and [`docs/plan/improve_RL.md`](../docs/plan/improve_RL.md). The frozen protocol,
experiment cards and raw-artifact index are under
[`reports/rl-improvement/`](rl-improvement/).

## Acceptance evidence

- L0 runtime/protocol: [`rl-improvement/protocol.json`](rl-improvement/protocol.json)
- L0 diagnosis: [`rl-improvement/baseline.md`](rl-improvement/baseline.md)
- L0 tables/curve: `rl-improvement/tables/` and `rl-improvement/plots/`
- L1 ledger and decisions: [`rl-improvement/experiments.csv`](rl-improvement/experiments.csv)
- L1 screening table: `rl-improvement/tables/l1_screening_summary.csv`
- L2.1 forecast summary/contrast/audit: `rl-improvement/tables/l2_forecast_summary.csv`,
  `l2_forecast_contrast.json`, `l2_forecast_verification.json`
- L2.3 PBRS summary/contrast/audit/scale probe: `rl-improvement/tables/l2_pbrs_summary.csv`,
  `l2_pbrs_contrast.json`, `l2_pbrs_verification.json`, `l2_pbrs_scale_probe.json`
- Structural run audit: `rl-improvement/tables/verification.json`

Three core Rust runs completed at 245,760 transitions and 100 validation days
for seeds 11/22/33. Each has checkpoints, metadata, 20 learning-curve points,
2,000 paired per-day validation rows, and diagnostics with finite checks for
logits/values/returns/losses/gradients. No held-out split was read.

The selected L1 group was GAE lambda, registered as `{0.95 control, 0.98,
1.0}` with all other PPO, physics, reward, mask, runtime and budget settings
fixed. Seed-11 screening completed for both candidates. Best validation costs:

| arm | lambda | best cost | delta vs control | registered outcome |
|---|---:|---:|---:|---|
| core control | 0.95 | 13,199.1325 | 0.000% | control retained |
| GAE-098 | 0.98 | 13,270.0425 | -0.537% reduction | REJECT at screening |
| GAE-100 | 1.00 | 13,434.0263 | -1.780% reduction | REJECT at screening |

Neither candidate reached the registered 2% screening minimum, so neither was
shortlisted for seeds 22/33. This is a valid L1 no-change decision; no candidate
is presented as a confirmed improvement. L1 service limits were registered
before screening and were not used to rescue either candidate.

## L2.1 causal forecast (selected first L2 direction, REJECT)

Diagnostic trigger: L0 shows wait cost dominates (`waiting_pm` ~9,140 vs
`onboard_pm` ~10,100 at wait weight 1.0) and the policy observes only a 5x2-minute
past arrival history; the `forecast` observation slot was identically zero. The
registered hypothesis was that a causal 15-minute-ahead arrival forecast would
let the policy anticipate demand and lower `total_cost_core` without changing
physics, masks or reward.

Card `L2-FORECAST` in [`rl-improvement/experiments.csv`](rl-improvement/experiments.csv)
was registered before the runs. Single-group diff vs `L0-CORE`:
`[forecast] enabled = false -> true`; the forecaster is fit on the 500 train
arrival tapes only, frozen, and saved as `forecaster.npz` beside each checkpoint
(same SHA-256 for all seeds: `5a95908a...`, confirming train-only provenance).
All other settings, seeds, 245,760-transition budget and 100 validation days
match the control.

Best-checkpoint validation on the same 100 days:

| arm | seed | best transition | cost | P95 wait | worst-route wait |
|---|---:|---:|---:|---:|---:|
| L0-CORE | 11 | 159,744 | 13,199.1325 | 11.325 | 5.704 |
| L0-CORE | 22 | 98,304 | 13,113.9725 | 11.103 | 5.667 |
| L0-CORE | 33 | 233,472 | 13,213.4563 | 11.376 | 5.744 |
| L2-FORECAST | 11 | 221,184 | 13,520.5813 | 13.133 | 7.259 |
| L2-FORECAST | 22 | 147,456 | 13,620.8613 | 12.978 | 6.077 |
| L2-FORECAST | 33 | 110,592 | 13,189.2700 | 11.360 | 5.715 |

Contrast (candidate - control), averaging model seeds per validation day first:
mean `delta = +268.05` (`+2.034%`), paired 100-day bootstrap 95% CI
`[240.38, 295.72]` entirely above zero, lower cost on only 1/3 seeds. Seed 11
increases P95 wait by 1.81 min and worst-route wait by 1.56 min, both outside the
pre-registered +1.0 min limits. The candidate is rejected by every registered
quality/service rule and the core control is retained.

Structural audit (`l2_forecast_verification.json`): all six runs reached
245,760 transitions, 2,048 episodes, 20 evaluation points and 100 unique days;
zero finite-check failures; native hash matches the frozen L0 build. Forecast
observation means were ~0.042 for every candidate seed (never identically zero),
so the negative result is a policy-quality outcome, not a wiring failure.
The 2,048-transition smoke, the causality unit suite and the native
`test_forecast_flow.py` wiring/causality check all passed before the full runs.
The frozen forecaster's causal error is MAE 0.01906 (bias 2.7e-05) on validation
and MAE 0.01908 (bias 1.9e-04) on train (`l2_forecast_error.json`); the good
forecast error did not translate into better control, and this report does not
read MAE as a policy-quality result.

Unselected L2 directions (encoder/memory, curriculum or BC warm-start, PopArt)
remain SKIPPED/DEFERRED: L0 diagnostics do not trigger them.

## L2.3 potential-based reward shaping (selected second L2 direction, REJECT)

Diagnostic trigger: L0 shows the same dominant `waiting_pm` cost and the delayed
dispatch/cooldown effects it named for the L1 GAE round; L1 then rejected both
`gae_lambda` candidates, so the credit-assignment explanation was still open. The
registered hypothesis was that a fixed causal potential on observed queue and
excessive-wait counts would densify credit assignment and lower
`total_cost_core` at fixed B without changing physics, masks or reward weights.

Card `L2-PBRS` was registered before the runs. Single-group diff vs `L0-CORE`:
`[shaping] enabled = false -> true` with `queue_weight=2.0`, `excess_weight=5.0`.
The potential is `Phi(o) = -(2*W + 5*E)/3000` from the observation's waiting (`W`)
and excessive-wait (`E`) counts, forced to 0 at terminal states; the training-only
`PotentialShapingVecEnv` returns `r + gamma*Phi_next - Phi_now`. Evaluation is
never wrapped and `total_cost_core` is recomputed from raw components. The weights
were fixed before trials from a train-only probe (`l2_pbrs_scale_probe.json`:
`std(Phi)=0.0216` vs `std(per-step reward)=0.0249`) and were not tuned on
validation. Policy invariance holds for the shaped return under a fixed
initial-state distribution because `Phi(s_T)=0` (Ng et al., 1999).

Best-checkpoint validation on the same 100 days:

| arm | seed | best transition | cost | P95 wait | worst-route wait |
|---|---:|---:|---:|---:|---:|
| L0-CORE | 11 | 159,744 | 13,199.1325 | 11.325 | 5.704 |
| L0-CORE | 22 | 98,304 | 13,113.9725 | 11.103 | 5.667 |
| L0-CORE | 33 | 233,472 | 13,213.4563 | 11.376 | 5.744 |
| L2-PBRS | 11 | 73,728 | 14,119.9975 | 13.759 | 6.580 |
| L2-PBRS | 22 | 135,168 | 13,197.7900 | 11.287 | 5.713 |
| L2-PBRS | 33 | 122,880 | 13,628.5075 | 13.480 | 7.391 |

Contrast (candidate - control), averaging model seeds per validation day first:
mean `delta = +473.24` (`+3.592%`), paired 100-day bootstrap 95% CI
`[447.66, 498.57]` entirely above zero, lower cost on 0/3 seeds. Seeds 11 and 33
increase P95 wait by 2.43 and 2.10 min and worst-route wait by 0.88 and 1.65 min;
the P95 and seed-33 worst-route deltas are outside the registered +1.0 min
limits. The candidate is rejected by every registered quality/service rule and
the core control is retained.

Structural audit (`l2_pbrs_verification.json`): all six runs reached 245,760
transitions, 2,048 episodes, 20 evaluation points and 100 unique days; zero
finite-check failures; native hash matches the frozen L0 build; metadata records
the shaping config. The unit telescoping/terminal-zero/causality suite and the
native `test_shaping_flow.py` reward-accounting test passed before the full runs.

L2.2 encoder/memory, L2.4 curriculum/BC warm-start and L2.5 PopArt remain
SKIPPED/DEFERRED: L0 diagnostics do not trigger them (flat MLP has no measured
sample-efficiency failure; PPO already beats heuristics; explained variance is
high), and the first-round limit of two L2 directions is now used.

## Compute and limitations

The three core runs consumed approximately 327.6 s wall time in concurrent
launches; the two screening candidates consumed approximately 184.0 s. Exact
per-run `wall_time_s`, transitions, backend/build hashes and effective runtime
flags are in each run's `metadata.json`. The host was shared, so these walls are
reported as observed run measurements, not a cross-session speed claim.
The three L2 forecast runs consumed 76.3, 81.5 and 81.6 s wall time including
forecaster fitting and per-step prediction (forecast extra compute is inside
these observed walls, not a separate claim). The three L2 PBRS runs consumed
99.6, 81.7 and 103.7 s; shaping adds one O(observation) potential per training
step and no extra transitions. Exact values are in each run's `metadata.json`,
`l2_forecast_verification.json` and `l2_pbrs_verification.json`.

The Python oracle, historical R4/runtime runs, and incomplete `runs/core-11`
checkpoint were not reused as L0 quality evidence. Held-out ID/OOD evaluation,
T9 ablations and fixed-wall-clock comparison remain unopened. No global defaults
were changed and no candidate was combined or adopted.

## Next task

**Single next task: STOP algorithm exploration and freeze this L0/L1/L2 decision
for L3 planning.** Two L2 directions (causal forecast and PBRS) were selected and
rejected; the remaining L2 directions (encoder/memory, warm-start, PopArt and all
other L2 tasks) stay SKIPPED/DEFERRED and are not missing implementation work.
The next opening phase is L3, which requires its own registered held-out matrix
and must not reuse any test split read here (none was read).
