# L0 baseline diagnosis

Status: **L0 accepted; one bounded L1 round registered.**

Protocol: [`protocol.json`](protocol.json). Source artifacts are gitignored run
outputs under `runs/rl-improvement/`; the curated tables are in `tables/`.
No held-out split was read.

## L0.1: frozen runtime and inventory

The study uses the accepted Rust backend and the research-only config
`configs/experiments/rl-improvement/core.toml`. The loaded native library hash
recorded by every run is
`2d5fa9e3d7672966277efbdba9fc179bdfe6cd8e3bd72f3b60e730d65192fe26`.
See each run's `metadata.json` and the frozen protocol for the complete value.

The effective study settings are Rust, CPU Torch threads 2, native batch
training, evaluator batch 16 with pool reuse, distribution validation off as an
explicit opt-in, observation validation on, and no forecast. The core PPO
settings are 4 environments, 256-step rollouts, batch 256, 4 epochs, learning
rate 3e-4, entropy coefficient .01, gamma 1, GAE lambda .95 and clip range .2.
Budget is 245,760 transitions, validation every 12,288 transitions on 100
validation scenarios, seeds 11/22/33. Best is lowest `total_cost_core`, with
first checkpoint on a tie.

Historical checkpoints were not reused: the R4/runtime seed-11 runs lack this
telemetry and/or use an incompatible runtime protocol; `runs/core-11/best.zip`
is incomplete because it has no metadata. This inventory is recorded in
`protocol.json`.

## L0.2: diagnostics and correctness evidence

The study callback writes `training_diagnostics.jsonl`, with policy entropy and
normalized entropy on the valid action support, K=1 handling, per-family valid
slots/opportunities/probability mass/conditional selection, observation/reward/
return/advantage summaries, PPO entropy/KL/clip/value/explained-variance/loss
telemetry, and finite checks for logits, values, returns, losses and gradients.
Invalid K=0, nonfinite values, or probability mass on masked actions fail the
run rather than becoming quality results. `evaluations.csv` stores the learning
curve and `validation_metrics.csv` stores paired raw per-day rows.

The diagnostics unit suite passed (4 tests). A 2,048-transition native smoke
produced finite telemetry and checkpoint metadata. Failure traces are selected
before inspection: the first finite/mask failure, or the three highest-cost
validation days at each selected best checkpoint. All three full runs contain
20 validation points, 2,000 per-day rows, 240 PPO updates, 960 policy samples,
2,048 completed training episodes, and zero recorded finite-check failures.

Selected aggregate telemetry (mean across registered samples):

| seed | mean K | mean H | mean H/log(K) | mean approx KL | mean clip fraction | mean explained variance | mean return std |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 3.682 | 0.774 | 0.699 | 0.00322 | 0.0159 | 0.981 | 1.678 |
| 22 | 3.363 | 0.584 | 0.563 | 0.00335 | 0.0158 | 0.977 | 1.446 |
| 33 | 3.588 | 0.744 | 0.682 | 0.00374 | 0.0166 | 0.980 | 1.469 |

Entropy did not collapse and KL/clip fraction do not support a target-KL
stability intervention. Explained variance became high, so PopArt/value-scale
work is not supported by L0. Action-family rates are interpreted with their
opportunity denominators; for example, absent RECALL opportunities are not a
policy rejection claim.

## L0.3: core and heuristic validation

All three core runs used the same manifest, Rust build, budget, validation days
and instrumentation. The best-checkpoint summary is:

| method | seed | best transition | mean total_cost_core | unfinished share | abandoned share | completed share | mean wait (min) | P95 wait (min) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| core | 11 | 159,744 | 13,199.1325 | 0.0000 | 0.0000 | 1.0000 | 5.557 | 11.325 |
| core | 22 | 98,304 | 13,113.9725 | 0.0000 | 0.0000 | 1.0000 | 5.523 | 11.103 |
| core | 33 | 233,472 | 13,213.4563 | 0.0000 | 0.0000 | 1.0000 | 5.581 | 11.376 |
| fixed | n/a | frozen | 16,016.8738 | 0.0000 | 0.0000 | 1.0000 | 7.409 | 14.899 |
| proportional | n/a | frozen | 16,195.8963 | 0.0037 | 0.0000 | 0.9963 | 7.055 | 14.742 |
| threshold | n/a | frozen | 16,641.4425 | 0.0034 | 0.0000 | 0.9966 | 7.204 | 15.434 |

Heuristic search budget was zero in this round: fixed, threshold and
proportional implementations were evaluated with their declared defaults and
then frozen. Their scalar Rust evaluation used `eval_batch_size=1` because the
batched path is PPO-only; physics, tapes and metrics were unchanged.

Learning curves and the reproducible summary are:

- `tables/core_learning_curves.csv`
- `tables/baseline_validation_summary.csv`
- `tables/baseline_summary.json`
- `plots/core_learning_curves.png`
- `selected_failure_traces.json` in each core run (top three validation-cost days,
  120 decision rows per trace)

The curves improve substantially from early checkpoints but remain noisy: the
best checkpoint occurs at 98k, 160k and 233k transitions for seeds 22, 11 and
33 respectively. This is evidence against declaring a fixed-B monotone
convergence result, not evidence to inspect held-out data.

## L0.4: decision and refutation criteria

**Registered next step: one GAE/credit-assignment L1 round.** The delayed effect
of dispatch, cooldown and terminal settlement makes GAE lambda a distinguishable
hypothesis, while entropy and update diagnostics do not justify changing
exploration or adding KL early stopping. The first round will compare exactly
`gae_lambda={0.95,0.98,1.0}`, with gamma 1, rollout/minibatch settings, reward,
physics, masks, runtime, budget and validation schedule unchanged. The control
is the frozen core. A candidate is refuted for this round if it fails
correctness, materially regresses registered service limits, or fails the
pre-registered multi-seed 2%/paired-CI rule.

Pre-registered service limits for the candidate-vs-control contrast are:
mean unfinished-share increase at most 0.005, mean abandoned-share increase at
most 0.002, mean P95 wait increase at most 1.0 minute, and mean worst-route wait
increase at most 1.0 minute. These limits were fixed from the L0 service
contract before screening; no held-out result informed them.

Unselected directions: target-KL early stopping (no instability trigger),
entropy coefficient/schedule (no collapse trigger), value normalization/PopArt
(good critic diagnostics), extended-budget study (not the registered first
round), forecast, encoder/memory, reward shaping, warm-start and all L2 tasks.
They are deferred/skipped, not implemented as prerequisites.
