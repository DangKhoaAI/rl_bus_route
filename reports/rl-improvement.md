# RL improvement study status

Status: **L0 accepted; L1 first round accepted with KEEP CORE / no algorithm change.**

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

## Compute and limitations

The three core runs consumed approximately 327.6 s wall time in concurrent
launches; the two screening candidates consumed approximately 184.0 s. Exact
per-run `wall_time_s`, transitions, backend/build hashes and effective runtime
flags are in each run's `metadata.json`. The host was shared, so these walls are
reported as observed run measurements, not a cross-session speed claim.

The Python oracle, historical R4/runtime runs, and incomplete `runs/core-11`
checkpoint were not reused as L0 quality evidence. Held-out ID/OOD evaluation,
T9 ablations, L2 directions and fixed-wall-clock comparison remain unopened.
No global defaults were changed and no candidate was combined or adopted.

## Next task

**Single next task: STOP algorithm selection and freeze this L0/L1 decision for
L3 planning.** Before any L2 direction, a new written hypothesis and budget is
required. Unselected directions (target-KL, entropy, extended budget, forecast,
encoder/memory, reward shaping, warm-start, PopArt and all other L2 tasks) stay
SKIPPED/DEFERRED; they are not missing implementation work.
