# RL Improvement Implementation Plan

Date: 2026-09-12. Status: planned; no experiment below is accepted yet.

## 1. Scope and prerequisites

Execute the [RL improvement specification](../spec/improve_RL.md) only after the [Rust implementation plan](rust_improve.md) accepts R4. The [Rust specification](../spec/rust_improve.md) is authoritative for backend parity and speed gates. The [base specification](../spec/spec_v1.0.md) and [original plan](plan_v1.0.md) define the original study and its ablations.

Sequence: **accepted Rust R4 -> L0 baseline -> L1 PPO tuning -> optional L2 extensions -> L3 final evaluation/report**.

R4 requires equivalent behavior, at least **2x simulation and isolated learn speedup** under unprofiled measurement, and a faster measured full workflow. **4x is a stretch goal**. A fast but incorrect backend does not unlock RL experiments.

Task completion requires evidence, not an improved score. A negative result may satisfy research acceptance; missing runs, leakage, or incomparable experiments do not. Record task ID, source/build revision, commands, configs, seeds, hashes, artifact paths, status, and limitations in `reports/rl-improvement.md`. All proposed artifacts below are future outputs.

## 2. Dependencies and deliverables

| Task | Depends on | Deliverable |
|---|---|---|
| L0.1 | Accepted Rust R4 | Frozen experiment protocol and inventory |
| L0.2 | L0.1 | Verified training/evaluation diagnostics |
| L0.3 | L0.2 | Three-seed core baseline and heuristics |
| L0.4 | L0.3 | Baseline diagnosis and next-step decision |
| L1.1 | Accepted L0 | Bounded candidate design |
| L1.2 | L1.1 | Single-seed screening results |
| L1.3 | L1.2 | Multi-seed confirmation and selection |
| L2.1 | Accepted L1, selection decision | Optional causal forecast comparison |
| L2.2 | Accepted L1, selection decision | Optional architecture comparison |
| L2.3 | Accepted L1, selection decision | Optional objective comparison |
| L2.4 | Accepted L1, selection decision | Optional curriculum/warm-start comparison |
| L3.1 | L1 and selected L2 accepted or skipped | Frozen final matrix and ablations |
| L3.2 | L3.1 | Paired held-out evaluation |
| L3.3 | L3.2 | Statistics, report, and reproducibility package |

Optional L2 tasks are independent research choices, not a requirement to implement every idea. Mark unselected tasks SKIPPED with a reason; do not mark them experimentally validated.

## 3. L0: establish the baseline

### L0.1 - Freeze protocol and audit available artifacts

- [ ] Verify the Rust R4 acceptance report and freeze backend/build, Python revision, dependencies, manifests, and schema/physics/reward hashes.
- [ ] Inventory existing checkpoints and results; classify each as reusable, historical-only, or incomplete with a reason.
- [ ] Freeze core budget **B=245,760 transitions**, seeds **11,22,33**, validation every **12,288 transitions**, and **100 validation days**.
- [ ] Preserve the current core PPO config: 4 envs, n_steps=256, batch_size=256, n_epochs=4, LR=3e-4, ent_coef=0.01, gamma=1.0, gae_lambda=0.95, clip_range=0.2, and current MLP/physical/reward settings.
- [ ] Specify checkpoint selection by lowest validation cost, earliest checkpoint on ties; define fixed-transition and fixed-wall-clock measurement protocols separately.
- [ ] Register held-out splits and explicitly exclude them from candidate selection.

**Outputs:** proposed `reports/rl-improvement/protocol.json`, artifact inventory, and experiment ledger schema.

**Verification:** validate all referenced files/hashes; compare any reusable R4 seed against every L0 setting, including validation schedule and model metadata.

**Acceptance:** the protocol is complete before experiments run; each reused artifact has an equivalence justification; no result is inferred solely from an existing plan or pilot report.

### L0.2 - Add and verify diagnostics

- [ ] Persist validation cost by transitions and elapsed wall time, best/last checkpoint metrics, and completed episode counts.
- [ ] Capture PPO entropy, approximate KL, clip fraction, value loss, explained variance, and finite-value failures.
- [ ] Capture raw cost components, completion/abandonment/unfinished rates, per-route waits, and action frequencies paired with valid action opportunities.
- [ ] Add observation/reward/return distribution summaries and reproducible failure-trace selection criteria.
- [ ] Keep instrumentation settings consistent across compared candidates and record their overhead.

**Implementation surface:** `training/train.py`, `training/callbacks.py`, `evaluation/runner.py`, experiment logging; proposed `tests/test_training_diagnostics.py` where behavior is new.

**Verification:** short smoke confirms records join correctly by run/seed/timestep; fixture checks for invalid-value detection, absent action opportunities, and elapsed-time ordering; save/load retains metadata.

**Acceptance:** required diagnostics are populated or explicitly marked unavailable with a reason; no metric claims a policy ignores an action when its mask never permits it; diagnostics do not alter environment semantics.

### L0.3 - Run core and heuristic validation

- [ ] Train core for B transitions for all three seeds on the accepted Rust backend; retain best and last checkpoints and every run status.
- [ ] Validate fixed, threshold, and proportional controllers on identical tapes and metrics.
- [ ] Tune heuristic parameters only on validation, with a recorded search budget, then freeze them before held-out testing.
- [ ] Produce per-day raw metrics and learning curves using all 100 validation days.

**Outputs:** `runs/rl-improvement/core/<seed>/`, validation CSVs, proposed `reports/rl-improvement/baseline.md`.

**Verification:** inspect actual transitions, model/backend hashes, day IDs, checkpoint selection, and controller inputs; reject duplicates, missing days, and privileged future inputs.

**Acceptance:** three complete core runs and all heuristic validation results are reproducible; comparisons use the same metric and days. PPO is not required to beat heuristics.

### L0.4 - Diagnose failures and choose hypotheses

- [ ] Determine whether validation curves are improving, plateauing, or regressing.
- [ ] Attribute high cost to components and inspect service trade-offs, especially unfinished passengers and terminal cost.
- [ ] Relate action opportunities, choices, and traces to exploration and credit-assignment hypotheses.
- [ ] Choose a small next-step experiment based on recorded evidence rather than launching a full hyperparameter grid.

**Verification:** Trace each hypothesis to baseline data and confirm that all cited days belong to train or validation; review whether the proposed test can distinguish competing explanations.

**Acceptance:** baseline report links each proposed hypothesis to actual plots/metrics/traces and states what result would refute it. No held-out test result influences this decision.

**L0 gate:** L0.1-L0.4 accepted; a reproducible three-seed baseline exists.

## 4. L1: tune PPO while preserving the task

### L1.1 - Register a bounded candidate set

- [ ] Choose one parameter group per round from the specification: learning rate/update strength, entropy, rollout/minibatch size, or evidence-driven value scaling.
- [ ] Register at most **six candidates per round**, with core control, hypothesis, exact config diff, B, validation protocol, and rejection criteria.
- [ ] Keep gamma=1.0, physical dynamics, observation/action semantics, reward objective, and backend revision unchanged.
- [ ] If curves still improve, register a separate budget experiment at 2B and then 4B; do not label it a fixed-B improvement.
- [ ] For normalization, specify training-only fitting, frozen evaluation statistics, checkpoint persistence, and raw core-cost reporting before running.

**Outputs:** candidate configs and proposed `reports/rl-improvement/experiments.csv` entries created before execution.

**Verification:** compare config hashes/diffs; ensure minibatch size divides rollout size; check budget and validation schedules across candidates.

**Acceptance:** candidates are bounded and interpretable; no Cartesian sweep or unrecorded objective change; extended-budget runs are a separately labeled comparison.

### L1.2 - Screen candidates on seed 11

- [ ] Run the registered candidates on seed 11 with B transitions and identical validation conditions.
- [ ] Record all failed/divergent runs, consumed compute, and causes; never silently replace an unfavorable seed.
- [ ] Select at most **two candidates** for confirmation using the registered validation criteria.

**Verification:** reconcile the ledger with run directories, actual timesteps, best/last checkpoints, and diagnostics; investigate nonfinite loss or environment failures.

**Acceptance:** every registered trial has a terminal status and evidence; selected candidates have a documented rationale. Single-seed screening is not reported as a final improvement claim.

### L1.3 - Confirm across seeds and freeze the selection

- [ ] Run selected candidates on seeds 22 and 33, combining with seed 11 only when the protocol matches exactly.
- [ ] Compare against all three core seeds at the same B; report validation cost, service metrics, seed variability, transitions, and wall time.
- [ ] Retain core if candidates are not reliably better; document inconclusive or negative outcomes.
- [ ] For an extended-budget study, compare against the same core training trajectory at matching budgets and report additional compute separately.

**Verification:** Audit the three-seed result matrix and config hashes; recompute validation summaries from raw per-day metrics and check budget parity.

**Acceptance:** every selected candidate has three-seed evidence; the choice uses validation only; all configs and auxiliary artifacts are frozen. A valid decision to keep core passes this task.

**L1 gate:** L1.1-L1.3 accepted, including a documented no-change decision when appropriate.

## 5. L2: optional extensions with separate hypotheses

For each selected L2 task, register a control and candidate before running, use seeds 11/22/33 and matched transition/validation budgets, preserve raw metrics, and record added compute. Confirm a change independently before combining it with another extension. Unselected tasks receive an explicit SKIPPED decision.

### L2.1 - Causal forecast contribution

- [ ] Compare the existing forecast configuration against a matching no-forecast learning configuration.
- [ ] Fit the forecaster using train arrival logs only, freeze it, and save its artifact/hash with the policy.
- [ ] Verify past-equivalent tapes produce identical predictions regardless of future demand or scenario identity.
- [ ] Report forecast error alongside policy cost, unfinished share, and service outcomes.

**Verification:** existing forecast causality tests plus backend integration tests; matched three-seed validation comparison.

**Acceptance:** no future information leaks; control and candidate differ only in forecast availability; report does not treat improved MAE alone as policy improvement.

### L2.2 - Structured encoder or temporal memory

- [ ] Use baseline diagnostics to choose a structured route/stop/vehicle encoder or a separate temporal-memory hypothesis.
- [ ] Keep the original observation/action schema for the first structured-encoder experiment where possible.
- [ ] Complete a compatibility spike for MaskablePPO, masks, checkpoint load, and episode-state reset before any recurrent full run.
- [ ] Record architecture, parameter count, inference latency, training wall, and any changed schema/version.

**Verification:** shape/finite checks, gradient-flow tests for new modules, mask handling, checkpoint round trip, and recurrent-state isolation/reset tests when applicable; then matched three-seed validation.

**Acceptance:** the model integrates correctly and has comparable evidence. A schema change is versioned and never presented as a checkpoint-compatible backend-only change.

### L2.3 - Reward or credit-assignment experiment

- [ ] Select one hypothesis tied to a cost component, terminal penalty, fairness trade-off, discounting, or explicitly defined shaping.
- [ ] Version the changed learning objective/config; preserve physical dynamics and guards.
- [ ] Retain raw cost components and recompute `total_cost_core` using original weights for the primary comparison.
- [ ] Report completion, abandonment, wait/P95, route fairness, and operating cost alongside the scalar score.

**Verification:** reward-component tests and a fixture showing both objectives evaluated on identical components; matched three-seed validation.

**Acceptance:** the effect is separated from metric redefinition; no unsupported claim of policy invariance; any service regression is visible.

### L2.4 - Curriculum or heuristic warm-start

- [ ] Register the training distribution schedule or demonstration source and its train-only provenance.
- [ ] Include the target training distribution in the final curriculum stage; freeze the schedule before evaluation.
- [ ] Compare with PPO from scratch and account for demonstration generation, pretraining, transitions, and wall time.

**Verification:** split-provenance checks and checkpoint transfer smoke; matched multi-seed validation under the target distribution.

**Acceptance:** no held-out data informs demonstrations/scheduling, and auxiliary compute is disclosed. Extra data or compute is not described as a free same-budget improvement.

**L2 gate:** all selected tasks are accepted with evidence; remaining tasks are explicitly skipped. No positive result is required.

## 6. L3: final comparison and acceptance

### L3.1 - Freeze the final experiment matrix

- [ ] Select final candidates and controls using validation only; freeze checkpoint selection rules, configs, backend revision, and analysis protocol.
- [ ] Register practical service-regression limits for unfinished, abandoned, and worst-route metrics, and the minimum meaningful cost improvement before held-out evaluation.
- [ ] Complete the original T9 matrix: core, no_reassign, no_short, fairness_zero, each at seeds 11/22/33 and the same learning configuration/budget.
- [ ] If reporting tuned-policy ablations, register and run a separate coherent matrix; do not combine tuned core with older incompatible ablation arms.
- [ ] Register the fixed-wall-clock comparison budget and time-accounting rules; report it separately from fixed transitions.

**Verification:** matrix audit for missing seeds/arms, config diffs, budget parity, unchanged manifests, and test-blind selection history.

**Acceptance:** every planned contrast has a valid control and frozen protocol. Required missing runs remain incomplete, even if a partial report is produced.

### L3.2 - Run paired held-out evaluation

- [ ] Evaluate frozen policies and baselines on **200 test_id + 200 burst + 200 traffic days** using identical day IDs/tapes per comparison.
- [ ] Export raw per-day, per-seed components, core cost, all-demand wait/P95 and censoring, completed/abandoned/unfinished shares, route quality, headways, operating/deadhead time, and denied passengers.
- [ ] Use deterministic evaluation and preserve all failed cases with explanations.
- [ ] Run the registered fixed-wall-clock experiment under identical accounting; report actual transitions and evaluation cost separately.

**Verification:** check expected row counts, uniqueness, missing values, scenario hashes, checkpoint hashes, and original core-weight recomputation for reward ablations.

**Acceptance:** the complete matrix is paired and reproducible; test outcomes do not feed back into candidate selection. Further tuning after looking at test results is labeled exploratory and requires new confirmatory data.

### L3.3 - Statistics, report, and reproduction

- [ ] Report per-seed mean/std and paired day-level differences, with `delta = candidate cost - control cost`.
- [ ] Average model seeds per day before paired bootstrap: **2,000 resamples, seed 6001, 95% CI**. Do not treat 3xN rows as independent days.
- [ ] Test statistics on a synthetic constant-difference fixture; state that day-bootstrap uncertainty does not replace reporting training-seed variability.
- [ ] Plot validation cost against transitions and wall time; include final ID/OOD comparisons, service trade-offs, and reproducibly selected failure traces.
- [ ] Link every table to raw data/config/checkpoint hashes and every claimed speedup to the Rust benchmark report.
- [ ] Provide exact reproduction commands and inspect rendered report plots for labels, units, missing/censored data, and misleading axes.

**Verification:** Run the constant-difference statistics check, regenerate tables from raw results, inspect plots, and execute the documented reproduction checks.

**Acceptance:** a cost-improvement claim on a split requires the paired CI to lie below zero; practical/service thresholds must also be respected for an overall adoption claim. If the CI crosses zero, report insufficient evidence. ID-only gains do not establish OOD robustness. Complete negative results pass research acceptance.

## 7. Artifact layout and completion rules

```text
reports/rl-improvement.md                  # final status, evidence, conclusions
reports/rl-improvement/protocol.json       # frozen protocol and thresholds
reports/rl-improvement/baseline.md          # L0 diagnosis
reports/rl-improvement/experiments.csv      # all candidates and statuses
reports/rl-improvement/                    # paired tables, statistics, plots
runs/rl-improvement/<experiment>/<seed>/   # raw logs, configs, checkpoints
```

Keep pilot, Python diagnosis, and earlier study artifacts unchanged. Use stable run IDs and retain failed-trial evidence. Proposed filenames may be refined during implementation, but the report must index every output and preserve the protocol/evidence contract.

- [ ] Rust R4 acceptance and frozen revision verified.
- [ ] L0 baseline and diagnostics accepted.
- [ ] L1 bounded search and multi-seed decision accepted.
- [ ] Selected L2 experiments accepted; others explicitly skipped.
- [ ] L3 full paired matrix, statistics, service metrics, and report accepted.
- [ ] Throughput, sample efficiency, and wall-clock efficiency are reported separately.
- [ ] No unfinished task or experiment is marked complete because the result appears favorable.
