# RL Improvement Implementation Plan

Updated: 2026-09-13. Status: L0 and the first bounded L1 round accepted; core retained. See `reports/rl-improvement.md`.

## 0. Implementer execution contract — read before working

**This is an experiment menu, not a feature backlog. Do not implement every listed direction.** Follow spec §0 for registration, screening, confirmation and KEEP/REJECT rules. Start with L0 only. A task checkbox inside L1/L2 applies only after that experiment is selected; unselected tasks are SKIPPED/DEFERRED with a reason, not missing features to fill in.

Required execution order:

1. Complete L0 diagnostics and write a next-step decision. If no supported hypothesis exists, keep core and skip extensions.
2. Register one parameter group/hypothesis and an experiment card before implementation. At most one initial L1 round, six configurations including control, and two shortlisted candidates.
3. Implement only the selected candidate's minimal isolated path, default off; run correctness and smoke tests. No unrelated architecture/reward changes.
4. Screen on seed 11, then confirm shortlisted candidates on 22/33. Apply pre-registered effect/service rules; record KEEP, REJECT or INCONCLUSIVE. Never adopt from seed 11 alone.
5. After L1, optionally select at most two L2 directions, **one at a time**, initially one candidate plus its control per direction. Do not start a later direction until the earlier decision is recorded. The first-round limit bounds search breadth, not the total number of improvements. Proactively continue evidence-supported rounds or combinations with a new written hypothesis/budget; no user re-approval is needed for routine in-scope work. Track the best accepted custom configuration as incumbent, with the original core and appropriate ablations retained.
6. KEEP retains every compatible improvement for an evolving custom algorithm, not just a single winner. If A and B independently pass, proactively implement and evaluate A+B against control/A/B under a registered budget. Promote the combination to incumbent if it passes quality/service rules; for practical ties, assess complementary benefits, compute and complexity. If interaction hurts, retain the successful standalone variants and investigate. No separate user approval is required for in-scope combination trials; global defaults and the historical core remain unchanged.
7. Freeze selected configurations and perform L3, including the original T9 matrix. A complete study may have no winning algorithm changes and zero implemented L2 extensions.

Experiment card required fields: ID, diagnostic evidence, hypothesis/refutation, control ID, exact config diff (one group), implementation surface, correctness checks, B/seeds/runtime, extra data/compute, minimum cost improvement, explicit service limits, selection rule, stopping rule and artifact paths.

Default confirmation rule from spec §0: ≥2% mean core-cost reduction (unless a different threshold is justified before trials), validation paired day-bootstrap 95% CI below zero, lower mean cost on at least 2/3 seeds, and all registered service limits satisfied. This is validation selection, not final test evidence. Do not invent service thresholds after seeing scores. Nonfinite/leakage/mask/checkpoint failures are INVALID, not quality results; fix and rerun before proceeding.

Each task handoff must state: selected experiment, code/config changed, checks and raw metrics, consumed compute, decision with reason, and **the single next task or STOP**. Explicitly list remaining directions as unselected; do not tell the next implementer to finish every checkbox.

## 1. Scope and prerequisites

Execute the [RL improvement specification](../spec/improve_RL.md) on the accepted Rust runtime. R4 and memory M1/M3 are complete; O1/O2 and the configurable distribution-validation fast path have evidence in [runtime-optimization.md](../../reports/runtime-optimization.md). L0 must verify and freeze that evidence, not reopen migration or wait for deferred optimizations. The [Rust specification](../spec/rust_improve.md) is authoritative for backend parity and speed gates. The [base specification](../spec/spec_v1.0.md) and [original plan](plan_v1.0.md) define the original study and its ablations.

Sequence: **verify accepted R4 + freeze research runtime -> L0 baseline -> L1 PPO tuning -> optional L2 extensions -> L3 final evaluation/report**.

R4 requires equivalent behavior, at least **2x simulation and isolated learn speedup** under unprofiled measurement, and a faster measured full workflow. **4x is a stretch goal**. A fast but incorrect backend does not unlock RL experiments.

Task completion requires evidence, not an improved score. A negative result may satisfy research acceptance; missing runs, leakage, or incomparable experiments do not. Record task ID, source/build revision, commands, configs, seeds, hashes, artifact paths, status, and limitations in `reports/rl-improvement.md`. All proposed artifacts below are future outputs.

### 1.1 Current code, test and report conventions

Follow [repository layout](../../README.md), [test conventions](../../tests/README.md), [reports convention](../../reports/README.md) and [spec §9.1](../spec/improve_RL.md#91-convention-code-và-tests-khi-triển-khai-l0l3). Rust is the default maintained backend. Python oracle/wrapper are deprecated, require explicit `--backend python --legacy-python`, and receive no fixes. Archived R0–R4 parity is historical evidence; do not restore the retired golden/parity suite or require current Python/Rust numerical lockstep.

Place code by ownership: `src/bus_rl/learning/` for policy/forecast/training, `execution/` for runtime/environment/scenarios, `evaluation/` for metrics/statistics/plots. New candidate modules are conditional on selection. Keep core physics in `crates/bus-sim-core/` and bridge changes in `crates/bus-sim-python/`; do not implement L2 objectives by changing deprecated Python simulator code.

Unit tests use `tests/unit/<src path>/test_<module>.py`, one module plus stubs/data/utilities. Integration directories mirror the entry point; filenames use `test_<subject>[_<aspect>].py`, with optional aspect from `contract`, `payload`, `lifecycle`, `mapping`, `determinism`, `flow`. Cross-subsystem checks belong in integration. Shared builders go in `tests/support/`; `conftest.py` is for fixtures/collection only. Tests hold code, not committed data; regenerate fixtures or reference indexed report evidence.

For implementation changes run relevant checks, then the required gates: `uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest -q` (4 workers, legacy skipped; `-n 0` for serial debugging). Native/FFI changes also require a current extension build, `cargo test -p bus-sim-core`, and `uv run pytest -m native -q`. Write new build provenance with `uv run python scripts/build_native.py --output runs/rl-improvement/<experiment>/<seed>/<run-id>/native-build.json`; preserve archived build records. Mark extension-dependent tests `native`; missing-extension skips are not native acceptance. Historical `--legacy-python` checks are not required study gates. Record commands, exit codes and pass/skip counts separately from multi-seed quality evidence.

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
| L2.5 | Accepted L1, critic-scale evidence | Optional PopArt critic comparison |
| L3.1 | L1 and selected L2 accepted or skipped | Frozen final matrix and ablations |
| L3.2 | L3.1 | Paired held-out evaluation |
| L3.3 | L3.2 | Statistics, report, and reproducibility package |

Optional L2 tasks are independent research choices, not a requirement to implement every idea. Mark unselected tasks SKIPPED with a reason; do not mark them experimentally validated.

## 3. L0: establish the baseline

### L0.1 - Freeze protocol and audit available artifacts

- [x] Verify existing R4/memory/runtime reports and the actual loaded binary hash; freeze backend/build, Python revision, dependencies, manifests and schema/physics/reward hashes. Do not rerun migration just to reopen its completed gates.
- [x] Create `configs/experiments/rl-improvement/core.toml`, preserving core learning settings and setting CPU Torch threads=2, Rust backend, eval_batch_size=16, reuse_eval_pool=true, native_batch=true, validate_distributions=false. Preserve historical configs and global defaults.
- [x] Record requested/effective runtime flags, thread settings and instrumentation. Register scalar Rust evaluation for unsupported heuristic batch combinations; keep physics/tapes/metrics identical and report compute separately. Forecast/new-policy paths require compatibility checks, never silent fallback.
- [x] Inventory existing checkpoints and results; classify each as reusable, historical-only, or incomplete with a reason.
- [x] Freeze core budget **B=245,760 transitions**, seeds **11,22,33**, validation every **12,288 transitions**, and **100 validation days**.
- [x] Preserve the current core PPO config: 4 envs, n_steps=256, batch_size=256, n_epochs=4, LR=3e-4, ent_coef=0.01, gamma=1.0, gae_lambda=0.95, clip_range=0.2, and current MLP/physical/reward settings.
- [x] Specify checkpoint selection by lowest validation cost, earliest checkpoint on ties; define fixed-transition and fixed-wall-clock measurement protocols separately.
- [x] Register held-out splits and explicitly exclude them from candidate selection.

**Outputs:** proposed `reports/rl-improvement/protocol.json`, artifact inventory, and experiment ledger schema.

**Verification:** validate all referenced files/hashes; compare any reusable R4 seed against every L0 setting, including validation schedule, model metadata and required L0 diagnostics. Repeated seed-11 runtime runs do not replace seeds 22/33. Missing training telemetry cannot be reconstructed from final weights; rerun if necessary. Re-evaluation may supply missing per-day metrics but not historical training curves. Do not reuse pre-instrumentation timing as an equivalent wall-clock control.

**Acceptance:** the protocol is complete before experiments run; each reused artifact has an equivalence justification; no result is inferred solely from an existing plan or pilot report.

### L0.2 - Add and verify diagnostics

- [x] Persist validation cost by transitions and elapsed wall time, best/last checkpoint metrics, and completed episode counts.
- [x] Capture PPO entropy, approximate KL, clip fraction, value loss, explained variance, and finite-value failures.
- [x] Log valid-action count K, entropy H and H/log(K) only for K>1; mark K=1 separately (H=0), reject K=0. For each action family record valid-slot count, opportunity count (at least one valid slot), probability mass, and conditional selection rate. This is telemetry, not a normalized-entropy loss change.
- [x] Register finite checks for logits, values, returns, losses and gradients despite distribution validation being off; record check cadence/overhead and fail explicitly. Preserve RNG/model behavior and use identical instrumentation across arms.
- [x] Capture raw cost components, completion/abandonment/unfinished rates, per-route waits, and action frequencies paired with valid action opportunities.
- [x] Add observation/reward/return distribution summaries and reproducible failure-trace selection criteria.
- [x] Keep instrumentation settings consistent across compared candidates and record their overhead.

**Implementation surface (repo-relative):** `src/bus_rl/learning/training/{train,callbacks,checkpoint}.py`, `src/bus_rl/evaluation/{runner,summary}.py`, experiment logging. Isolated tests mirror each module under `tests/unit/bus_rl/`; if a new `learning/training/diagnostics.py` is justified, its proposed unit test is `tests/unit/bus_rl/learning/training/test_diagnostics.py`. Cross-module telemetry/checkpoint checks extend `tests/integration/bus_rl/learning/training/test_training_flow.py`. See [spec §9](../spec/improve_RL.md#9-nguồn-và-điểm-bắt-đầu-triển-khai) for the current code map after refactoring.

**Verification:** short smoke confirms records join correctly by run/seed/timestep; fixture checks for invalid-value detection, absent action opportunities, K=1, masked zero probabilities, nonfinite logits with a valid mask, and elapsed-time ordering; save/load retains metadata.

**Acceptance:** required diagnostics are populated or explicitly marked unavailable with a reason; no metric claims a policy ignores an action when its mask never permits it; diagnostics do not alter environment semantics.

### L0.3 - Run core and heuristic validation

- [x] Train core for B transitions for all three seeds on the accepted Rust backend; retain best and last checkpoints and every run status.
- [x] Validate fixed, threshold, and proportional controllers on identical tapes and metrics.
- [x] Tune heuristic parameters only on validation, with a recorded search budget, then freeze them before held-out testing (this round's search budget was zero; defaults were frozen).
- [x] Produce per-day raw metrics and learning curves using all 100 validation days.

**Outputs:** raw validation CSVs, metadata and checkpoints in `runs/rl-improvement/core/<seed>/<run-id>/`; curated tables/plots under `reports/rl-improvement/`, with analysis in proposed `reports/rl-improvement/baseline.md`.

**Verification:** inspect actual transitions, model/backend hashes, day IDs, checkpoint selection, and controller inputs; reject duplicates, missing days, and privileged future inputs.

**Acceptance:** three complete core runs and all heuristic validation results are reproducible; comparisons use the same metric and days. PPO is not required to beat heuristics.

### L0.4 - Diagnose failures and choose hypotheses

- [x] Determine whether validation curves are improving, plateauing, or regressing.
- [x] Attribute high cost to components and inspect service trade-offs, especially unfinished passengers and terminal cost.
- [x] Relate action opportunities, choices, and traces to exploration and credit-assignment hypotheses.
- [x] Write one next-step decision with a completed experiment card and spec §0 acceptance thresholds, or STOP/KEEP CORE with a reason. Mark all other directions unselected; do not implement them as prerequisites.

**Verification:** Trace each hypothesis to baseline data and confirm that all cited days belong to train or validation; review whether the proposed test can distinguish competing explanations.

**Acceptance:** baseline report links each proposed hypothesis to actual plots/metrics/traces and states what result would refute it. No held-out test result influences this decision.

**L0 gate:** L0.1-L0.4 accepted; a reproducible three-seed baseline exists (`reports/rl-improvement/baseline.md`).

## 4. L1: tune PPO while preserving the task

### L1.1 - Register a bounded candidate set

- [x] Choose one parameter group per round from the specification: the first round selected GAE lambda based on L0 diagnostics; other groups remain unselected.
- [x] Register gae_lambda={0.95,0.98,1.0} with gamma=1, fixed n_steps and other settings; compare advantage variance, critic diagnostics and cost. Lambda is not described as a hard planning horizon.
- [x] Leave target-KL early stopping unselected because L0 did not show KL/update instability; no target-KL candidate was registered.
- [x] Register at most **six configurations per round including the core control**, with hypothesis, exact config diff, B, validation protocol, and rejection criteria (`experiments.csv`).
- [ ] Keep gamma=1.0, physical dynamics, observation/action semantics, reward objective, and backend revision unchanged.
- [ ] If curves still improve, register a separate budget experiment at 2B and then 4B; do not label it a fixed-B improvement.
- [ ] For normalization, specify training-only fitting, frozen evaluation statistics, checkpoint persistence, and raw core-cost reporting before running.

**Outputs:** candidate configs and proposed `reports/rl-improvement/experiments.csv` entries created before execution.

**Verification:** test nullable target_kl round-trip/wiring and expected update-stop logging if selected; algorithm candidates need correctness and multi-seed quality evidence, not identical weights to core. Runtime-only changes require equivalence against the frozen native control, not the retired Python parity suite. Compare config hashes/diffs; ensure minibatch size divides rollout size; check budget and validation schedules across candidates.

**Acceptance:** candidates are bounded and interpretable; no Cartesian sweep or unrecorded objective change; extended-budget runs are a separately labeled comparison.

### L1.2 - Screen candidates on seed 11

- [x] Run the registered candidates on seed 11 with B transitions and identical validation conditions.
- [x] Record all failed/divergent runs, consumed compute, and causes; never silently replace an unfavorable seed.
- [x] Select at most **two candidates** for confirmation using the registered validation criteria (none shortlisted; both were rejected at screening).

**Verification:** reconcile the ledger with run directories, actual timesteps, best/last checkpoints, and diagnostics; investigate nonfinite loss or environment failures.

**Acceptance:** every registered trial has a terminal status and evidence; selected candidates have a documented rationale. Single-seed screening is not reported as a final improvement claim.

### L1.3 - Confirm across seeds and freeze the selection

- [x] Run selected candidates on seeds 22 and 33, combining with seed 11 only when the protocol matches exactly (no candidates were selected, so this was correctly skipped).
- [x] Compare against all three core seeds at the same B; report validation cost, service metrics, seed variability, transitions, and wall time.
- [x] Apply spec §0 confirmation criteria and assign KEEP/REJECT/INCONCLUSIVE to each shortlisted candidate. Keep the control when no candidate passes; no candidate passed screening, so no combination or default change was made.
- [x] For an extended-budget study, compare against the same core training trajectory at matching budgets and report additional compute separately (deferred; not part of this registered first round).

**Verification:** Audit the three-seed result matrix and config hashes; recompute validation summaries from raw per-day metrics and check budget parity.

**Acceptance:** every selected candidate has three-seed evidence; the choice uses validation only; all configs and auxiliary artifacts are frozen. A valid decision to keep core passes this task.

**L1 gate:** L1.1-L1.3 accepted, including the documented no-change decision in `reports/rl-improvement.md`.

## 5. L2: optional extensions with separate hypotheses

Use the research shortlist and primary sources in spec §6.5–6.6. Select at most two L2 directions in the first round; forecasts may occupy one slot. For each selected L2 task, register a control and candidate before running, use seeds 11/22/33 and matched transition/validation budgets, preserve raw metrics, and record added compute. Confirm a change independently before combining it with another extension. Unselected tasks receive an explicit SKIPPED decision.

**Round decision (2026-09-13).** Two directions were selected, one at a time,
and both were rejected on three-seed confirmation; the first-round limit is now
used.

- **L2.1 causal forecast** (`configs/experiments/rl-improvement/forecast.toml`,
  single-group `[forecast] enabled=false -> true`; card `L2-FORECAST`): mean cost
  +2.03% worse than the L0 core control, paired 100-day bootstrap 95% CI
  `[240.38, 295.72]`, lower on 1/3 seeds, seed-11 P95/worst-route wait deltas
  over +1.0 min. **REJECT**.
- **L2.3 potential-based reward shaping** (`configs/experiments/rl-improvement/pbrs.toml`,
  single-group `[shaping] enabled=false -> true`, `queue_weight=2.0`,
  `excess_weight=5.0`; card `L2-PBRS`): mean cost +3.59% worse, paired CI
  `[447.66, 498.57]`, lower on 0/3 seeds, seed-11/33 P95 and seed-33 worst-route
  wait deltas over +1.0 min. **REJECT**.

Core is retained. **L2.2 encoder/memory, L2.4 curriculum/warm-start and L2.5
PopArt are SKIPPED** — L0 diagnostics do not trigger them (flat MLP has no
measured sample-efficiency failure, PPO already beats heuristics, explained
variance is high) and the first-round limit of two L2 directions is used.
Evidence: `reports/rl-improvement.md`,
`reports/rl-improvement/tables/l2_forecast_*.{csv,json}` and
`reports/rl-improvement/tables/l2_pbrs_*.{csv,json}`.

### L2.1 - Causal forecast contribution (SELECTED; REJECT)

- [x] Compare the existing forecast configuration against a matching no-forecast learning configuration.
- [x] Fit the forecaster using train arrival logs only, freeze it, and save its artifact/hash with the policy.
- [x] Verify past-equivalent tapes produce identical predictions regardless of future demand or scenario identity.
- [x] Report forecast error alongside policy cost, unfinished share, and service outcomes.

Outcome: **REJECT** on three-seed confirmation; core control retained. See the
round decision above and `reports/rl-improvement.md`.

**Verification:** `tests/unit/bus_rl/learning/test_forecasting.py` for isolated causality plus native environment/training integration under the entry-point path; matched three-seed validation comparison.

**Acceptance:** no future information leaks; control and candidate differ only in forecast availability; report does not treat improved MAE alone as policy improvement.

### L2.2 - Structured encoder or temporal memory

- [ ] Use baseline diagnostics to choose a structured route/stop/vehicle encoder or a separate temporal-memory hypothesis.
- [ ] Keep the original observation/action schema for the first structured-encoder experiment where possible. Use per-stop/per-vehicle shared encoders and validity-masked pooling/fusion; retain route/direction/position signals. Compare against a similar-parameter-budget MLP and report capacity differences.
- [ ] For SET-PPO, follow Deep Sets-inspired entity sharing without assuming topology permutation symmetry. Verify padding invariance and preserve entity-to-action identity; test permutations only with valid joint ID/action remapping. Register a similar-parameter MLP control and keep 221 slots.
- [ ] Complete a compatibility spike for MaskablePPO, masks, checkpoint load, and episode-state reset before any recurrent full run.
- [ ] Record architecture, parameter count, inference latency, training wall, and any changed schema/version.

**Verification:** shape/finite checks, gradient-flow tests for new modules, mask handling, checkpoint round trip, and recurrent-state isolation/reset tests when applicable; then matched three-seed validation.

**Acceptance:** the model integrates correctly and has comparable evidence. A schema change is versioned and never presented as a checkpoint-compatible backend-only change.

### L2.3 - Reward or credit-assignment experiment (SELECTED; REJECT)

- [x] Select one hypothesis tied to a cost component, terminal penalty, fairness trade-off, discounting, or explicitly defined shaping.
- [x] If shaping is selected, register a fixed causal potential with r_shaped=r+gamma*Phi_next-Phi_now, zero potential at all true terminal states, explicit truncation/bootstrap handling and no hidden/future inputs. Add telescoping fixtures across action traces; keep original terminal settlement and raw rewards. Do not promise equivalent learned PPO weights or improved learning.
- [x] Version the changed learning objective/config; preserve physical dynamics and guards.
- [x] Retain raw cost components and recompute `total_cost_core` using original weights for the primary comparison.
- [x] Report completion, abandonment, wait/P95, route fairness, and operating cost alongside the scalar score.

Outcome: **REJECT** on three-seed confirmation (mean cost +3.59% vs control,
paired CI entirely above zero, 0/3 seeds lower, P95/worst-route limits violated);
core control retained. See the round decision above and
`reports/rl-improvement.md`.

**Verification:** `tests/unit/bus_rl/learning/test_shaping.py` (telescoping, terminal-zero, causality, finite) and native `tests/integration/bus_rl/learning/training/test_shaping_flow.py` (reward accounting on the real kernel), plus matched three-seed validation.

**Acceptance:** the effect is separated from metric redefinition; no unsupported claim of policy invariance; any service regression is visible.

### L2.4 - Curriculum or heuristic warm-start

- [ ] Register the training distribution schedule or demonstration source and its train-only provenance.
- [ ] For BC-PPO, freeze a validation-selected threshold/proportional teacher and collect train-only (obs, mask, action) episodes. Initial proposal: D=12,288 demo transitions, at most 5 masked-NLL epochs, train-day grouped split for pretraining selection. Do not pretrain critic from action labels; report NOOP/family imbalance and teacher quality.
- [ ] Compare BC(D)+PPO(B) with scratch(B), scratch(B+D), and the teacher; record environment transitions, teacher labels, BC updates and total compute separately. A DAgger follow-up requires its own budget and train-only learner-state relabeling protocol; it is not part of the initial BC run.
- [ ] Include the target training distribution in the final curriculum stage; freeze the schedule before evaluation.
- [ ] Compare with PPO from scratch and account for demonstration generation, pretraining, transitions, and wall time.

**Verification:** split-provenance checks and checkpoint transfer smoke; matched multi-seed validation under the target distribution.

**Acceptance:** no held-out data informs demonstrations/scheduling, and auxiliary compute is disclosed. Extra data or compute is not described as a free same-budget improvement.

### L2.5 - PopArt value-target normalization (conditional)

- [ ] Require return-scale and critic-error evidence from L0/L1; otherwise mark SKIPPED. Use spec §6.5 and the original PopArt paper; no claim of a built-in supported switch.
- [ ] Register matched PPO control, B/seeds, train-only stats, update boundary, epsilon/std floor and optimizer-state treatment. Do not simultaneously change reward normalization, vf_coef or encoder.
- [ ] Implement a custom critic/head-loss compatibility spike that preserves unnormalized outputs when stats change. Keep GAE/bootstrap/stored values in consistent raw units and normalize targets only at the declared value-loss boundary.
- [ ] Persist stats with checkpoints and freeze during eval. Test constant targets, output preservation, GAE units, gradients and save/load before full runs.
- [ ] Compare raw-unit value error/explained variance, original core cost, service metrics and compute across 3 seeds; retain negative outcomes.

**Verification:** numeric fixtures for stats rescaling and terminal/bootstrap handling, compatibility tests with the pinned MaskablePPO, then matched multi-seed experiments.

**Acceptance:** no metric/return unit substitution, no eval stats updates, complete cost/service/compute evidence. Equal model weights to the baseline are not required for this algorithm change.

**L2 gate:** all selected tasks have a completed decision and evidence, including valid REJECT/INCONCLUSIVE outcomes; remaining tasks are explicitly skipped/deferred. KEEP requires spec §0 confirmation; task completion does not imply adoption. No positive result is required.

## 6. L3: final comparison and acceptance

### L3.1 - Freeze the final experiment matrix

- [x] Select final candidates and controls using validation only; freeze checkpoint selection rules, configs, backend revision, and analysis protocol.
- [x] Register practical service-regression limits for unfinished, abandoned, and worst-route metrics, and the minimum meaningful cost improvement before held-out evaluation.
- [x] Complete the original T9 matrix: core, no_reassign, no_short, fairness_zero, each at seeds 11/22/33 and the same learning configuration/budget.
- [x] If reporting tuned-policy ablations, register and run a separate coherent matrix; do not combine tuned core with older incompatible ablation arms. (No tuned policy was adopted, so the frozen core matrix is used.)
- [x] Register the fixed-wall-clock comparison budget and time-accounting rules; report it separately from fixed transitions.

Frozen in `reports/rl-improvement/l3_protocol.json`; config audit in
`reports/rl-improvement/tables/l3_config_audit.json`. The L3 audit found that the
native kernel ignored `run.reward`, making the `fairness_zero` arm invalid; the
reward passthrough was fixed, the extension rebuilt, and all 12 arms re-run on one
frozen build (`d6bf1551…`).

**Verification:** matrix audit for missing seeds/arms, config diffs, budget parity, unchanged manifests, and test-blind selection history.

**Acceptance:** every planned contrast has a valid control and frozen protocol. Required missing runs remain incomplete, even if a partial report is produced.

### L3.2 - Run paired held-out evaluation

- [x] Evaluate frozen policies and baselines on **200 test_id + 200 burst + 200 traffic days** using identical day IDs/tapes per comparison.
- [x] Export raw per-day, per-seed components, core cost, all-demand wait/P95 and censoring, completed/abandoned/unfinished shares, route quality, headways, operating/deadhead time, and denied passengers.
- [x] Use deterministic evaluation and preserve all failed cases with explanations.
- [x] Run the registered fixed-wall-clock experiment under identical accounting; report actual transitions and evaluation cost separately. (Registered; because no adopted algorithm change alters per-step compute it reduces to the fixed-transition wall-clock view, reported in `reports/rl-improvement.md`.)

**Verification:** check expected row counts, uniqueness, missing values, scenario hashes, checkpoint hashes, and original core-weight recomputation for reward ablations.

**Acceptance:** the complete matrix is paired and reproducible; test outcomes do not feed back into candidate selection. Further tuning after looking at test results is labeled exploratory and requires new confirmatory data.

### L3.3 - Statistics, report, and reproduction

- [x] Report per-seed mean/std and paired day-level differences, with `delta = candidate cost - control cost`.
- [x] Average model seeds per day before paired bootstrap: **2,000 resamples, seed 6001, 95% CI**. Do not treat 3xN rows as independent days.
- [x] Test statistics on a synthetic constant-difference fixture; state that day-bootstrap uncertainty does not replace reporting training-seed variability.
- [x] Plot validation cost against transitions and wall time; include final ID/OOD comparisons, service trade-offs, and reproducibly selected failure traces.
- [x] Link every table to raw data/config/checkpoint hashes. Cite Rust, memory and runtime reports for infrastructure speedups; cite matched-runtime experiments for algorithm speed/quality claims. Do not multiply historical cross-session speedups into a measured end-to-end result.
- [x] Provide exact reproduction commands and inspect rendered report plots for labels, units, missing/censored data, and misleading axes.

Reproduce with `uv run python scripts/l3_analysis.py`.

**Verification:** Run the constant-difference statistics check, regenerate tables from raw results, inspect plots, and execute the documented reproduction checks.

**Acceptance:** a cost-improvement claim on a split requires the paired CI to lie below zero; practical/service thresholds must also be respected for an overall adoption claim. If the CI crosses zero, report insufficient evidence. ID-only gains do not establish OOD robustness. Complete negative results pass research acceptance.

## 7. Artifact layout and completion rules

```text
reports/rl-improvement.md                  # study status, decisions, final analysis
reports/rl-improvement/README.md           # curated artifact index and reproduction
reports/rl-improvement/protocol.json       # frozen protocol and thresholds
reports/rl-improvement/baseline.md          # L0 diagnosis
reports/rl-improvement/experiments.csv      # candidate ledger, decisions and run links
reports/rl-improvement/tables/             # aggregated paired results and statistics
reports/rl-improvement/plots/              # selected rendered figures
reports/rl-improvement/evidence/           # only raw evidence justified for commit
runs/rl-improvement/<experiment>/<seed>/<run-id>/
                                          # raw per-invocation data; gitignored
```

Raw per-day/per-seed CSVs, telemetry, failure traces, profiler dumps, build records, metadata and checkpoints belong in the run directory. Register stable run IDs before execution and use a new ID for reruns. Curated reports link to their source run/config/hash and reproduction command. If a raw artifact must be committed as evidence, retain the selected copy under `reports/rl-improvement/evidence/`, explain why in the index and avoid duplicate copies. Do not store fixture data in `tests/` or expand the archived oracle manifest hash scope to include this study. Update `reports/README.md` when actual curated outputs are created, not to claim planned results exist.

Keep pilot, Python diagnosis, and earlier study artifacts unchanged. Use stable run IDs and retain failed-trial evidence. Proposed filenames may be refined during implementation, but the report must index every output and preserve the protocol/evidence contract.

- [x] Existing R4/memory/runtime evidence verified; research config and actual loaded binary frozen.
- [x] L0 baseline and diagnostics accepted.
- [x] L1 bounded search and multi-seed decision accepted (no candidate shortlisted; core retained).
- [x] Selected L2 experiments accepted; others explicitly skipped. (L2.1 forecast and L2.3 PBRS selected and REJECTED with evidence; L2.2, L2.4, L2.5 explicitly SKIPPED.)
- [x] L3 full paired matrix, statistics, service metrics, and report accepted. (Core retained; all T9 ablations worse; no CI below zero.)
- [x] Throughput, sample efficiency, and wall-clock efficiency are reported separately.
- [x] No unfinished task or experiment is marked complete because the result appears favorable.
