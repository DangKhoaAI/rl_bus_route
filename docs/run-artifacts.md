# Run Artifact Contract

`experiments/` stores local working state. `runs/` stores Git-synchronized evidence.
`reports/` stores the human-authored final study. Standard runs do not overwrite
accepted historical tables or protocols.

## Layout

```text
experiments/                         # ignored
  archive/                           # retained historical evidence and selected models
  infrastructure/native-build.json   # current local build provenance
  standard-v1/
    <policy>/<seed>/train/            # best.zip, metadata, learning/validation CSVs
    <policy>/<seed>/eval/<split>/     # results and evaluation context
    <policy>/report/                 # technical summary across model seeds
    baselines/shared/eval/<split>/   # deterministic baseline results
runs/                                # tracked
  README.md
  standard-v1/
    artifacts.json                   # exported inventory, byte counts and checksums
    protocol.json                    # frozen matrix and implementation/config hashes
    configs/                         # study, base and policy snapshots
    data/manifest.json               # seeds, scenario hashes and tape paths; no tapes
    <policy>/<seed>/train/            # metadata.json and evaluations.csv
    <policy>/<seed>/eval/<split>/     # per-day results and portable training history
    <policy>/report/                 # English figures, tables, HTML and provenance
    baselines/shared/eval/<split>/
    summary.csv
    comparisons.csv
reports/                             # final synthesis written by humans
```

## Standard matrix

`configs/studies/standard.toml` registers all policies, seeds, data counts and
baseline controllers. Policy configs explicitly set equal 245,760-step budgets,
validation every 12,288 steps, two Torch threads and the same Rust execution path.
Model seeds are 11, 22 and 33. The dataset has 500 training days, 100 validation
days and 200 days each for normal demand, demand surge and traffic disruption.
The test dataset has been used in earlier studies; this matrix is a reproducibility
snapshot rather than a newly unseen confirmatory benchmark.

MLP10 is the feed-forward control, GRU10 is the current default window encoder,
and Belief20-fade is the fixed discounted demand-rate feature variant. Belief is
a local Gamma–Poisson rate approximation, not a full POMDP belief-state solver.
All registered configurations, including unsuccessful ones, are exported.

Step-zero validation is measured directly and is ineligible for checkpoint
selection. The best post-training validation checkpoint is retained, with the
earlier checkpoint winning a tie. All model training finishes before test
scores are opened. Shared baselines are evaluated on the same scenario tapes.

## Export policy

`artifact export` requires a completed experiment and a fresh destination outside
the local source tree. It copies JSON, TOML, CSV, Markdown, PNG, SVG and HTML
evidence. It omits weights, optimizer state, diagnostic JSONL/logs and raw
`validation_metrics.csv`. Files larger than 8 MiB require an explicit storage
choice rather than silently being added to Git. The exported inventory records
all file hashes and lists omitted local state.

`artifact verify` checks the exact inventory, byte sizes and SHA256 hashes.
Verification establishes artifact integrity, not policy quality. Metadata may
record local source/checkpoint paths for provenance; those paths do not imply
that model weights were uploaded. Data tapes are regenerated from the frozen
base config and counts and checked against the published scenario manifest.

Use the per-policy technical reports, root summary/contrast CSVs and provenance
to author the final narrative under `reports/`. Preserve unfavorable outcomes,
service trade-offs, model-seed variability and the limits of day-bootstrap CIs.

## Historical cleanup

Previous local `runs/` content is now under `experiments/archive/`. Retained state
includes the final L3 benchmark/ablation best checkpoints and raw evaluations,
three original GRU encoder checkpoints, and the native performance reference.
Raw diagnostics, redundant last checkpoints, speed-probe weights and superseded
report demos were removed. A deletion inventory and retained checkpoint hashes
are stored locally in `experiments/cleanup-20261022.json`.

Historical report records preserve the paths that existed when they were
produced. Regeneration scripts now read the corresponding moved archive paths.
New build/performance diagnostics write to `experiments/`, leaving final report
curation under human control.
