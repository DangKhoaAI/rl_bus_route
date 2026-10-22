# Report Tables

- `results.csv`: episode-level evaluations; original inputs are preserved in provenance/inputs/.
- `summary.csv`: daily mean metrics after averaging model seeds; counts and seed SD are reported separately.
- `comparisons.csv`: matched baseline minus RL costs, relative reductions and paired 95% intervals. Positive savings favor RL.
- `training.csv`: measured reference, lowest validation cost, selected step and training coverage per run.
- `training_history.csv`: normalized histories with real seed/encoder labels, when available.
- `benchmark.md`, `comparison.md`, `training.md`, `service.md`: presentation tables.

Distinct scenario hashes define paired days. All model seeds must cover identical days. Missing metric values are excluded per column. Seed SD is unavailable with fewer than two model seeds.

| Metric | Definition |
|---|---|
| `total_cost_core` | Core-weighted operational cost, passenger-minute equivalents; lower is better. |
| `mean_wait` | Mean observed passenger wait (minutes), including end-of-episode censored waits. |
| `p95_wait` | Mean daily P95 wait (minutes), rather than the pooled passenger-level percentile. |
| `worst_route_mean_wait` | Mean wait of the worst-performing route in each episode (minutes). |
| `completed_share` | Completed passengers / generated passengers, fraction [0, 1]. |
| `unfinished_share` | Passengers waiting or onboard at episode end / generated passengers. |
| `abandoned_share` | Abandoned passengers / generated passengers, fraction [0, 1]. |
| `unique_denied_share` | Passengers denied boarding at least once / generated passengers. |
