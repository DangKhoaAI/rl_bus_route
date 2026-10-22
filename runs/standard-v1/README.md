# Standard Run Evidence

Study: `standard-v1`. Seeds: 11, 22, 33. Evaluation conditions: test_id, test_ood_burst, test_ood_traffic.

| Configuration | Evidence report |
|---|---|
| `mlp10` | [mlp10/report/README.md](mlp10/report/README.md) |
| `gru10` | [gru10/report/README.md](gru10/report/README.md) |
| `belief20-fade` | [belief20-fade/report/README.md](belief20-fade/report/README.md) |

`summary.csv` and `comparisons.csv` consolidate every registered policy, including unsuccessful configurations. `protocol.json` freezes budgets, configs, scenario identities and implementation hashes.

Each policy/seed contains train metadata and test evaluations. Policy-level reports aggregate all seeds. Shared baseline evaluations are under `baselines/shared/eval/`. These artifacts support the human-authored final study in `reports/`.

Local checkpoint weights, validation episode telemetry and diagnostic logs are excluded from Git evidence. Regenerate tapes with `bus-rl data generate` using configs/base.toml and the split counts in configs/study.toml; compare the resulting manifest against data/manifest.json.
