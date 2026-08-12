# New scripts (branch `nika_dev`)

| Script | Purpose | Usage | Output |
|---|---|---|---|
| `scripts/dataset_stats_windfarm.py` | Raw-data stats: row counts, missing data, sensor value ranges | `python scripts/dataset_stats_windfarm.py` | `results/windfarm_dataset_stats.json` |
| `scripts/benchmark_windfarm.py` | Trains/evaluates the ridge regression baseline for one WindFarm sub-task | `python scripts/benchmark_windfarm.py <subtask_name> [data_frac] [alpha]` | `results/benchmark/ridge_windfarm_<subtask_name>_alpha<alpha>.json` |

`subtask_name` (default `odd_time_predict48h`): one of `odd_time_predict48h`,
`odd_space_predict48h`, `odd_spacetime_predict48h`, `odd_time_predict72h`,
`odd_space_predict72h`, `odd_spacetime_predict72h`.
`data_frac` (default `1`): fraction of data to use, for quick test runs.
`alpha` (default `1`): ridge regularization strength.

Also added `scikit-learn` to `requirements.txt`. Everything else in
`scripts/` existed before, only bugs were fixed there, not new scripts.

All 6 sub-tasks have been run (feature-engineered, `alpha=100`); see
`WindFarm_benchmark_report.md` for results.
