# Case 1 Competition Pipeline

This implementation runs three unique approaches with an identical structure and CLI contract:

- `elastic_net`
- `tree_ensembles`
- `pls_pcr`

Each approach writes the same artifacts:

- `metrics_outer.csv`
- `tuning_trace.csv`
- `summary.json`
- `run_metadata.json`
- `chosen_config.json` (from refit)
- `predictions.csv` (from refit)

## Folder Layout

- `Case1/src/case1_comp/common` shared data/preprocessing/CV utilities
- `Case1/src/case1_comp/approaches/<approach>` per-approach package with same file names
- `Case1/src/case1_comp/orchestrator` all-approach runner and aggregation
- `Case1/hpc` LSF scripts

## Local Run (All Approaches + Final Selection)

From repository root:

```bash
python -m pip install -e .
export PYTHONPATH=Case1/src
python -m case1_comp.orchestrator.run_all --student-no 123456
```

Outputs:

- `Case1/results/runs/<approach>/...` intermediate approach artifacts
- `Case1/results/leaderboard.csv`
- `Case1/results/winner.json`
- `Case1/results/sample_predictions_<student-no>.csv`
- `Case1/results/sample_estimatedRMSE_<student-no>.csv`

## Local Run (Single Approach)

```bash
python -m case1_comp.approaches.elastic_net.run_nested_cv --out-dir Case1/results/runs/elastic_net/nested
python -m case1_comp.approaches.elastic_net.run_refit_predict --out-dir Case1/results/runs/elastic_net/refit
```

Swap `elastic_net` with `tree_ensembles` or `pls_pcr`.

## HPC Run (LSF, Python 3.12)

1. Copy repository to HPC and `cd` into repo root.
2. Set root path:

```bash
export COMPDATA_ROOT=/path/to/CompData
```

3. Optional: set student number and RMSE safety margin:

```bash
export STUDENT_NO=123456
export RMSE_MARGIN=0.2
```

4. Submit all jobs (one per approach + dependent aggregation):

```bash
bash Case1/hpc/submit_all.sh
```

Logs are written to `Case1/results/logs`.

## Notes

- Nested CV uses outer assessment and inner tuning.
- The 1-SE rule is used for robust hyperparameter selection.
- Final winner is selected by lowest mean outer RMSE, then lowest RMSE std as tie-break.
- RMSE self-estimate is computed as `mean_outer_rmse + RMSE_MARGIN * std_outer_rmse`.
