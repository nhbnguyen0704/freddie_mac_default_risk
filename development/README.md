# Development

The research pipeline: from the Freddie Mac quarterly archives to the two-segment XGBoost model that `../deployment` serves.

Each step is a standalone job `src/jobs/NN_<name>_main.py`. Numbering is execution order. A job reads earlier outputs (HDFS tables, `artifacts/*.json`) and writes a Markdown report to `docs/<name>.md` plus a JSON artifact in `artifacts/`.

## Pipeline

| jobs | step |
|---|---|
| 01 | Ingestion: quarterly zips to `origination` and `performance` parquet on HDFS |
| 02 | Data quality report |
| 03_05, 04 | Vintage and roll-rate analysis, which set the bad definition (`data/curated/observations_labelspec.json`) |
| 06 | Loan-month observations with labels and population exclusions |
| 07 | Feature engineering to `training_set` (see `docs/feature_dictionary.md`) |
| 08–10 | Logistic regression: WoE binning, IV/correlation/VIF selection, model versions |
| 11–15 | XGBoost: split, correlation selection, SHAP elimination, resampling, Optuna tuning |
| 16–17 | Random forest: permutation elimination, tuning |
| 18–19 | Model score correlation and error analysis |
| 20–23 | Regional features, regional model on all rows, two-segment merge, retuning |
| 24 | F1-optimal cut on the merged probability, chosen on validation |

## Setup

- Python environment: `../venv`.
- `.env` sets the local data paths, the HDFS URI and curated directory, and the observation period. It is not versioned.
- HDFS must be running. The curated tables live at `hdfs://localhost:9000/user/binhn/fre/curated/<grain>`, partitioned by `year`/`month`.

## Running a job

Run from `development/`, with `src` on the path:

```bash
PYTHONPATH=src ../venv/bin/python -m jobs.15_gbm_tuning_main --stage tune
PYTHONPATH=src ../venv/bin/python -m jobs.01_ingestion_main verify
```

Most later jobs take `--stage` so long steps can be rerun piecemeal, and many write-by-month jobs take `--resume`.

## Conventions

- The train/valid/test split is never written out. It is recomputed from a salted hash with the config in `artifacts/gbm_split.json`. Test is every month after 2022-09; valid is 20% of loans in the earlier months.
- Keys are `(loan_identifier, period)`; the target is `label`.
- The primary metric is average precision.

## Outputs

- `docs/`: one report per job.
- `artifacts/`: metadata JSON, Optuna studies and model binaries. Only the JSON is versioned.
