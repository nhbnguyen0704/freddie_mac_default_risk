# Deployment

Monthly batch scoring of probability of default with the two-segment XGBoost model built in `../development`.

## Model

| segment | rows | model | features |
|---|---|---|---|
| clean | no 30+ DPD in the last 3 months and no 60+ DPD in the last 12 | `clean_regional` | 53 (38 loan features + 15 regional aggregates) |
| delinquent | every other row | `recency_12` | 38 |

Each segment's margin is mapped to a probability with a Platt fit on validation. A loan is flagged when PD ≥ 0.1642, the F1-optimal cut chosen on validation. The model is registered in MLflow as `fre_pd_two_segment`, and alias `champion` points to the version in use.

## Monthly run

For an as-of month t, the Airflow DAG `fre_pd_monthly` runs five jobs in order:

| task | does | writes |
|---|---|---|
| `check_inputs` | fails if any performance month t-11..t is missing | — |
| `regional` | delinquency aggregates by zip3, MSA and state from every active loan | `regional_monthly` |
| `fe` | eligible loans of t with their features | `scoring_features` |
| `predict` | segment, margin, PD, flag and decile per loan, via `mapInPandas` | `scores` |
| `monitor` | score and feature PSI, flag rates, 6-month proxy and 25-month backtest | MLflow run, `runs/<YYYY-MM>.json` |

The tables are written to `hdfs://localhost:9000/user/binhn/fre/deployment/<table>`, partitioned by `year`/`month`. Each job overwrites only its own month, so a rerun is safe.

Scored loans: performing in t (not 90+ DPD), no workout ever, not exiting in t, and no disaster flag in t-11..t.

## Layout

```
config.json   input and output paths, Spark and MLflow settings
dag.py        Airflow DAG
spark.py      Spark session
io_helpers/   reads, writes, partition queries
logic/        eligibility, features, scoring function and MLflow model, PSI and metrics
jobs/         one module per task, plus register_model_main
runs/         monitoring results per month
```

## Commands

Run from `deployment/`, with HDFS up.

```bash
# one job for one month
PYTHONPATH=. MLFLOW_DISABLE_AGENT_HINT=1 ../venv/bin/python -m jobs.fe_main --asof 2023-01

# register the model with its PSI baselines; moves champion to the new version
PYTHONPATH=. ../venv/bin/python -m jobs.register_model_main

# Airflow
export AIRFLOW_HOME=$PWD/airflow AIRFLOW__CORE__DAGS_FOLDER=$PWD AIRFLOW__CORE__LOAD_EXAMPLES=False
../airflow_venv/bin/airflow dags test fre_pd_monthly 2023-01-01
PATH=$PWD/../airflow_venv/bin:$PATH airflow standalone    # UI at http://localhost:8080

# MLflow UI at http://127.0.0.1:5001
MLFLOW_DISABLE_AGENT_HINT=1 ../venv/bin/python -m mlflow ui --backend-store-uri sqlite:///$PWD/mlflow/mlflow.db --port 5001
```

## Status

- Backfill 2022-10 to 2026-03 is complete (42 months).
- On the training-set rows, results match development: merged test AP 0.2217 (development: 0.2216); precision 0.376 and recall 0.221 at the cut.
- Known limitation: training excludes loans that later prepay good, while deployment scores them, so realised AP on all scored rows is about 0.10 and precision about 0.18.
