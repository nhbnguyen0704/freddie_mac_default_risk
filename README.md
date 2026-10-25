# Freddie Mac Default Risk

Probability-of-default modelling on the Freddie Mac Single-Family Loan-Level dataset, from the raw quarterly archives to a monthly batch scoring service.

A loan-month is **bad** when the loan reaches 90+ days past due, a terminal credit event or a workout (modification, deferral, forbearance or repayment plan) within the next 25 months. Loans that pay off inside the window are censored.

## Layout

| folder | what it holds |
|---|---|
| [`development/`](development/README.md) | Research pipeline: ingestion, label definition, features, and the logistic regression, XGBoost and random forest tracks, ending in the two-segment XGBoost model |
| [`deployment/`](deployment/README.md) | Monthly scoring of that model on Spark, orchestrated by Airflow, registered and monitored in MLflow |
| `venv/` | Python 3.12 environment shared by both (PySpark 4.2, XGBoost 3.4, scikit-learn, MLflow 3.17, pandas/pyarrow) |
| `airflow_venv/` | Separate environment for Airflow 3.3 |

## Results

Test = every month after 2022-09 (187.5M loan-months, 1.2% bad). Average precision (AP) is the primary metric.

| model | test AP |
|---|---|
| Logistic regression (WoE, recency weighted) | 0.1399 |
| Random forest (undersampled) | 0.1895 |
| XGBoost (recency weighted) | 0.2174 |
| **Two-segment XGBoost merge (deployed)** | **0.2216** |

At the F1-optimal cut chosen on validation, the deployed model flags 0.70% of test rows with precision 0.376 and recall 0.221.

Known limitation: training rows exclude loans that later prepay without going bad, while deployment scores every eligible loan. On all scored rows the realised AP is about 0.10 and precision about 0.18.

## Infrastructure

Everything runs locally: PySpark in local mode, and a local HDFS at `hdfs://localhost:9000` that holds the curated tables and the deployment outputs. HDFS must be running for nearly every job.

Not under version control (see `.gitignore`): the environments, `.env`, raw data, parquet tables, model binaries, the MLflow store and the Airflow home.
