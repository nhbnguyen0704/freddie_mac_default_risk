"""Airflow DAG fre_pd_monthly: input check, regional aggregates, features, scores and monitoring for one month."""
from datetime import datetime, timedelta
from pathlib import Path
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

HERE = Path(__file__).resolve().parent
PYTHON = HERE.parent / 'venv' / 'bin' / 'python'
JOBS = ['check_inputs', 'regional', 'fe', 'predict', 'monitor']


def command(job: str) -> str:
    return (f"cd {HERE} && PYTHONPATH=. MLFLOW_DISABLE_AGENT_HINT=1 {PYTHON} -m jobs.{job}_main "
            "--asof {{ logical_date.strftime('%Y-%m') }}")


with DAG(dag_id='fre_pd_monthly', schedule='@monthly', start_date=datetime(2022, 10, 1), end_date=datetime(2026, 3, 1),
         catchup=True, max_active_runs=1,
         default_args={'retries': 2, 'retry_delay': timedelta(minutes=5)}) as dag:
    tasks = [BashOperator(task_id=job, bash_command=command(job)) for job in JOBS]
    for upstream, downstream in zip(tasks, tasks[1:]):
        upstream >> downstream
