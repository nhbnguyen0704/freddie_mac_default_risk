import json
from pathlib import Path
from typing import Tuple
import mlflow
from pyspark.sql import Column, DataFrame, SparkSession, functions as F

ROOT = Path(__file__).resolve().parents[1]


def local_path(path: str) -> Path:
    """A path from config.json, relative to deployment/."""
    return (ROOT / path).resolve()

def read_json(path: str) -> dict:
    return json.loads(local_path(path).read_text())

def config() -> dict:
    return read_json('config.json')

def month_index(asof: str) -> int:
    """'YYYY-MM' -> year * 12 + month."""
    y, m = asof.split('-')
    return int(y) * 12 + int(m)

def partition_index() -> Column:
    return F.col('year') * 12 + F.col('month')

def months(spark: SparkSession, path: str, first: int, last: int) -> DataFrame:
    """The year/month partitions of a table from month index first to last."""
    return spark.read.parquet(path).where(partition_index().between(first, last))

def tracking(cfg: dict) -> None:
    """Point MLflow at the SQLite store and the experiment under deployment/mlflow."""
    d = local_path(cfg['mlflow']['dir'])
    d.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f'sqlite:///{d / "mlflow.db"}')
    if mlflow.get_experiment_by_name(cfg['mlflow']['experiment']) is None:
        mlflow.create_experiment(cfg['mlflow']['experiment'], artifact_location=(d / 'artifacts').as_uri())
    mlflow.set_experiment(cfg['mlflow']['experiment'])

def champion(cfg: dict) -> Tuple[object, str]:
    """The champion TwoSegmentModel (bundle and boosters loaded) and its registry version."""
    tracking(cfg)
    name, alias = cfg['mlflow']['model_name'], cfg['mlflow']['alias']
    version = mlflow.MlflowClient().get_model_version_by_alias(name, alias).version
    return mlflow.pyfunc.load_model(f'models:/{name}@{alias}').unwrap_python_model(), version
