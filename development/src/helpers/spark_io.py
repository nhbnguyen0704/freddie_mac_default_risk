import logging
import os
from pathlib import Path
from typing import Dict, Optional
from helpers.config import settings
from pyspark.sql import DataFrame, SparkSession

log = logging.getLogger(__name__)

def curated_path(grain: str) -> str:
    """The HDFS URI of a curated table."""
    return f'{settings.hdfs_curated_url}/{grain}'

def resolve_path(path: str) -> str:
    """path as is when it has a scheme or does not exist locally, else as a file:// URI."""
    if "://" in path:
        return path
    local = Path(path).expanduser()
    if local.exists():
        log.info("%s exists locally; reading it as file://", path)
        return local.resolve().as_uri()
    return path

def spark_session(app_name: str, conf: Optional[Dict[str, str]] = None) -> SparkSession:
    """The local Spark session the jobs share. SPARK_MASTER defaults to local[*,4]: local mode ignores
    spark.task.maxFailures and gives each task one attempt, so a transient failure (macOS 'No buffer space
    available' on a Python worker socket) would kill the job; four attempts are safe because reads are
    idempotent and writes commit only on success. Shuffle partitions default to 400: at 1.4B loan-months the
    earlier 40 put ~36M rows in a partition and spilled enough to fill the disk."""
    builder = SparkSession.builder
    for k, v in (conf or {}).items():
        builder = builder.config(k, v)
    spark = (builder
             .master(os.getenv('SPARK_MASTER', 'local[*,4]'))
             .appName(app_name)
             .config('spark.hadoop.fs.defaultFS', settings.hdfs_uri)
             .config('spark.dynamicAllocation.enabled', 'true')
             .config('spark.dynamicAllocation.maxExecutors', '20')
             .config('spark.dynamicAllocation.minExecutors', '0')
             .config('spark.sql.shuffle.partitions',
                     os.getenv('SPARK_SHUFFLE_PARTITIONS', '400'))
             .config('spark.driver.memory', '10g')
             .config('spark.executor.memory', '10g')
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    return spark

def read_curated(spark: SparkSession, path: str) -> DataFrame:
    resolved = resolve_path(path)
    df = spark.read.parquet(resolved)
    return df
