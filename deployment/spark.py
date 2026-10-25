import os
import sys
from pyspark.sql import SparkSession


def spark_session(app: str, cfg: dict) -> SparkSession:
    """Local Spark on the project venv, reading and writing HDFS, overwriting only the partitions it writes."""
    os.environ.setdefault('PYSPARK_PYTHON', sys.executable)
    spark = (SparkSession.builder
             .master(cfg['master'])
             .appName(app)
             .config('spark.hadoop.fs.defaultFS', cfg['default_fs'])
             .config('spark.sql.shuffle.partitions', cfg['shuffle_partitions'])
             .config('spark.driver.memory', cfg['memory'])
             .config('spark.executor.memory', cfg['memory'])
             .config('spark.sql.sources.partitionOverwriteMode', 'dynamic')
             .getOrCreate())
    spark.sparkContext.setLogLevel('WARN')
    return spark
