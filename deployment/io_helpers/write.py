import json
from pathlib import Path
from pyspark.sql import DataFrame, functions as F


def write_month(df: DataFrame, path: str, files: int) -> None:
    """Write df as year/month partitions of path, replacing only the months it holds."""
    (df.withColumn('year', F.year('period')).withColumn('month', F.month('period'))
     .repartition(files).write.mode('overwrite').partitionBy('year', 'month').parquet(path))

def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
