import re
from typing import List
from pyspark.sql import SparkSession
from io_helpers.read import months


def month_label(i: int) -> str:
    """Month index -> 'YYYY-MM'."""
    return f'{(i - 1) // 12}-{(i - 1) % 12 + 1:02d}'

def missing_months(spark: SparkSession, path: str, first: int, last: int) -> List[str]:
    """The months from index first to last with no partition under path."""
    files = ' '.join(months(spark, path, first, last).inputFiles())
    have = {int(y) * 12 + int(m) for y, m in re.findall(r'year=(\d+)/month=(\d+)/', files)}
    return [month_label(i) for i in range(first, last + 1) if i not in have]
