"""Shared by the XGBoost jobs (11_gbm_split_main onwards).

The train / valid / test split is never written out: every job derives it on
the fly from the same seeded hash, recorded once in gbm_split.json by
11_gbm_split_main. A copy of the 764M-row training set with a split column
would not fit next to the original on this disk.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from pyspark.sql import Column, DataFrame, functions as F
from pyspark.sql.types import NumericType, StringType
from helpers.config import settings
from helpers.spark_io import resolve_path

TARGET = 'label'
KEYS: tuple = ('loan_identifier', 'period')
NOT_FEATURES: tuple = KEYS + (TARGET, 'observation_date', 'year', 'month')
SPLIT = 'split'

DEV_END = '2022-09'
VALID_SHARE = 0.2
SPLIT_SALT = 'gbm_valid'
HASH_BITS = 31

SPLIT_JSON = settings.model_dir / 'gbm_split.json'


def ym(year: str = 'year', month: str = 'month') -> Column:
    return F.col(year) * 100 + F.col(month)

def uniform(*cols: Column, salt: str) -> Column:
    """A seeded U[0, 1) from the given columns."""
    return F.pmod(F.xxhash64(*cols, F.lit(salt)), F.lit(1 << HASH_BITS)) / F.lit(float(1 << HASH_BITS))

def row_id() -> Column:
    """A 64-bit id per (loan, month): how jobs 14 and 15 name a row to each other."""
    return F.xxhash64(F.col('loan_identifier'), F.col('period'))

def split_column(cfg: Dict[str, object]) -> Column:
    """'test' after the train window; inside it a loan is wholly 'valid' or
    wholly 'train', so no loan's months sit on both sides."""
    u = uniform(F.col('loan_identifier'), salt=str(cfg['salt']))
    return (F.when(ym() > F.lit(int(cfg['end_ym'])), F.lit('test'))
            .when(u < F.lit(float(cfg['valid_share'])), F.lit('valid'))
            .otherwise(F.lit('train')))

def dev_rows(spark, path: str, cfg: Dict[str, object]) -> DataFrame:
    """The train window (partition pruned: year/month are partition columns) with the split column."""
    df = spark.read.parquet(resolve_path(path)).where(ym() <= F.lit(int(cfg['end_ym'])))
    return with_split(df, cfg)

def sample_meta_path(sample: str) -> Path:
    """The local meta file of a sample kept on HDFS: data/curated/<name>_meta.json."""
    return settings.data_curated / f"{sample.rstrip('/').rsplit('/', 1)[-1]}_meta.json"

def load_split(path: Path = SPLIT_JSON) -> Dict[str, object]:
    """The split config written by 11_gbm_split_main."""
    if not path.exists():
        raise SystemExit(f'No split at {path}: run 11_gbm_split_main first.')
    return json.loads(path.read_text())

def with_split(df: DataFrame, cfg: Dict[str, object]) -> DataFrame:
    """df with the split column: 'test' after DEV_END (the logistic regression's test months), else 'valid' or
    'train' by loan."""
    return df.withColumn(SPLIT, split_column(cfg))

def feature_kinds(df: DataFrame, exclude: Optional[List[str]] = None) -> Dict[str, str]:
    """'numerical' for numeric columns, 'categorical' for strings, as 08_lr_woe_binning_main."""
    exclude = set(exclude or [])
    kinds: Dict[str, str] = {}
    for f in df.schema.fields:
        if f.name in NOT_FEATURES or f.name in exclude or f.name == SPLIT:
            continue
        if isinstance(f.dataType, NumericType):
            kinds[f.name] = 'numerical'
        elif isinstance(f.dataType, StringType):
            kinds[f.name] = 'categorical'
    return kinds

def average_precision(score: np.ndarray, bad: np.ndarray, n: np.ndarray) -> float:
    """Average precision from score groups, as
    sklearn.metrics.average_precision_score: rows with the same score enter
    the curve together, and AP = sum over thresholds of (R_k - R_k-1) * P_k.
    bad and n may be weighted sums, and two groups may share a score."""
    neg, inv = np.unique(-np.asarray(score, dtype=np.float64), return_inverse=True)
    tp = np.cumsum(np.bincount(inv, weights=bad, minlength=len(neg)))
    predicted = np.cumsum(np.bincount(inv, weights=n, minlength=len(neg)))
    if tp[-1] <= 0:
        return float('nan')
    precision = tp / predicted
    recall = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def to_frame(pdf, feats: List[str], kinds: Dict[str, str], cats: Dict[str, List[str]]):
    """XGBoost input: numeric as float32, strings as categoricals on the train
    category list (a fixed code per category; anything else is missing)."""
    import pandas as pd
    out = {}
    for c in feats:
        if kinds[c] == 'categorical':
            out[c] = pd.Categorical(pdf[c], categories=cats[c])
        else:
            out[c] = pdf[c].to_numpy(dtype=np.float32, na_value=np.nan)
    return pd.DataFrame(out, index=pdf.index)
