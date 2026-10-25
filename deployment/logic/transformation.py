import json
from pathlib import Path
from typing import Dict, List
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from pyspark.sql import DataFrame, functions as F
from scipy.special import expit
from logic.aggregation import estimated_property_value


def interactions(df: DataFrame) -> DataFrame:
    """The interaction features the models read, as 07_feature_engineering_main.add_interactions, dropping the slopes."""
    upb, orig_upb = F.col('current_actual_upb'), F.col('original_upb')
    orig_value = F.when((orig_upb > 0) & (F.col('original_loan_to_value') > 0),
                        orig_upb / F.col('original_loan_to_value') * F.lit(100))
    cols = {'ltv_x_dti': F.col('estimated_loan_to_value') * F.col('original_debt_to_income_ratio') / F.lit(100),
            'upb_pct_of_original': F.when(orig_upb > 0, upb / orig_upb * F.lit(100)),
            'estimated_property_value': estimated_property_value(),
            'property_value_change': estimated_property_value() / orig_value - F.lit(1),
            'upb_pct_of_original_roc_l12m': F.when(orig_upb > 0, F.col('_upb_slope_l12m') / orig_upb * F.lit(100)),
            'property_value_roc_l3m': F.col('_value_slope_l3m') / orig_value * F.lit(100),
            'property_value_roc_l12m': F.col('_value_slope_l12m') / orig_value * F.lit(100)}
    return (df.select('*', *[e.cast('double').alias(n) for n, e in cols.items()])
            .drop('_upb_slope_l12m', '_value_slope_l3m', '_value_slope_l12m'))

def clean(pdf: pd.DataFrame) -> pd.Series:
    """The clean segment: no 30+ DPD in the last 3 months and no 60+ DPD in the last 12 (missing = never)."""
    return pdf['dpd_cnt30_l3m'].fillna(0).eq(0) & ~pdf['months_since_last_dpd60'].le(11)

def to_frame(pdf: pd.DataFrame, feats: List[str], cats: Dict[str, List[str]]) -> pd.DataFrame:
    """XGBoost input: numerics as float32, strings as categoricals on the train category list."""
    return pd.DataFrame({c: pd.Categorical(pdf[c], categories=cats[c]) if c in cats
                         else pdf[c].to_numpy(dtype=np.float32, na_value=np.nan) for c in feats}, index=pdf.index)

def score(pdf: pd.DataFrame, boosters: Dict[str, xgb.Booster], bundle: dict) -> pd.DataFrame:
    """Segment, margin of the segment's model, Platt probability and flag per row."""
    out = pd.DataFrame({'segment': np.where(clean(pdf), 'clean', 'delinquent')}, index=pdf.index)
    out['margin'] = np.nan
    out['pd'] = np.nan
    for seg, s in bundle['segments'].items():
        rows = out['segment'] == seg
        if not rows.any():
            continue
        d = xgb.DMatrix(to_frame(pdf[rows], s['features'], bundle['categories']), enable_categorical=True, nthread=1)
        m = boosters[seg].predict(d, output_margin=True, iteration_range=(0, s['best_iteration'] + 1))
        out.loc[rows, 'margin'] = m
        out.loc[rows, 'pd'] = expit(s['platt'][0] + s['platt'][1] * m)
    out['flag'] = out['pd'] >= bundle['threshold']
    return out


class TwoSegmentModel(mlflow.pyfunc.PythonModel):
    """The two-segment merge as one MLflow model; predict takes the feature frame and returns score()."""

    def load_context(self, context) -> None:
        self.bundle = json.loads(Path(context.artifacts['bundle']).read_text())
        self.boosters = {s: xgb.Booster(model_file=context.artifacts[s]) for s in self.bundle['segments']}

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        return score(model_input, self.boosters, self.bundle)
