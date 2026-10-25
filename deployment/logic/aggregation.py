import math
from typing import Dict
import numpy as np
from pyspark.sql import Column, DataFrame, Window, functions as F

RECENCY_MONTHS = 12
UPB_ROUNDED_MAX_AGE = 6
CURTAIL_MULTIPLE = 1.5
LEVELS = {'zip3': 'postal_code', 'msa': 'metropolitan_statistical_area_or_metropolitan_division',
          'state': 'property_state'}


def estimated_property_value() -> Column:
    return F.when((F.col('current_actual_upb') > 0) & (F.col('estimated_loan_to_value') > 0),
                  F.col('current_actual_upb') / F.col('estimated_loan_to_value') * F.lit(100))

def loan_features(panel: DataFrame) -> DataFrame:
    """The loan window features the models read, as 07_feature_engineering_main, on every row of the panel."""
    ordered = Window.partitionBy('loan_identifier').orderBy('month_idx')
    idx, dpd = F.col('month_idx'), F.col('dpd')

    def frame(w: int, offset: int = 1):
        return ordered.rangeBetween(-(w - offset), 0)

    def prev(col: str) -> Column:
        return F.max(col).over(ordered.rangeBetween(-1, -1))

    def count(cond: Column, w: int, offset: int) -> Column:
        return F.sum(F.when(F.coalesce(cond, F.lit(False)), 1).otherwise(0)).over(frame(w, offset)).cast('tinyint')

    def mom_change_avg(col: str, w: int) -> Column:
        return F.avg('_mom_' + col).over(frame(w, 2))

    df = (panel.withColumn('month_idx', F.year('period') * 12 + F.month('period'))
          .withColumn('_value', estimated_property_value()))
    df = df.select('*', *[prev(c).alias(f'_prev_{c}') for c in
                          ('dpd', 'current_actual_upb', 'current_interest_rate', 'remaining_months_to_legal_maturity',
                           'loan_age', '_value')],
                   F.lag('month_idx').over(ordered).alias('_prev_row_idx'))

    seasoned = F.col('_prev_loan_age') > UPB_ROUNDED_MAX_AGE
    df = (df.withColumn('_mom_upb', F.when(seasoned, F.col('current_actual_upb') - F.col('_prev_current_actual_upb')))
          .withColumn('_mom_value', F.when(seasoned, F.col('_value') - F.col('_prev__value'))))

    gap_before = F.col('_prev_row_idx').isNull() | (idx - F.col('_prev_row_idx') > 1)
    df = df.withColumn('_break_current', F.when(~F.coalesce(dpd == 0, F.lit(False)), idx).when(gap_before, idx - 1))

    bal = F.col('_prev_current_actual_upb')
    r = F.col('_prev_current_interest_rate') / F.lit(1200)
    n = F.col('_prev_remaining_months_to_legal_maturity')
    payment = F.when(n > 0, F.when(r > 0, bal * r / (F.lit(1) - F.pow(F.lit(1) + r, -n))).otherwise(bal / n))
    scheduled = payment - bal * r
    judged = (seasoned & (F.col('_prev_dpd') == 0) & (dpd == 0) & (bal > 0) & (F.col('current_actual_upb') > 0)
              & (n > 0))
    df = df.withColumn('_curtail', F.when(judged, F.when(scheduled > 0, (bal - F.col('current_actual_upb'))
                                                         > scheduled * CURTAIL_MULTIPLE)))

    last60 = F.max(F.when(dpd >= 2, idx)).over(frame(RECENCY_MONTHS))
    cure = (F.col('_prev_dpd') >= 1) & (dpd == 0)
    return df.select(
        'loan_identifier', 'period', 'dpd',
        count(dpd >= 1, 3, 1).alias('dpd_cnt30_l3m'),
        F.coalesce(idx - last60, F.lit(RECENCY_MONTHS)).cast('tinyint').alias('months_since_last_dpd60'),
        *[(idx - F.coalesce(F.max('_break_current').over(frame(w)), idx - w)).cast('tinyint')
          .alias(f'current_streak_l{w}m') for w in (3, 6, 12)],
        *[count(cure, w, 2).alias(f'n_cure_l{w}m') for w in (3, 6, 12)],
        count(dpd > F.col('_prev_dpd'), 6, 2).alias('n_roll_forward_l6m'),
        *[count(F.col('_curtail'), w, 2).alias(f'n_curtail_l{w}m') for w in (3, 12)],
        F.when(F.size(F.collect_set('servicer_name').over(frame(RECENCY_MONTHS))) > 1, 1).otherwise(0)
        .cast('tinyint').alias('servicer_changed_l12m'),
        mom_change_avg('upb', 12).alias('_upb_slope_l12m'),
        *[mom_change_avg('value', w).alias(f'_value_slope_l{w}m') for w in (3, 12)])

def regional(active: DataFrame) -> Dict[str, DataFrame]:
    """Per level, (region, period) -> the five regional features, as 20_regional_features_main.build_regional."""
    out = {}
    for lv, col in LEVELS.items():
        g = (active.where(F.col(col).isNotNull()).groupBy(F.col(col).alias('region'), 'period')
             .agg(F.count(F.lit(1)).alias('loans'),
                  F.avg((F.col('dpd') >= 1).cast('double')).alias('dpd30_rate'),
                  F.avg((F.col('dpd') >= 3).cast('double')).alias('dpd90_rate'),
                  F.avg((F.col('dpd') == 1).cast('double')).alias('new_dpd30')))
        w = Window.partitionBy('region').orderBy(F.year('period') * 12 + F.month('period'))
        out[lv] = (g.withColumn('new_dpd30_3m', F.avg('new_dpd30').over(w.rangeBetween(-2, 0)))
                   .withColumn('dpd30_chg_6m', F.col('dpd30_rate') - F.first('dpd30_rate').over(w.rangeBetween(-6, -6)))
                   .select(F.col('region').alias(col), 'period',
                           *[F.col(f).alias(f'{lv}_{f}') for f in ('loans', 'dpd30_rate', 'dpd90_rate',
                                                                   'new_dpd30_3m', 'dpd30_chg_6m')]))
    return out

def bucket(col: str, spec: dict) -> Column:
    """The PSI bin of a value: -1 missing; for numerics the number of edges below it; for strings the
    1-based position in the baseline categories, 0 for any other."""
    c = F.col(col)
    if 'edges' in spec:
        return F.when(c.isNull(), -1).otherwise(sum((c > F.lit(e)).cast('int') for e in spec['edges']))
    return (F.when(c.isNull(), -1).when(c.isin(spec['categories']),
                                        F.array_position(F.array(*[F.lit(v) for v in spec['categories']]), c))
            .otherwise(0))

def shares(df: DataFrame, specs: Dict[str, dict]) -> Dict[str, Dict[str, float]]:
    """Per column, the share of rows in each PSI bin, in one pass."""
    pairs = F.explode(F.array(*[F.struct(F.lit(c).alias('c'), bucket(c, s).cast('int').alias('b'))
                                for c, s in specs.items()]))
    counts = df.select(pairs.alias('p')).groupBy('p.c', 'p.b').count().collect()
    out: Dict[str, Dict[str, float]] = {c: {} for c in specs}
    for r in counts:
        out[r['c']][str(r['b'])] = r['count']
    return {c: {b: n / sum(v.values()) for b, n in v.items()} for c, v in out.items()}

def psi(expected: Dict[str, float], actual: Dict[str, float], eps: float = 1e-6) -> float:
    """Population stability index over the union of bins."""
    total = 0.0
    for b in set(expected) | set(actual):
        e, a = expected.get(b, 0.0) + eps, actual.get(b, 0.0) + eps
        total += (a - e) * math.log(a / e)
    return total

def average_precision(score: np.ndarray, bad: np.ndarray, n: np.ndarray) -> float:
    """Average precision from score groups, as development's helpers.gbm.average_precision."""
    neg, inv = np.unique(-np.asarray(score, dtype=np.float64), return_inverse=True)
    tp = np.cumsum(np.bincount(inv, weights=bad, minlength=len(neg)))
    predicted = np.cumsum(np.bincount(inv, weights=n, minlength=len(neg)))
    return float(np.sum(np.diff(np.r_[0.0, tp / tp[-1]]) * tp / predicted))

def outcome_metrics(scored: DataFrame, prefix: str) -> Dict[str, float]:
    """AP of pd, precision and recall at the flag, flag share and, per segment, bad rate and mean pd,
    from scores with a 0/1 'bad' column."""
    g = (scored.groupBy(F.round('pd', 7).alias('p'))
         .agg(F.sum('bad').alias('bad'), F.count(F.lit(1)).alias('n')).toPandas())
    flagged = scored.where('flag').agg(F.sum('bad').alias('tp'), F.count(F.lit(1)).alias('n')).first()
    bads, rows = float(g['bad'].sum()), float(g['n'].sum())
    out = {f'{prefix}_ap': average_precision(g['p'].to_numpy(), g['bad'].to_numpy(float), g['n'].to_numpy(float)),
           f'{prefix}_precision': (flagged['tp'] or 0) / flagged['n'], f'{prefix}_recall': (flagged['tp'] or 0) / bads,
           f'{prefix}_flag_share': flagged['n'] / rows, f'{prefix}_bad_rate': bads / rows}
    for r in scored.groupBy('segment').agg(F.avg('bad').alias('b'), F.avg('pd').alias('p')).collect():
        out[f"{prefix}_bad_rate_{r['segment']}"] = r['b']
        out[f"{prefix}_mean_pd_{r['segment']}"] = r['p']
    return out
