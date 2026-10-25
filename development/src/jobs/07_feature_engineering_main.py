import argparse
import json
import logging
from pathlib import Path
from typing import List, Sequence, Tuple
from pyspark.sql import DataFrame, Window, functions as F
from helpers.config import settings
from helpers.period import add_arguments as add_period_arguments, from_args as period_from_args, month_partitions
from helpers.spark_io import curated_path, resolve_path, spark_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s")
log = logging.getLogger('feature_engineering_main')

WINDOWS: tuple = (3, 6, 12)

DPD_MEASURES: tuple = ('max', 'cnt30', 'cnt60')
DPD_COUNT_THRESHOLDS = {'cnt30': 1, 'cnt60': 2}

UPB_SLOPE_PREFIX = '_upb_slope_l'

VALUE_SLOPE_PREFIX = '_value_slope_l'

NOT_FEATURES: tuple = ('first_payment_date', 'maturity_date', 'vintage')

RECENCY_MONTHS = 12

STREAK_STATES = {'current': lambda d: d == 0,
                 'dpd30': lambda d: d >= 1,
                 'dpd60': lambda d: d >= 2}

UPB_ROUNDED_MAX_AGE = 6

CURTAIL_MULTIPLE = 1.5

INTERACTIONS = {
    'rate_gap': ('current_interest_rate', 'original_interest_rate'),
    'ltv_x_dti': ('estimated_loan_to_value', 'original_debt_to_income_ratio'),
    'upb_pct_of_original': ('current_actual_upb', 'original_upb'),
    'ltv_ratio_to_original': ('estimated_loan_to_value', 'original_loan_to_value'),
    'estimated_property_value': ('current_actual_upb', 'estimated_loan_to_value'),
    'negative_equity_flag': ('estimated_loan_to_value',),
    'property_value_change': ('current_actual_upb', 'estimated_loan_to_value', 'original_upb',
                              'original_loan_to_value'),
}

def feature_names(windows: Sequence[int] = WINDOWS) -> List[str]:
    """The DPD window feature names for the given windows."""
    return [f'dpd_{m}_l{w}m' for w in windows for m in DPD_MEASURES]

def add_window_features(df: DataFrame, windows: Sequence[int] = WINDOWS, loan_col: str = 'loan_identifier',
                        order_col: str = 'period') -> Tuple[DataFrame, List[str]]:
    """Per window w: the worst dpd and the counts of 30+ and 60+ DPD months over the last w months, t included."""
    if 'dpd' not in df.columns:
        log.warning('No window features: dpd is not in the panel.')
        return df, []

    df = df.withColumn('month_idx', F.year(order_col)*12 + F.month(order_col))
    ordered = Window.partitionBy(loan_col).orderBy('month_idx')

    made: List[str] = []
    for w in windows:
        frame = ordered.rangeBetween(-(w-1), 0)
        cols = {f'dpd_max_l{w}m': F.max('dpd').over(frame).cast('float')}
        for m, level in DPD_COUNT_THRESHOLDS.items():
            cols[f'dpd_{m}_l{w}m'] = (F.sum(F.when(F.col('dpd') >= level, 1).otherwise(0))
                                      .over(frame).cast('tinyint'))
        for name, expr in cols.items():
            df = df.withColumn(name, expr)
        made.extend(cols)
    return df.drop('month_idx'), made


def estimated_property_value(upb: str = 'current_actual_upb', eltv: str = 'estimated_loan_to_value'):
    """The current property value implied by the balance and the estimated LTV."""
    return F.when((F.col(upb) > 0) & (F.col(eltv) > 0), F.col(upb) / F.col(eltv) * F.lit(100))

def original_property_value(upb: str = 'original_upb', ltv: str = 'original_loan_to_value'):
    """The property value at origination implied by the original balance and LTV."""
    return F.when((F.col(upb) > 0) & (F.col(ltv) > 0), F.col(upb) / F.col(ltv) * F.lit(100))

def _add_mom_change_avg(df: DataFrame, col: str, prefix: str, windows: Sequence[int], loan_col: str, order_col: str,
                        exact_balance: bool = False) -> Tuple[DataFrame, List[str]]:
    """Per window w, the average month-on-month change of col over the last w-1 changes. With exact_balance, a
    change is used only after the loan is UPB_ROUNDED_MAX_AGE months old, as Freddie rounds the balance before
    that."""
    df = df.withColumn('month_idx', F.year(order_col)*12 + F.month(order_col))
    ordered = Window.partitionBy(loan_col).orderBy('month_idx')
    prev = F.max(col).over(ordered.rangeBetween(-1, -1))
    change = F.col(col) - prev
    if exact_balance:
        prev_age = F.max('loan_age').over(ordered.rangeBetween(-1, -1))
        change = F.when(prev_age > UPB_ROUNDED_MAX_AGE, change)
    df = df.withColumn('_mom', change)
    made: List[str] = []
    for w in windows:
        name = f'{prefix}{w}m'
        df = df.withColumn(name, F.avg('_mom').over(ordered.rangeBetween(-(w-2), 0)))
        made.append(name)
    return df.drop('month_idx', '_mom'), made

def add_upb_slopes(df: DataFrame, windows: Sequence[int] = WINDOWS, loan_col: str = 'loan_identifier',
                   order_col: str = 'period') -> Tuple[DataFrame, List[str]]:
    """The average monthly balance change per window (temporary columns, turned into percentages of the original
    balance by add_interactions)."""
    if not {'current_actual_upb', 'loan_age'} <= set(df.columns):
        log.warning('No balance rate of change')
        return df, []
    return _add_mom_change_avg(df, 'current_actual_upb', UPB_SLOPE_PREFIX, windows,
                               loan_col, order_col, exact_balance=True)

def add_value_slopes(df: DataFrame, windows: Sequence[int] = WINDOWS, loan_col: str = 'loan_identifier',
                     order_col: str = 'period') -> Tuple[DataFrame, List[str]]:
    """The average monthly change of the estimated property value per window (temporary columns, turned into
    percentages of the original value by add_interactions)."""
    if not {'current_actual_upb', 'estimated_loan_to_value', 'loan_age'} <= set(df.columns):
        log.warning('No property value rate of change')
        return df, []
    df = df.withColumn('_value', estimated_property_value())
    df, made = _add_mom_change_avg(df, '_value', VALUE_SLOPE_PREFIX, windows, loan_col, order_col,
                                   exact_balance=True)
    return df.drop('_value'), made

def add_rate_slopes(df: DataFrame, windows: Sequence[int] = WINDOWS, loan_col: str = 'loan_identifier',
                    order_col: str = 'period') -> Tuple[DataFrame, List[str]]:
    """The average monthly change of the current interest rate per window; already in rate points, so the columns
    get their final names here."""
    if 'current_interest_rate' not in df.columns:
        log.warning('No interest rate rate of change')
        return df, []
    return _add_mom_change_avg(df, 'current_interest_rate', 'current_interest_rate_roc_l',
                               windows, loan_col, order_col)

def _prev_month(col: str, ordered):
    return F.max(col).over(ordered.rangeBetween(-1, -1))

def add_delinquency_dynamics(df: DataFrame, windows: Sequence[int] = WINDOWS, loan_col: str = 'loan_identifier',
                             order_col: str = 'period') -> Tuple[DataFrame, List[str]]:
    """dpd itself, months since the last 30+ and 60+ DPD month (capped at RECENCY_MONTHS), per window the length of
    the current, 30+ and 60+ DPD streaks (a missing month breaks a streak), and the number of roll-forwards and
    cures."""
    if 'dpd' not in df.columns:
        log.warning('No delinquency dynamics')
        return df, []
    df = df.withColumn('month_idx', F.year(order_col)*12 + F.month(order_col))
    ordered = Window.partitionBy(loan_col).orderBy('month_idx')
    idx, dpd = F.col('month_idx'), F.col('dpd')
    made: List[str] = ['dpd']

    for label, level in (('dpd30', 1), ('dpd60', 2)):
        last = F.max(F.when(dpd >= level, idx)).over(ordered.rangeBetween(-(RECENCY_MONTHS-1), 0))
        name = f'months_since_last_{label}'
        df = df.withColumn(name, F.coalesce(idx - last, F.lit(RECENCY_MONTHS)).cast('tinyint'))
        made.append(name)

    df = df.withColumn('_prev_row_idx', F.lag('month_idx').over(ordered))
    gap_before = F.col('_prev_row_idx').isNull() | (idx - F.col('_prev_row_idx') > 1)
    breaks = []
    for state, cond in STREAK_STATES.items():
        col = f'_break_{state}'
        df = df.withColumn(col, F.when(~F.coalesce(cond(dpd), F.lit(False)), idx).when(gap_before, idx-1))
        breaks.append((state, col))
    for w in windows:
        frame = ordered.rangeBetween(-(w-1), 0)
        for state, col in breaks:
            name = f'{state}_streak_l{w}m'
            last = F.max(col).over(frame)
            df = df.withColumn(name, (idx - F.coalesce(last, idx - w)).cast('tinyint'))
            made.append(name)

    df = df.withColumn('_prev_dpd', _prev_month('dpd', ordered))
    prev = F.col('_prev_dpd')
    moves = {'n_roll_forward': dpd > prev, 'n_cure': (prev >= 1) & (dpd == 0)}
    for w in windows:
        frame = ordered.rangeBetween(-(w - 2), 0)
        for stem, cond in moves.items():
            name = f'{stem}_l{w}m'
            df = df.withColumn(name, F.sum(F.when(F.coalesce(cond, F.lit(False)), 1).otherwise(0))
                               .over(frame).cast('tinyint'))
            made.append(name)

    return df.drop('month_idx', '_prev_row_idx', '_prev_dpd', *[c for _, c in breaks]), made

def add_curtailment(df: DataFrame, windows: Sequence[int] = WINDOWS, loan_col: str = 'loan_identifier',
                    order_col: str = 'period') -> Tuple[DataFrame, List[str]]:
    """Per window, the number of months with a curtailment: a current loan, past the rounded-balance age, repaying
    more than CURTAIL_MULTIPLE times its scheduled principal from the amortisation formula."""
    need = {'current_actual_upb', 'current_interest_rate', 'remaining_months_to_legal_maturity', 'loan_age', 'dpd'}
    if not need <= set(df.columns):
        log.warning('No curtailment counts: missing %s', ', '.join(sorted(need - set(df.columns))))
        return df, []
    df = df.withColumn('month_idx', F.year(order_col)*12 + F.month(order_col))
    ordered = Window.partitionBy(loan_col).orderBy('month_idx')
    for c in ('current_actual_upb', 'current_interest_rate', 'remaining_months_to_legal_maturity', 'loan_age', 'dpd'):
        df = df.withColumn(f'_prev_{c}', _prev_month(c, ordered))
    bal = F.col('_prev_current_actual_upb')
    r = F.col('_prev_current_interest_rate') / F.lit(1200)
    n = F.col('_prev_remaining_months_to_legal_maturity')
    payment = F.when(n > 0, F.when(r > 0, bal * r / (F.lit(1) - F.pow(F.lit(1) + r, -n))).otherwise(bal / n))
    scheduled = payment - bal * r
    repaid = bal - F.col('current_actual_upb')
    judged = ((F.col('_prev_loan_age') > UPB_ROUNDED_MAX_AGE) & (F.col('_prev_dpd') == 0) & (F.col('dpd') == 0)
              & (bal > 0) & (F.col('current_actual_upb') > 0) & (n > 0))
    df = df.withColumn('_curtail', F.when(judged, F.when(scheduled > 0, repaid > scheduled * CURTAIL_MULTIPLE)))
    made: List[str] = []
    for w in windows:
        name = f'n_curtail_l{w}m'
        df = df.withColumn(name, F.sum(F.when(F.coalesce(F.col('_curtail'), F.lit(False)), 1).otherwise(0))
                           .over(ordered.rangeBetween(-(w-2), 0)).cast('tinyint'))
        made.append(name)
    return df.drop('month_idx', '_curtail', *[c for c in df.columns if c.startswith('_prev_')]), made

def add_servicer_change(df: DataFrame, loan_col: str = 'loan_identifier',
                        order_col: str = 'period') -> Tuple[DataFrame, List[str]]:
    """1 when the loan had more than one servicer over the last 12 months."""
    if 'servicer_name' not in df.columns:
        log.warning('No servicer change')
        return df, []
    df = df.withColumn('month_idx', F.year(order_col)*12 + F.month(order_col))
    ordered = Window.partitionBy(loan_col).orderBy('month_idx')
    names = F.collect_set('servicer_name').over(ordered.rangeBetween(-(RECENCY_MONTHS-1), 0))
    df = df.withColumn('servicer_changed_l12m',
                       F.when(F.size(names) > 1, 1).otherwise(0).cast('tinyint'))
    return df.drop('month_idx'), ['servicer_changed_l12m']

def add_interactions(df: DataFrame) -> Tuple[DataFrame, List[str]]:
    """The ratio and interaction features (rate gap, LTV x DTI, balance and LTV against origination, negative
    equity, property value and its change) and the slope features as percentages of the original balance or
    value; the temporary slope columns are dropped. A feature whose inputs are missing is skipped with a
    warning."""
    have = set(df.columns)
    made: List[str] = []

    def put(name, expr):
        nonlocal df
        df = df.withColumn(name, expr.cast('double'))
        made.append(name)

    if {'current_interest_rate', 'original_interest_rate'} <= have:
        put('rate_gap', F.col('current_interest_rate') - F.col('original_interest_rate'))
    if {'estimated_loan_to_value', 'original_debt_to_income_ratio'} <= have:
        put('ltv_x_dti', F.col('estimated_loan_to_value') * F.col('original_debt_to_income_ratio') / F.lit(100))
    if {'current_actual_upb', 'original_upb'} <= have:
        put('upb_pct_of_original', F.when(F.col('original_upb') > 0,
                                          F.col('current_actual_upb') / F.col('original_upb') * F.lit(100)))

    if {'estimated_loan_to_value', 'original_loan_to_value'} <= have:
        put('ltv_ratio_to_original', F.when(F.col('original_loan_to_value') > 0,
                                            F.col('estimated_loan_to_value') / F.col('original_loan_to_value')))

    if 'estimated_loan_to_value' in have:
        put('negative_equity_flag', F.when(F.col('estimated_loan_to_value').isNotNull(),
                                           (F.col('estimated_loan_to_value') > 100).cast('int')))
    if {'current_actual_upb', 'estimated_loan_to_value'} <= have:
        put('estimated_property_value', estimated_property_value())
    orig_value_ok = {'original_upb', 'original_loan_to_value'} <= have
    if orig_value_ok and {'current_actual_upb', 'estimated_loan_to_value'} <= have:
        put('property_value_change', estimated_property_value() / original_property_value() - F.lit(1))

    slopes = sorted((c for c in have if c.startswith(UPB_SLOPE_PREFIX)),
                    key=lambda c: int(c[len(UPB_SLOPE_PREFIX):-1]))
    if slopes and 'original_upb' in have:
        for c in slopes:
            put(f'upb_pct_of_original_roc_l{c[len(UPB_SLOPE_PREFIX):]}',
                F.when(F.col('original_upb') > 0, F.col(c) / F.col('original_upb') * F.lit(100)))
    elif slopes:
        log.warning('No upb_pct_of_original_roc')
    df = df.drop(*slopes)

    slopes = sorted((c for c in have if c.startswith(VALUE_SLOPE_PREFIX)),
                    key=lambda c: int(c[len(VALUE_SLOPE_PREFIX):-1]))
    if slopes and orig_value_ok:
        for c in slopes:
            put(f'property_value_roc_l{c[len(VALUE_SLOPE_PREFIX):]}',
                F.col(c) / original_property_value() * F.lit(100))
    elif slopes:
        log.warning('No property_value_roc')
    df = df.drop(*slopes)

    for name, needs in INTERACTIONS.items():
        if name not in made:
            absent = [c for c in needs if c not in have]
            log.warning("No %s: missing %s", name, ", ".join(absent))
    return df, made

def _months(start, end) -> List[Tuple[int, int]]:
    """Every (year, month) from start to end."""
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out

def _shift(y: int, m: int, back: int) -> Tuple[int, int]:
    """(year, month) moved back by back months."""
    i = y * 12 + (m - 1) - back
    return i // 12, i % 12 + 1

def _idx(y: int, m: int) -> int:
    return y * 12 + m

def main() -> None:
    """Join each observation month to its loan window features and write training_set month by month. Each chunk is
    repartitioned to one Spark partition per (month, bucket), the bucket a hash of the loan, so each month is
    written as about files_per_month files; a plain coalesce would leave every month's rows in every partition
    after the join and write files_per_month x months-in-chunk files per month. Only this run's months are
    written, as the output can hold months from other runs."""
    p = argparse.ArgumentParser(description='feature_engineering_main')
    add_period_arguments(p)
    p.add_argument('--performance', default=curated_path('performance'))
    p.add_argument('--observations', default=curated_path('observations'),
                   help='output of src/jobs/observations_main.py')
    p.add_argument('--out', default=curated_path('training_set'))
    p.add_argument('--stats-json', default=str(settings.data_curated / 'training_set_stats.json'))
    p.add_argument('--windows', default=','.join(str(x) for x in WINDOWS),
                   help='trailing horizons in months')
    p.add_argument('--emit-months', type=int, default=0)
    p.add_argument('--files-per-month', type=int, default=8)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()

    period = period_from_args(a)
    windows = tuple(int(x) for x in a.windows.split(',') if x.strip())
    w_max = max(windows)
    emit = a.emit_months if a.emit_months > 0 else w_max

    spark = spark_session('feature_engineering')
    spark.conf.set('spark.sql.sources.partitionOverwriteMode', 'dynamic')
    try:
        parts = month_partitions(spark, a.performance)
        if not parts:
            raise SystemExit(f'{a.performance} is not partitioned by year/month')
        have = {(y, m): uri for y, m ,uri in parts}
        root = resolve_path(a.performance)

        obs_parts = month_partitions(spark, a.observations)
        obs_have = {(y, m): uri for y, m, uri in obs_parts} if obs_parts else {}
        obs_root = resolve_path(a.observations)
        partitioned = bool(obs_have)
        obs_all = None
        if not partitioned:
            obs_all = spark.read.parquet(a.observations)
            log.warning('The observations at %s are NOT partitioned by year/month', a.observations)

        wanted = _months(period.start, period.end)
        done_already = set()
        if a.resume:
            try:
                d = spark.read.parquet(a.out).select('year', 'month').distinct()
                done_already = {(int(r['year']), int(r['month'])) for r in d.collect()}
                log.info('Resuming: %s months already written', len(done_already))
            except Exception:
                log.info('Resuming: nothing written yet')

        chunks = [wanted[i:i + emit] for i in range(0, len(wanted), emit)]
        log.info("%s observation months in %s chunks of up to %s, each reading "
                 "%s partitions (%s of history + the months it emits)",
                 len(wanted), len(chunks), emit, emit + w_max - 1, w_max - 1)

        window_cols: List[str] = []
        made: List[str] = []
        written = 0
        for n, chunk in enumerate(chunks, 1):
            todo = [ym for ym in chunk if ym not in done_already]
            if not todo:
                log.info("[%s/%s] %s-%02d .. %s-%02d already written, skipping",
                         n, len(chunks), chunk[0][0], chunk[0][1],
                         chunk[-1][0], chunk[-1][1])
                continue
            lo = _shift(chunk[0][0], chunk[0][1], w_max - 1)
            hi = chunk[-1]

            uris = [uri for (y, m), uri in sorted(have.items()) if _idx(*lo) <= _idx(y, m) <= _idx(*hi)]
            if not uris:
                log.warning("[%s/%s] no panel partitions for %s-%02d .. %s-%02d",
                            n, len(chunks), lo[0], lo[1], hi[0], hi[1])
                continue

            panel = spark.read.option('basePath', root).parquet(*uris)
            keep = (['loan_identifier', 'period']
                    + [c for c in dict.fromkeys(('dpd', 'current_actual_upb', 'estimated_loan_to_value',
                                                 'current_interest_rate', 'remaining_months_to_legal_maturity',
                                                 'loan_age', 'servicer_name')) if c in panel.columns])
            panel, window_cols = add_window_features(panel.select(*keep), windows=windows)
            panel, rate_cols = add_rate_slopes(panel, windows=windows)
            panel, dyn_cols = add_delinquency_dynamics(panel, windows=windows)
            panel, curtail_cols = add_curtailment(panel, windows=windows)
            panel, servicer_cols = add_servicer_change(panel)
            window_cols = window_cols + rate_cols + dyn_cols + curtail_cols + servicer_cols
            panel, slope_cols = add_upb_slopes(panel, windows=windows)
            panel, value_cols = add_value_slopes(panel, windows=windows)
            slope_cols += value_cols
            first_emit = _idx(*chunk[0])
            panel = (panel
                     .filter((F.year('period') * 12 + F.month('period')).between(first_emit, _idx(*hi)))
                     .select('loan_identifier', 'period', *window_cols, *slope_cols))

            if partitioned:
                obs_uris = [obs_have[ym] for ym in chunk if ym in obs_have]
                if not obs_uris:
                    log.info("[%s/%s] no observations for %s-%02d .. %s-%02d, "
                             "skipping", n, len(chunks), chunk[0][0], chunk[0][1],
                             hi[0], hi[1])
                    continue
                obs = (spark.read.option('basePath', obs_root).parquet(*obs_uris))
            else:
                obs = obs_all.filter((F.year('period') * 12 + F.month('period')).between(first_emit, _idx(*hi)))

            feat = obs.join(panel, ['loan_identifier', 'period'], how='left')
            feat, made = add_interactions(feat)
            feat = feat.drop(*NOT_FEATURES)

            files = max(1, a.files_per_month)
            bucket = F.pmod(F.hash('loan_identifier'), F.lit(files))
            (feat
             .withColumn('year', F.year('period'))
             .withColumn('month', F.month('period'))
             .repartition(files * len(chunk), 'year', 'month', bucket)
             .write.mode('overwrite').partitionBy('year', 'month')
             .parquet(a.out)
             )
            written += len(chunk)
            log.info("[%s/%s] %s-%02d .. %s-%02d written from %s partitions",
                     n, len(chunks), chunk[0][0], chunk[0][1], hi[0], hi[1],
                     len(uris))

        if not window_cols:
            if a.resume and done_already:
                log.info('Resuming: every month with observations is already written; nothing to do.')
                return
            raise SystemExit("Nothing was written.")
        if made:
            log.info('Interactions: %s', ', '.join(made))

        in_run = (F.year('period') * 12 + F.month('period')).between(_idx(*wanted[0]), _idx(*wanted[-1]))
        done = spark.read.parquet(a.out).filter(in_run)
        cols = window_cols + made
        census = done.agg(
            F.count(F.lit(1)).alias('rows'),
            *[F.sum(F.when(F.col(c).isNotNull(), 1).otherwise(0)).alias(f'nn::{c}')
              for c in cols]).collect()[0].asDict()
        rows = int(census['rows'])

        obs_rows = None
        if partitioned:
            run_uris = [obs_have[ym] for ym in wanted if ym in obs_have]
            obs_rows = (int(spark.read.option('basePath', obs_root).parquet(*run_uris).count())
                        if run_uris else 0)
        elif obs_all is not None:
            obs_rows = int(obs_all.filter(in_run).count())
        if obs_rows is not None and obs_rows != rows:
            log.warning("ROW COUNT MISMATCH: %s observations but %s training "
                        "rows (%+d).",f"{obs_rows:,}", f"{rows:,}", rows - obs_rows)
        elif obs_rows is not None:
            log.info("Row count conserved: %s observations in, %s training rows "
                     "out", f"{obs_rows:,}", f"{rows:,}")
        empty = [c for c in cols if not census.get(f"nn::{c}")]
        log.info("Training set: %s rows, %s engineered columns, %s months",
                 f"{rows:,}", len(cols), written)
        if empty:
            log.warning("ALL NULL: %s",", ".join(empty))
        stats = {"rows": rows, "observation_rows": obs_rows, "window_features": window_cols, "interactions": made,
                 "windows": list(windows),
                 "emit_months": emit, "partitions_per_chunk": emit + w_max - 1,
                 "chunks": len(chunks), "months_written": written,
                 "observations_partitioned": partitioned,
                 "all_null_columns": empty,
                 "non_null": {c: int(census.get(f"nn::{c}") or 0) for c in cols},
                 "observations": a.observations, "source": "src/jobs/feature_engineering_main.py"}
    finally:
        spark.stop()

    Path(a.stats_json).write_text(json.dumps(stats, indent=2, default=str))
    log.info("Stats in %s", a.stats_json)

if __name__ == "__main__":
    main()

