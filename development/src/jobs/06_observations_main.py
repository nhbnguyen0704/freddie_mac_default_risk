import argparse
import json
import logging
from datetime import date
from pathlib import Path
from typing import Dict, Tuple
from pyspark.sql import DataFrame, Window, functions as F
from helpers.config import settings
from helpers.label import LabelSpec, resolve
from helpers.layout import (CENSORING_CODES, CREDIT_EVENT_CODES, PREPAID_CODES)
from helpers.period import (add_arguments as add_period_arguments, from_args as period_from_args,
                            observation_mask_spark, read_panel_spark, resolve_asof)
from helpers.population import restrict, workout_condition
from helpers.spark_io import curated_path, read_curated, resolve_path, spark_session

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s | %(message)s')
log = logging.getLogger('build')

PERF_NEEDED = ['loan_identifier', 'period', 'dpd', 'loan_age', 'current_actual_upb', 'current_interest_rate',
               'remaining_months_to_legal_maturity', 'estimated_loan_to_value', 'modification_flag',
               'payment_deferral_flag', 'borrower_assistance_plan',
               'current_non_interest_bearing_upb', 'servicer_name', 'zero_balance_code']

def _zbc(col):
    """The zero-balance code padded to two digits, blank as null."""
    trimmed = F.trim(col)
    return F.when(col.isNull() | (trimmed == ''), None).otherwise(F.lpad(trimmed, 2, "0"))

def build_observations(perf: DataFrame, orig: DataFrame, spec: LabelSpec, asof: date,
                       period=None) -> Tuple[DataFrame, Dict[str, object]]:
    """The labelled loan-month observations and the count each rule removes. A month is bad when the loan reaches
    the DPD threshold, a credit event or (with special mention) a workout ever. The label is 1 when the loan is
    bad in any month of t+1..t+window ('ever') or in the window's last month ('at_end'). Kept rows: inside the
    period (rule 0), not already bad in t (rule 2), window closed by asof (rule 4), no censoring exit (whole
    loan sale, securitisation, defect) inside the window (rule 5), no payoff inside the window when prepay is
    'censor', and no unexplained disappearance before the window ends. Origination attributes are joined on."""
    thr = spec.dpd_threshold
    w = spec.window_months

    df = perf.select(*[c for c in PERF_NEEDED if c in perf.columns])

    df = df.withColumn('month_idx', F.year('period') * 12 + F.month('period'))

    part = Window.partitionBy('loan_identifier')
    ordered = part.orderBy('month_idx')
    fwd = ordered.rangeBetween(0, Window.unboundedFollowing)
    whole = ordered.rangeBetween(Window.unboundedPreceding, Window.unboundedFollowing)

    zbc = _zbc(F.col('zero_balance_code'))
    df = df.withColumn('zb', zbc)

    is_bad = F.coalesce(F.col('dpd') >= thr, F.lit(False))
    if spec.include_credit_event:
        is_bad = is_bad | F.coalesce(F.col('zb').isin(sorted(CREDIT_EVENT_CODES)), F.lit(False))

    bad_dpd_only = is_bad

    workout = (workout_condition(perf, include_prior=True) if spec.include_special_mention else F.lit(False))
    if spec.include_special_mention:
        is_bad = is_bad | workout

    df = (df.withColumn('is_bad_month', F.when(is_bad, 1).otherwise(0))
          .withColumn('_bad_dpd_only', F.when(bad_dpd_only, 1).otherwise(0)))

    df = (df
          .withColumn('next_bad_month', F.min(F.when(F.col('is_bad_month') == 1, F.col('period'))).over(fwd))
          .withColumn('next_bad_month_dpd_only',
                      F.min(F.when(F.col('_bad_dpd_only') == 1, F.col('period'))).over(fwd))
          .withColumn('last_observed_month', F.max('period').over(part))
          )

    all_exits = sorted(CREDIT_EVENT_CODES | PREPAID_CODES | CENSORING_CODES)
    is_exit = F.col('zb').isin(all_exits)
    df = (df
          .withColumn('exit_month', F.min(F.when(is_exit, F.col('period'))).over(part))
          .withColumn('exit_code', F.first(F.when(is_exit, F.col('zb')), True).over(whole)))

    df = df.withColumn('window_end', F.add_months(F.col('period'), w))
    asof_lit = F.lit(asof)

    in_period = observation_mask_spark(period) if period is not None else F.lit(True)

    already = F.col('is_bad_month') == 1

    closed = F.col('window_end') <= asof_lit

    if spec.basis == 'ever':
        label = (F.col('next_bad_month').isNotNull()
                 & (F.col('next_bad_month') > F.col('period'))
                 & (F.col('next_bad_month') <= F.col('window_end')))
    elif spec.basis == 'at_end':
        at_end = (df.select(F.col('loan_identifier').alias('_l'),
                            F.col('period').alias('_p'),
                            F.col('is_bad_month').alias('bad_at_window_end')))
        df = df.join(at_end,
                     (df['loan_identifier'] == at_end['_l']) & (df['window_end'] == at_end['_p']),
                     'left').drop('_l', '_p')
        label = F.coalesce(F.col('bad_at_window_end') == 1, F.lit(False))
    else:
        raise ValueError(f'Unknown label basis {spec.basis!r}')
    df = df.withColumn('label', F.when(label, 1).otherwise(0))

    def _exit_in_window(codes):
        return (F.col('exit_code').isin(codes)
                & (F.col('exit_month') > F.col('period'))
                & (F.col('exit_month') <= F.col('window_end'))
                & (F.col('label') == 0))

    censored = F.coalesce(_exit_in_window(CENSORING_CODES), F.lit(False))
    prepaid_drop = (F.coalesce(_exit_in_window(PREPAID_CODES), F.lit(False))
                    if spec.prepay == 'censor' else F.lit(False))

    unexplained = (F.coalesce(F.col('last_observed_month') < F.col('window_end'), F.lit(False))
                   & (F.col('label') == 0) & F.col('exit_code').isNull())

    e0 = in_period
    e2 = e0 & ~already
    e4 = e2 & closed
    e5 = e4 & ~censored
    e6 = e5 & ~prepaid_drop
    e7 = e6 & ~unexplained
    keep = e7

    seen_by_dpd = (F.col('next_bad_month_dpd_only').isNotNull()
                   & (F.col('next_bad_month_dpd_only') > F.col('period'))
                   & (F.col('next_bad_month_dpd_only') <= F.col('window_end')))

    def _n(cond):
        return F.sum(F.when(cond, 1).otherwise(0))

    log.info('Counting what each rule removes')
    c = df.agg(
        _n(F.lit(True)).alias('read'),
        _n(e0).alias('in_period'),
        _n(e0 & already).alias('already_bad'),
        _n(e2 & ~closed).alias('not_enough_observed_months'),
        _n(e4 & censored).alias('censored'),
        _n(e5 & prepaid_drop).alias('prepaid'),
        _n(e6 & unexplained).alias('unexplained_exit'),
        _n(keep).alias('keep'),
        _n(keep & (F.col('label') == 1)).alias('num_bad'),
        _n(keep & (F.col('label') == 1) & ~seen_by_dpd).alias('num_workout_only_bad')
    ).collect()[0]

    stats = {k: v or 0 for k, v in c.asDict().items()}
    stats['bad_rate'] = (stats['num_bad'] / stats['keep']) if stats['keep'] > 0 else 0
    log.info('Rule 0: %s loan-months outside %s', f'{stats["read"] - stats["in_period"]:,}',
             getattr(period, 'label', 'the period'))
    log.info('Rule 2: %s dropped, bad that month%s', f'{stats["already_bad"]:,}',
             ' or in a workout that month' if spec.include_special_mention else '')
    log.info('Rule 4: %s dropped, insufficient follow-up (window ends after %s).',
             f'{stats["not_enough_observed_months"]:,}', asof)
    log.info('Rule 5: %s dropped, censored by a whole loan sale, securitisation or defect exit inside the window',
             f'{stats["censored"]:,}')
    if spec.prepay == 'censor':
        log.info('%s dropped, prepaid inside the window.', f'{stats["prepaid"]:,}')
    if stats['unexplained_exit']:
        log.warning('%s dropped, unexplained exit.', f'{stats["unexplained_exit"]:,}')
    if spec.include_special_mention:
        log.info('Special mention: %s of %s bads are workouts that days past due alone would have scored good',
                 f'{stats["num_workout_only_bad"]:,}', f'{stats["num_bad"]:,}')
    log.info('Observations: %s, of which %s bad (%.4f%%)',
             f'{stats["keep"]:,}', f'{stats["num_bad"]:,}', 100 * stats['bad_rate'])

    out = (df.filter(keep)
           .select('loan_identifier', 'period', 'loan_age', 'current_actual_upb', 'current_interest_rate',
                   'remaining_months_to_legal_maturity', 'estimated_loan_to_value', 'servicer_name', 'label')
           .withColumn('observation_date', F.col('period'))
           .withColumn('year', F.year('period'))
           .withColumn('month', F.month('period'))
           )

    if orig is not None:
        o = orig
        if "origination_quarter" in o.columns:
            o = o.withColumnRenamed("origination_quarter", "vintage")
        o = o.drop(*[c for c in ('year', 'quarter', 'month') if c in o.columns])
        out = out.join(o, 'loan_identifier', 'left')
    return out, stats

def main() -> None:
    p = argparse.ArgumentParser(description='Observations main')
    add_period_arguments(p)
    p.add_argument('--performance', default=curated_path('performance'))
    p.add_argument('--origination', default=curated_path('origination'))
    p.add_argument('--out', default=curated_path('observations'))
    p.add_argument('--stats-json', default=str(settings.data_curated / 'observations_stats.json'))
    p.add_argument('--labelspec-json', default=str(settings.data_curated / 'observations_labelspec.json'))
    p.add_argument('--roll-rate-json',
                   default=str(settings.data_curated / "roll_rate" / "recommended_definition.json"))
    p.add_argument('--vintage-json',
                   default=str(settings.data_curated / "vintage" / "suggested_window.json"))
    p.add_argument("--label-capture", type=float, default=None)
    p.add_argument("--keep-disaster", action="store_true")
    p.add_argument("--asof", default="auto")
    p.add_argument("--no-origination", action="store_true")
    a = p.parse_args()

    period = period_from_args(a)
    spec = resolve(Path(a.roll_rate_json), Path(a.vintage_json), a.label_capture)
    log.info("Label spec: %s (%s+ DPD, %s-month window)",
             spec.name, 30 * spec.dpd_threshold, spec.window_months)

    spark = spark_session("build")
    try:
        perf, period_stats = read_panel_spark(spark, a.performance, period)
        perf, population = restrict(perf, exclude_workouts=False, exclude_disaster=not a.keep_disaster)
        observed = perf.agg(F.max('period')).collect()[0][0]
        asof, asof_source = resolve_asof(a.asof, observed, period)
        log.info('As-of %s -- %s', asof, asof_source)

        orig = None
        if not a.no_origination:
            orig = read_curated(spark, resolve_path(a.origination))

        out, stats = build_observations(perf, orig, spec, asof, period)
        log.info('Writing observations to %s', a.out)
        out.write.mode('overwrite').partitionBy('year', 'month').parquet(a.out)

        done = spark.read.parquet(a.out)
        agg = [F.count(F.lit(1)).alias('rows')]
        if 'vintage' in done.columns:
            agg += [F.countDistinct("vintage").alias("vintages"),
                    F.min("vintage").alias("first_vintage"),
                    F.max("vintage").alias("last_vintage")]

        if 'original_upb' in done.columns:
            agg += [F.sum(F.when(F.col("original_upb").isNull(), 1)
                          .otherwise(0)).alias("missing_origination")]
        d = done.agg(*agg).collect()[0].asDict()
        stats.update({k: (int(v) if isinstance(v, (int, float)) and v is not None
                          else v) for k, v in d.items() if v is not None})
        if d.get("missing_origination"):
            log.warning("%s observations have no matching origination row",
                        f"{int(d['missing_origination']):,}")
        if d.get("vintages"):
            log.info("Vintages represented: %s (%s .. %s)", d["vintages"],
                     d["first_vintage"], d["last_vintage"])
    finally:
        spark.stop()

    stats.update(period_stats)
    stats["analysed_population"] = population
    stats["asof"] = str(asof)
    stats["asof_source"] = asof_source
    stats["excluded_disaster"] = not a.keep_disaster
    stats["label_spec"] = spec.name
    stats["source"] = "src/jobs/observations_main.py"
    Path(a.stats_json).write_text(json.dumps(stats, indent=2, default=str))
    Path(a.labelspec_json).write_text(json.dumps(spec.to_dict(), indent=2, default=str))
    log.info("Stats in %s, label spec in %s", a.stats_json, a.labelspec_json)

if __name__ == '__main__':
    main()

