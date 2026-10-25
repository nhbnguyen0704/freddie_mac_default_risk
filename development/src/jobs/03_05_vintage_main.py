import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from helpers.config import settings
from helpers.population import caveat, restrict, workout_condition
from helpers.report import md_table, write_report
from helpers.spark_io import curated_path, read_curated, resolve_path, spark_session
from helpers.period import add_arguments as add_period_arguments
from helpers.period import from_args as period_from_args, read_panel_spark
from helpers.period import observation_mask_spark
from helpers.layout import PREPAID_CODES
from pyspark.sql import Window, functions as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s")
log = logging.getLogger('vintage')

CAPTURE_TARGETS = (0.50, 0.75, 0.80)
RECOMMENDED_CAPTURE = 0.75
CROWDING_SHARE = 0.80

def _observation_spans(perf, bad_dpd: int, freq: str, asof: pd.Timestamp, period=None,
                       bad_includes_workout: bool = False):
    """Spans keyed on OBSERVATION month: every loan-month not yet bad opens a cohort membership, and age is months
    since that observation. Per member: at_risk_to (cohort cap if it went bad, else the last payable month),
    hazard_to (the age it went bad, else the last payable month), bad_at and prepaid_at. A loan with less than a
    month of legal term left has no payment to miss, so the span ends at maturity (unknown term stays eligible);
    bad_month is not gated on the term, or a loan whose first 90+ month falls after its term expired would never
    be recorded bad. With bad_includes_workout a workout month counts as bad, which agrees with the warehouse
    build and is the only way to see the modified loans that never register 90+ DPD because capitalising the
    arrears resets the status, and a month already in a workout cannot open a membership. Quarterly cohorts stop
    at the age their last month can reach, so every member can be seen at every age reported."""
    w = Window.partitionBy('loan_identifier')
    t0 = F.col('period')
    period_col = F.col('period')
    month_idx = F.year(period_col) * 12 + F.month(period_col)
    fwd = (Window.partitionBy('loan_identifier').orderBy('month_idx').rangeBetween(1, Window.unboundedFollowing))
    payable = F.coalesce(F.col('remaining_months_to_legal_maturity') >= 1, F.lit(True))

    zbc = F.lpad(F.trim(F.col('zero_balance_code')), 2, '0')
    is_prepaid = F.col('zero_balance_code').isNotNull() & zbc.isin(sorted(PREPAID_CODES))

    is_bad = F.col('dpd') >= bad_dpd
    if bad_includes_workout:
        is_bad = is_bad | (F.col('workout') == 1)

    wk = F.when(workout_condition(perf), 1).otherwise(0)

    df = (
        perf.select('loan_identifier', 'period', 'dpd',
                    'remaining_months_to_legal_maturity', 'zero_balance_code',
                    wk.alias('workout'))
        .withColumn('month_idx', month_idx)
        .withColumn('obs_end', F.max(F.when(payable, t0)).over(w))
        .withColumn('bad_month',
                    F.min(F.when(is_bad, t0)).over(fwd))
        .withColumn('prepaid_month', F.min(F.when(is_prepaid, t0)).over(fwd))
    )
    if period is not None:
        df = df.filter(observation_mask_spark(period))

    eligible = F.coalesce(F.col("dpd") < bad_dpd, F.lit(True)) & payable
    if bad_includes_workout:
        eligible = eligible & (F.col('workout') == 0)
    counted = df.agg(
        F.count(F.lit(1)).alias('rows'),
        F.sum(F.when(eligible, 1).otherwise(0)).alias('members'),
        F.countDistinct('loan_identifier').alias('loans')
    ).collect()[0]
    n_rows, n_members = int(counted['rows']), int(counted['members'])
    stats = {
        'panel_rows': n_rows,
        'cohort_members': n_members,
        'loans': int(counted['loans']),
        'dropped_already_bad': n_rows - n_members,
        'cohort_freq': 'quarterly' if freq.upper().startswith('Q') else 'monthly'
    }
    log.info("Observation cohorts: %s memberships from %s loan-months across %s "
             "loans. %s already bad observations dropped )",
             f"{n_members:,}", f"{n_rows:,}", f"{stats['loans']:,}",
             f"{n_rows - n_members:,}")

    k_obs = F.months_between(F.col('obs_end'), t0).cast('int')
    k_bad = F.months_between(F.col('bad_month'), t0).cast('int')
    k_prepaid = F.months_between(F.col('prepaid_month'), t0).cast('int')

    ever_bad = F.col('bad_month').isNotNull()
    ever_prepaid = F.col('prepaid_month').isNotNull()

    quarterly = freq.upper().startswith('Q')
    cohort_start = F.trunc(t0, 'quarter' if quarterly else 'month')
    label = (F.concat(F.year(cohort_start), F.lit('Q'), F.quarter(cohort_start))
             if quarterly else F.date_format(cohort_start, 'yyyy-MM'))
    cohort_cap = F.months_between(F.lit(pd.Timestamp(asof).date()), cohort_start).cast('int')
    cohort_end = F.add_months(cohort_start, 2) if quarterly else cohort_start

    spans = (df.filter(eligible)
             .withColumn('cohort_start', cohort_start)
             .withColumn('cohort_end', cohort_end)
             .withColumn('vintage', label)
             .withColumn('at_risk_to', F.when(ever_bad, cohort_cap).otherwise(k_obs))
             .withColumn('hazard_to', F.when(ever_bad, k_bad).otherwise(k_obs))
             .withColumn('bad_at', F.when(ever_bad, k_bad))
             .withColumn('prepaid_at', F.when(ever_prepaid, k_prepaid))
             .select('vintage', 'cohort_start', 'cohort_end', 'at_risk_to',
                     'hazard_to', 'bad_at', 'prepaid_at'))
    return spans, stats

def _origination_spans(perf, orig, bad_dpd: int, asof: pd.Timestamp, period=None,
                       bad_includes_workout: bool = False):
    """Spans keyed on ORIGINATION quarter, with loan age as the time axis.

    This is the conventional vintage curve and answers a different question
    from `_observation_spans`. There every loan-MONTH is a cohort member and
    the age axis is months since that observation, so one loan contributes many
    spans. Here each LOAN contributes exactly one, the cohort is the quarter it
    was originated in, and age is the loan's own age -- "of the loans written in
    2021Q3, what share had gone bad by month 18".

    THE ANCHOR is `first_payment_date` minus one month, read from the
    origination table. Deriving it from `loan_age` instead would lean on a field
    that is computed rather than recorded and can drift; the first payment date
    is stated at origination, and the month before it is when the loan existed
    but nothing was yet due.

    MATURITY. A loan with less than a month of legal term remaining has no
    payment left to miss, so its span ends there rather than at its last row.

    LEFT TRUNCATION is why this cannot just be a different `groupBy`. The panel
    is floored at the analysis period's start, so a loan originated before that
    floor first appears at some age > 0 and its early life is absent -- we
    cannot know whether it went bad at age 4 when age 4 is not in the data, and
    keeping it would count it as having survived ages it was never observed at.
    Only loans originated on or after the floor are kept, and what that discards
    is reported.

    Events are TIME TO FIRST, not recurrent. Recurrence is right in
    `_observation_spans`, where a cured loan genuinely rejoins the population at
    a fresh observation month. A vintage curve asks when a cohort first
    deteriorates, so a loan is counted once and stays counted.
    """
    payable = F.coalesce(F.col("remaining_months_to_legal_maturity") >= 1, F.lit(True))

    zbc = F.lpad(F.trim(F.col("zero_balance_code")), 2, "0")
    is_prepaid = F.col("zero_balance_code").isNotNull() & zbc.isin(sorted(PREPAID_CODES))

    is_bad = F.col("dpd") >= bad_dpd
    if bad_includes_workout:
        is_bad = is_bad | (F.col("workout") == 1)

    per_loan = (perf
                .select("loan_identifier", "period", "dpd",
                        "remaining_months_to_legal_maturity", "zero_balance_code",
                        F.when(workout_condition(perf), 1).otherwise(0).alias("workout"))
                .groupBy("loan_identifier")
                .agg(F.max(F.when(payable, F.col("period"))).alias("last_payable"),
                     F.min(F.when(is_bad, F.col("period"))).alias("bad_period"),
                     F.min(F.when(is_prepaid, F.col("period"))).alias("prepaid_period"))
                .cache())
    loans_all = per_loan.count()

    anchors = (orig.select("loan_identifier", "first_payment_date")
               .filter(F.col("first_payment_date").isNotNull())
               .withColumn("orig_month",
                           F.add_months(F.trunc(F.col("first_payment_date"), "month"), -1))
               .select("loan_identifier", "orig_month"))

    j = per_loan.join(anchors, "loan_identifier", "inner")
    if period is not None:
        j = j.filter((F.col("orig_month") >= F.lit(pd.Timestamp(period.start).date()))
                     & (F.col("orig_month") <= F.lit(pd.Timestamp(period.end).date())))

    j = (j.withColumn("last_age",
                      F.months_between(F.col("last_payable"), F.col("orig_month")).cast("int"))
         .withColumn("bad_age",
                     F.months_between(F.col("bad_period"), F.col("orig_month")).cast("int"))
         .withColumn("prepaid_age",
                     F.months_between(F.col("prepaid_period"), F.col("orig_month")).cast("int"))
         .filter(F.col("last_age").isNotNull() & (F.col("last_age") >= 0))
         .cache())

    ever_bad = F.col("bad_age").isNotNull() & (F.col("bad_age") >= 0)
    cohort_start = F.trunc(F.col("orig_month"), "quarter")
    label = F.concat(F.year(cohort_start), F.lit("Q"), F.quarter(cohort_start))
    cohort_cap = F.months_between(F.lit(pd.Timestamp(asof).date()), cohort_start).cast("int")
    cohort_end = F.add_months(cohort_start, 2)

    spans = (j
             .withColumn("cohort_start", cohort_start)
             .withColumn("cohort_end", cohort_end)
             .withColumn("vintage", label)
             .withColumn("at_risk_to",
                         F.when(ever_bad, cohort_cap).otherwise(F.col("last_age")))
             .withColumn("hazard_to",
                         F.when(ever_bad, F.col("bad_age")).otherwise(F.col("last_age")))
             .withColumn("bad_at", F.when(ever_bad, F.col("bad_age")))
             .withColumn("prepaid_at",
                         F.when(F.col("prepaid_age").isNotNull()
                                & (F.col("prepaid_age") >= 0), F.col("prepaid_age")))
             .select("vintage", "cohort_start", "cohort_end", "at_risk_to",
                     "hazard_to", "bad_at", "prepaid_at"))

    counted = j.agg(F.count(F.lit(1)).alias("loans"),
                    F.sum(F.when(ever_bad, 1).otherwise(0)).alias("bad")).collect()[0]
    kept = int(counted["loans"])
    stats = {
        "origination_loans": kept,
        "origination_loans_ever_bad": int(counted["bad"] or 0),
        "loans_in_panel": loans_all,
        "loans_outside_period_or_unanchored": loans_all - kept,
        "origination_cohort_freq": "quarterly",
        "origination_anchor": "first_payment_date minus one month (origination table)",
        "origination_left_truncation": (
            "loans originated before the analysis period are excluded: the panel "
            "is floored at the period start, so their early ages are absent and "
            "keeping them would count them as surviving ages never observed"),
    }
    log.info("Origination cohorts: %s loans anchored at first_payment_date-1 inside "
             "the period (%s ever bad). %s of the %s loans in the panel are outside "
             "it or carry no first payment date.",
             f"{kept:,}", f"{stats['origination_loans_ever_bad']:,}",
             f"{stats['loans_outside_period_or_unanchored']:,}", f"{loans_all:,}")
    return spans, stats


def _curves(spans, asof: pd.Timestamp) -> pd.DataFrame:
    """Per cohort and age: the fixed cohort size, the at-risk and hazard risk sets, new and cumulative bads and
    prepayments, the hazard and its Kaplan-Meier cumulative rate. The four age columns are melted into (kind,
    age) pairs so the spans, too large to cache, are aggregated in one groupBy rather than four full shuffles.
    cum_bad_rate is read on the cohort's size at age 0, not on the shrinking at-risk count: dividing by at_risk
    treats prepayment as independent censoring and, with 66% of current loans prepaying inside 54 months, makes
    the rate climb because the denominator drains. Against the fixed size prepayment is a competing risk and the
    curve plateaus at the cohort's lifetime bad rate. The KM rate, 1 - prod(1 - h), answers 'what would default
    if nobody could prepay' and is for shape and timing, not for quoting a level."""
    keys = ['vintage', 'cohort_start', 'cohort_end']

    melted = (spans.select(
        *keys,
        F.explode(F.array(
            F.struct(F.lit('at_risk').alias('kind'), F.col('at_risk_to').alias('age')),
            F.struct(F.lit('hazard').alias('kind'), F.col('hazard_to').alias('age')),
            F.struct(F.lit('bad').alias('kind'), F.col('bad_at').alias('age')),
            F.struct(F.lit('prepaid').alias('kind'), F.col('prepaid_at').alias('age')),
        )).alias('e'))
        .select(*keys, F.col('e.kind').alias('kind'), F.col('e.age').alias('age'))
        .filter(F.col('age').isNotNull()))
    hist = melted.groupBy(*keys, 'kind', 'age').count().toPandas()

    asof_m = pd.Timestamp(asof).to_period('M')

    def _index(kind, col):
        h = hist.loc[hist['kind'] == kind, keys + ['age', 'count']]
        h = h.rename(columns={'age': col})
        return ({k: g for k, g in h.groupby(keys)} if len(h) else {}), h

    at_risk_by_cohort, at_risk_hist = _index('at_risk', 'at_risk_to')
    hazard_by_cohort, _ = _index('hazard', 'hazard_to')
    bad_by_cohort, _ = _index('bad', 'bad_at')
    prepaid_by_cohort, _ = _index('prepaid', 'prepaid_at')

    def _hist(block, col, cap):
        """Per-age counts for one cohort, clipped into [0, cap]."""
        if block is None:
            return np.zeros(cap + 1)
        return np.bincount(block[col].clip(0, cap).to_numpy(int),
                           weights=block['count'].to_numpy(float),
                           minlength=cap + 1)

    out = []
    for (vintage, start, end), grp in at_risk_hist.groupby(keys):
        cap = (asof_m - pd.Timestamp(end).to_period('M')).n
        if cap < 1:
            continue

        counts = np.bincount(grp["at_risk_to"].clip(0, cap).to_numpy(int),
                             weights=grp["count"].to_numpy(float),
                             minlength=cap + 1)
        at_risk = counts[::-1].cumsum()[::-1]
        cohort_size = float(counts.sum())

        key = (vintage, start, end)
        b = bad_by_cohort.get(key)
        bad = _hist(b[b["bad_at"] <= cap] if b is not None else None, "bad_at", cap)
        cum_bad = bad.cumsum()

        pp = prepaid_by_cohort.get(key)
        prepaid = _hist(pp[pp["prepaid_at"] <= cap] if pp is not None else None,
                        "prepaid_at", cap)
        cum_prepaid = prepaid.cumsum()

        hz = _hist(hazard_by_cohort.get(key), "hazard_to", cap)
        at_risk_hazard = hz[::-1].cumsum()[::-1]

        with np.errstate(divide="ignore", invalid="ignore"):
            hazard = np.where(at_risk_hazard > 0, bad / at_risk_hazard, np.nan)
        km = 1.0 - np.cumprod(1.0 - np.nan_to_num(hazard, nan=0.0))

        for k in range(1, cap + 1):
            if at_risk[k] > 0:
                out.append({"vintage": vintage, "months_on_book": k,
                            "max_observable_age": cap,
                            "cohort_size": int(cohort_size),
                            "at_risk": int(at_risk[k]),
                            "at_risk_hazard": int(at_risk_hazard[k]),
                            "new_bad": int(bad[k]),
                            "cum_bad": int(cum_bad[k]),
                            "cum_prepaid": int(cum_prepaid[k]),
                            "hazard": (float(hazard[k])
                                       if np.isfinite(hazard[k]) else None),
                            "km_cum_bad_rate": float(km[k])})

    table = pd.DataFrame(out)
    if table.empty:
        return table
    table = table.sort_values(["vintage", "months_on_book"], ignore_index=True)
    table["cum_bad_rate"] = table["cum_bad"] / table["cohort_size"]
    table["cum_prepaid_rate"] = table["cum_prepaid"] / table["cohort_size"]
    table["hazard_bps"] = 10_000 * table["hazard"]
    return table

def build_vintage_table(perf, asof: pd.Timestamp, bad_dpd: int = 3, orig=None, freq: str = 'M',
                        period=None, bad_includes_workout: bool = False
                        ) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """The observation-month vintage table and its counts."""
    spans, stats = _observation_spans(perf, bad_dpd, freq, asof, period,
                                      bad_includes_workout)

    table = _curves(spans, asof)
    stats = dict(stats, bad_dpd=bad_dpd, vintages=int(table["vintage"].nunique()) if len(table) > 0 else 0)
    log.info("Vintage table: %s rows across %s cohorts",
             f"{len(table):,}", stats["vintages"])
    return table, stats

def build_origination_table(perf, orig, asof: pd.Timestamp, bad_dpd: int = 3,
                            period=None, bad_includes_workout: bool = False
                            ) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """The origination-quarter counterpart of `build_vintage_table`."""
    spans, stats = _origination_spans(perf, orig, bad_dpd, asof, period,
                                      bad_includes_workout)
    table = _curves(spans, asof)
    stats = dict(stats, bad_dpd=bad_dpd,
                 origination_cohorts=int(table["vintage"].nunique()) if len(table) else 0)
    log.info("Origination vintage table: %s rows across %s quarterly cohorts",
             f"{len(table):,}", stats["origination_cohorts"])
    return table, stats


def _round(v, places: int=4):
    """v rounded, or None when missing."""
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else round(v, places)

def _rate_at(pooled: pd.DataFrame, months: Optional[int]) -> Optional[float]:
    """The pooled cumulative bad rate at an age, or None."""
    if months is None:
        return None
    hit = pooled.loc[pooled["months_on_book"] == months, 'cum_bad_rate']
    return float(hit.iloc[0]) if len(hit) else None

def _captured_at(pooled: pd.DataFrame, final: float, months: Optional[int]) -> Optional[float]:
    """The share of the pooled final bad rate reached by an age, or None."""
    rate = _rate_at(pooled, months)
    if rate is None or not final or final <= 0:
        return None
    return float(min(1, rate / final))

def _capture_window(pooled: pd.DataFrame, final: float, share: float) -> Optional[int]:
    """The first age at which the pooled curve reaches share of its final rate, or None."""
    if not final or final <= 0 or pooled.empty:
        return None
    hit = pooled.loc[pooled['cum_bad_rate'] >= share * final, 'months_on_book']
    return int(hit.iloc[0]) if len(hit) else None

def _pool(t: pd.DataFrame) -> pd.DataFrame:
    """The cohorts pooled per age, on the same fixed denominator as the per-cohort curves so the pooled diagnostic
    cannot contradict the recommendation read off them; at_risk is summed and reported but is not the divisor.
    Adds the number of cohorts per age and the marginal rate in bps."""
    pooled = (t.groupby("months_on_book")[["cum_bad", "at_risk", "cohort_size"]].sum()
              .assign(cum_bad_rate=lambda d: d["cum_bad"] / d["cohort_size"])
              .reset_index())
    support = (t.groupby("months_on_book")["vintage"].nunique()
               .rename("cohorts").reset_index())
    pooled = pooled.merge(support, on="months_on_book", how="left")
    pooled["marginal_bps"] = pooled["cum_bad_rate"].diff() * 10_000
    return pooled


def _cohort_capture_ages(t: pd.DataFrame, share: float, min_depth: int) -> Dict[str, int]:
    """Each cohort's own capture age, read off its own plateauing curve."""
    out: Dict[str, int] = {}
    for v, g in t.groupby("vintage"):
        g = g.sort_values("months_on_book")
        if int(g["months_on_book"].max()) < min_depth:
            continue
        fin = float(g["cum_bad_rate"].iloc[-1])
        if fin <= 0:
            continue
        hit = g.loc[g["cum_bad_rate"] >= share * fin, "months_on_book"]
        if len(hit):
            out[str(v)] = int(hit.iloc[0])
    return out


def _captured_by_cohort(t: pd.DataFrame, months: Optional[int]) -> Dict[str, float]:
    """Of the bads a cohort eventually shows, the share that had arrived by `months`.

    Measured against each cohort's OWN eventual count, over only the cohorts deep
    enough to have reached `months` -- the same ones that voted for it in
    `_capture_fixed_point`. Reading it off the pooled curve instead compares a
    per-cohort window against a pooled total, which is how this column came to
    disagree with its own `basis` label: a window chosen so that 75% of a
    cohort's bads have arrived was reported as capturing 49%.
    """
    if not months or t.empty:
        return {}
    deep = t.groupby("vintage")["months_on_book"].max()
    voters = deep[deep >= months].index
    if not len(voters):
        return {}
    v = t[t["vintage"].isin(voters)]
    at = v.loc[v["months_on_book"] == months].groupby("vintage")["cum_bad"].sum()
    fin = v.sort_values("months_on_book").groupby("vintage")["cum_bad"].last()
    both = pd.concat([at.rename("at"), fin.rename("fin")], axis=1).dropna()
    both = both[both["fin"] > 0]
    if both.empty:
        return {}
    per = both["at"] / both["fin"]
    return {"captured": float(both["at"].sum() / both["fin"].sum()),
            "captured_median_cohort": float(per.median()),
            "captured_p25": float(per.quantile(0.25)),
            "captured_p75": float(per.quantile(0.75))}


def _capture_fixed_point(t: pd.DataFrame, share: float,
                         max_iter: int = 25) -> Tuple[Optional[int], Dict[str, int]]:
    """The age K that is the median capture age of the cohorts at least K deep.

    Pooling cohorts cannot answer this. The set contributing at each age keeps
    changing -- 120 cohorts at age 1, one at age 122 -- so the pooled `final` is
    whatever the single deepest cohort says, and "75% of that" lands just short
    of wherever the panel stops. Measured that way the window came back at
    0.87, 0.90, 0.96 and 1.00 times the horizon across two time axes and a
    doubling of the panel: it was reporting the depth of the data, not the book.

    Reading each cohort on its own curve removes the pooling entirely. The depth
    filter is self-consistent rather than tuned: a cohort cannot testify that
    the answer is 53 months unless it has 53 months of its own. Iterating to the
    fixed point removes the last free parameter, and it converges in two to four
    passes.
    """
    k = 1
    ages: Dict[str, int] = {}
    for _ in range(max_iter):
        ages = _cohort_capture_ages(t, share, k)
        if len(ages) < 2:
            return None, {}
        nxt = int(np.median(list(ages.values())))
        if nxt == k:
            break
        k = nxt
    return k, _cohort_capture_ages(t, share, k)


def suggest_window(vintage_table: pd.DataFrame) -> Tuple[dict, pd.DataFrame]:
    """The performance window recommendation and the pooled curve. For each capture target, the fixed-point age by
    which that share of a cohort's own bads have arrived, with its spread over cohorts, the capture it implies
    and, for comparison, the age read off the pooled curve, kept so the cost of pooling is visible.
    RECOMMENDED_CAPTURE picks the suggested window; a window over CROWDING_SHARE of the horizon is flagged as
    set by the panel rather than the book."""
    empty = {"suggested_window_months": None, "window_options": [],
             **{f"window_{int(100 * s)}pct_capture_months": None
                for s in CAPTURE_TARGETS}}

    if vintage_table.empty or "max_observable_age" not in vintage_table.columns:
        log.warning("No vintage-age cells to read a window from.")
        return empty, pd.DataFrame()

    pooled = _pool(vintage_table)
    horizon = int(pooled["months_on_book"].max())
    observable = (vintage_table.groupby("vintage")["max_observable_age"].max()
                  .sort_values(ascending=False))
    log.info("Full horizon: %s cohorts over ages 1..%s (%s cohort-age cells). "
             "The deepest age is carried by %s cohort(s).",
             len(observable), horizon, f"{len(vintage_table):,}",
             int(pooled["cohorts"].iloc[-1]))

    if (pooled["cum_bad_rate"].diff().dropna() < -1e-9).any():
        log.info("The pooled curve dips in places. On the full horizon that is "
                 "expected rather than alarming -- it is the cohort mix changing "
                 "with age. Read the `cohorts` column against the ages where it "
                 "happens.")

    pooled_bads = int(pooled["cum_bad"].iloc[-1]) if len(pooled) else 0
    final = float(pooled["cum_bad_rate"].iloc[-1]) if len(pooled) else float("nan")
    if pooled_bads <= 0:
        log.warning("The panel contains no bads at all, so no window can be read "
                    "from it -- there is no signal to capture a share of. Check "
                    "the analysed population: if loans in a workout are held out "
                    "and the book's deterioration runs through workouts, this is "
                    "what that looks like.")
        return dict(empty, full_horizon_months=horizon, pooled_bads_at_horizon=0,
                    pooled_bad_rate_at_horizon=0.0), pooled

    def _options(curve: pd.DataFrame, total: float, hz: int) -> list:
        out = []
        for share in CAPTURE_TARGETS:
            months, ages = _capture_fixed_point(vintage_table, share)
            v = np.array(sorted(ages.values())) if ages else np.array([])
            at = curve.loc[curve["months_on_book"] == months] if months else None
            out.append({
                "basis": f"{share:.0%} capture", "share": share, "months": months,
                "recommended": share == RECOMMENDED_CAPTURE,
                "criterion": f"the age by which {share:.0%} of a cohort's OWN bads "
                             f"have arrived, as the median over the "
                             f"{len(v)} cohorts at least that deep -- a "
                             f"self-consistent fixed point, not a share of a "
                             f"pooled total (horizon {hz} months)",
                "cohorts_voting": int(len(v)),
                "p25": int(np.percentile(v, 25)) if len(v) else None,
                "p75": int(np.percentile(v, 75)) if len(v) else None,
                "spread": [int(v.min()), int(v.max())] if len(v) else None,
                "pooled_months": _capture_window(curve, total, share),
                **_captured_by_cohort(vintage_table, months),
                "captured_pooled": _captured_at(curve, total, months),
                "cum_bad_rate": _rate_at(curve, months),
                "cohorts_at_window": (int(at["cohorts"].iloc[0])
                                      if at is not None and len(at) else None)})
        return out

    options = _options(pooled, final, horizon)
    pick = next((o for o in options if o["recommended"]), None)
    chosen = pick["months"] if pick else None

    result = {
        "suggested_window_months": chosen,
        "recommended_capture": RECOMMENDED_CAPTURE,
        "criterion": pick["criterion"] if pick else "no capture window could be read",
        "full_horizon_months": horizon,
        "cohorts": len(observable),
        "cohorts_at_horizon": int(pooled["cohorts"].iloc[-1]),
        "cohorts_at_window": pick["cohorts_at_window"] if pick else None,
        "pooled_bad_rate_at_horizon": round(final, 6),
        "pooled_bads_at_horizon": pooled_bads,
        "share_of_horizon_bads_captured": _round(pick["captured"]) if pick else None,
        "final_marginal_bps": _round(float(pooled["marginal_bps"].iloc[-1])
                                     if len(pooled) > 1 else float("nan"), 3),
        "window_options": [
            {k: (_round(v) if isinstance(v, float) else v) for k, v in o.items()}
            for o in options],
    }
    for share, o in zip(CAPTURE_TARGETS, options):
        result[f"window_{int(100 * share)}pct_capture_months"] = o["months"]

    crowding = bool(chosen and chosen >= CROWDING_SHARE * horizon)
    result["window_crowds_horizon"] = crowding

    if chosen:
        log.info("Suggested performance window: %s months -- the age by which "
                 "%.0f%% of a cohort's own bads have arrived, median over %s "
                 "cohorts at least that deep (IQR %s-%s, range %s). The pooled "
                 "reading would have said %s.", chosen,
                 100 * RECOMMENDED_CAPTURE, pick["cohorts_voting"],
                 pick["p25"], pick["p75"], pick["spread"], pick["pooled_months"])
        if crowding:
            log.warning("That window is %.0f%% of the %s-month horizon, so the "
                        "panel is probably what set it rather than the book. "
                        "Season the cohorts further before defending it.",
                        100 * chosen / horizon, horizon)
    else:
        log.warning("No window reached %.0f%% capture inside the panel.",
                    100 * RECOMMENDED_CAPTURE)
    for o in options:
        if o["months"] and not o["recommended"]:
            log.info("  %s: %s months (pooled bad rate %.3f%%, %s cohorts)",
                     o["basis"], o["months"], 100 * (o["cum_bad_rate"] or 0),
                     o["cohorts_at_window"])
    log.info("  The curve was still climbing %.2f bps/month at the horizon, so "
             "these shares are shares of a total that is not final.",
             result["final_marginal_bps"] or 0)
    return result, pooled


def _short(path: Path) -> str:
    """path relative to the working directory when it is under it."""
    try:
        return str(path.resolve().relative_to(Path.cwd()))
    except ValueError:
        return str(path)

SEED_DPD = 3

def resolve_bad_dpd(explicit: Optional[int], roll_rate_json: Path,
                    stage: str = "final") -> Tuple[int, str]:
    """The DPD threshold the curves count as bad and where it came from: --bad-dpd, the 90+ DPD seed at the seed
    stage, else the roll-rate recommendation, falling back to 90+ DPD."""
    if explicit is not None:
        return explicit, f"--bad-dpd {explicit} (explicit override)"
    if stage == "seed":
        log.info("Seed run: counting a loan bad at %s+ DPD (status %s). This is a "
                 "SEED, it exists to measure a window for the "
                 "roll-rate analysis to split.", 30 * SEED_DPD, SEED_DPD)
        return SEED_DPD, (f"the conventional {30 * SEED_DPD}+ DPD, held fixed as a "
                          f"SEED so that a window can be measured for the roll-rate "
                          f"analysis to split")
    if roll_rate_json.exists():
        try:
            rec = json.loads(roll_rate_json.read_text())
        except json.JSONDecodeError:
            rec = {}
        threshold = rec.get("dpd_threshold")
        if threshold:
            note = "" if rec.get("conclusive") else " -- flagged INCONCLUSIVE there"
            log.info("Bad definition from the roll-rate analysis: %s+ DPD (status "
                     "%s)%s", 30 * int(threshold), threshold, note)
            return int(threshold), f"roll-rate analysis ({_short(roll_rate_json)}){note}"
        log.warning("The roll-rate analysis ran but set no threshold: %s. The "
                   "curves below fall back to the conventional 90+ DPD.", rec.get("reason", "no reason recorded"))
        return SEED_DPD, (f"convention (90+ DPD) -- the roll-rate analysis ran but "
                          f"set no threshold: {rec.get('reason', 'no reason recorded')}")
    log.warning("No roll-rate recommendation at %s, so the curves below use the "
                "conventional 90+ DPD.", roll_rate_json)
    return SEED_DPD, "convention (90+ DPD) -- the roll-rate analysis has not been run"

def main() -> None:
    p = argparse.ArgumentParser(description="Vintage analysis")
    p.add_argument("--stage", choices=["seed", "final"], default="final",
                   help="'seed' is STEP 1: counts a loan bad at the conventional "
                        "90+ DPD and measures a window for the roll-rate analysis "
                        "to split in half. 'final' is STEP 4: counts a loan bad at "
                        "whatever threshold the roll rates chose and re-measures "
                        "the window against it -- that is the one the label uses "
                        "(default: %(default)s)")
    p.add_argument("--performance", default=curated_path("performance"))
    p.add_argument("--origination", default=curated_path("origination"),
                   help="origination table, read for first_payment_date to "
                        "anchor the origination-quarter cohorts")
    p.add_argument("--cohort-freq", choices=["M", "Q"], default="M")
    p.add_argument("--no-origination", action="store_true",
                   help="skip the origination-quarter vintage reported "
                        "alongside the observation-month one")
    p.add_argument("--bad-dpd", type=int)
    p.add_argument("--roll-rate-json",
                   default=str(settings.data_curated / "roll_rate"
                               / "recommended_definition.json"))
    p.add_argument("--keep-workouts", action="store_true")
    p.add_argument("--bad-includes-workout", action="store_true",
                   help="count a workout month as BAD, not just 90+ DPD. "
                        "Matches what the warehouse build labels, and is "
                        "the only way to see the modified loans that never "
                        "register 90+ DPD because the arrears were "
                        "capitalised. Use with --keep-workouts.")
    p.add_argument("--keep-disaster", action="store_true")
    p.add_argument("--out",
                   help="markdown report to write (default: "
                        "docs/vintage_analysis.md, or "
                        "docs/vintage_analysis_seed.md at --stage seed)")
    p.add_argument("--csv-dir",
                   help="where the curve data goes, for charting (default: "
                        "data/curated/vintage, or data/curated/vintage_seed at "
                        "--stage seed)")
    add_period_arguments(p)
    a = p.parse_args()

    period = period_from_args(a)

    seed = a.stage == "seed"
    if a.out is None:
        a.out = "docs/vintage_analysis_seed.md" if seed else "docs/vintage_analysis.md"
    if a.csv_dir is None:
        a.csv_dir = str(settings.data_curated / ("vintage_seed" if seed else "vintage"))

    settings.ensure_dirs()
    bad_dpd, bad_dpd_source = resolve_bad_dpd(a.bad_dpd, Path(a.roll_rate_json), a.stage)

    spark = spark_session("vintage")
    try:
        perf, period_stats = read_panel_spark(spark, a.performance, period)
        perf, population = restrict(perf, not a.keep_workouts, not a.keep_disaster)
        observed = perf.agg(F.max("period")).collect()[0][0]
        table, stats = build_vintage_table(perf=perf, asof=pd.Timestamp(observed), bad_dpd=bad_dpd, freq=a.cohort_freq,
                                           period=period,
                                           bad_includes_workout=a.bad_includes_workout)
        if a.no_origination:
            orig_table, orig_stats = pd.DataFrame(), {}
        else:
            orig = read_curated(spark, resolve_path(a.origination))
            orig_table, orig_stats = build_origination_table(
                perf=perf, orig=orig, asof=pd.Timestamp(observed),
                bad_dpd=bad_dpd, period=period,
                bad_includes_workout=a.bad_includes_workout)
    finally:
        spark.stop()
    stats["bad_dpd_source"] = bad_dpd_source
    stats["bad_includes_workout"] = bool(a.bad_includes_workout)
    stats["population_workouts_kept"] = bool(a.keep_workouts)
    if a.bad_includes_workout and not a.keep_workouts:
        log.warning("--bad-includes-workout with workouts EXCLUDED from the "
                    "population: every loan the definition would catch has "
                    "already been dropped, so the flag does nothing. Add "
                    "--keep-workouts.")
    stats["analysed_population"] = population
    stats.update(period_stats)

    suggestion, pooled = suggest_window(table)

    csv_dir = Path(a.csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_dir / "vintage_curves.csv", index=False)

    orig_suggestion: Dict[str, object] = {}
    orig_pooled = pd.DataFrame()
    if len(orig_table):
        orig_table.to_csv(csv_dir / "vintage_curves_origination.csv", index=False)
        orig_suggestion, orig_pooled = suggest_window(orig_table)

    suggestion.update(stats)
    if orig_suggestion:
        suggestion["origination_vintage"] = dict(orig_suggestion, **orig_stats)
    suggestion["stage"] = a.stage
    suggestion["bad_dpd"] = bad_dpd
    suggestion["source"] = "src/jobs/vintage_main.py"
    (csv_dir / "suggested_window.json").write_text(json.dumps(suggestion, indent=2))

    pivot = (table.pivot(index="months_on_book", columns="vintage",
                         values="cum_bad_rate") * 100).round(3)
    pivot.index.name = "months on book"

    chosen = suggestion.get("suggested_window_months")
    horizon = suggestion.get("full_horizon_months")
    captured = suggestion.get("share_of_horizon_bads_captured")

    seed_json = Path(settings.data_curated / "vintage_seed" / "suggested_window.json")
    seed_window = None
    if not seed and seed_json.exists():
        try:
            seed_window = json.loads(seed_json.read_text()).get("suggested_window_months")
        except json.JSONDecodeError:
            seed_window = None

    if chosen:
        verdict = (f"**Suggested performance window: {chosen} months.** It is the "
                   f"age by which "
                   f"{100 * suggestion.get('recommended_capture', 0):.0f}% of the "
                   f"bads visible anywhere in the panel have accumulated, over "
                   f"the full {horizon}-month horizon. It captures "
                   f"{100 * (captured or 0):.1f}% of them, and "
                   f"{suggestion.get('cohorts_at_window')} of "
                   f"{suggestion.get('cohorts')} cohorts are still contributing "
                   f"at that age.")
    else:
        verdict = (f"**No window could be read.** The panel holds no bads by month "
                   f"{horizon}, so there is no signal to capture a share of.")

    opts = pd.DataFrame(suggestion.get("window_options") or [])
    if len(opts):
        opts = opts.assign(**{
            "window (months)": opts["months"],
            "captured (%)": opts["captured"].astype(float) * 100,
            "median cohort (%)": opts["captured_median_cohort"].astype(float) * 100,
            "pooled curve would say (months)": opts["pooled_months"],
        })[["basis", "window (months)", "captured (%)", "median cohort (%)",
            "pooled curve would say (months)", "cohorts_at_window"]]
        opts = opts.rename(columns={"cohorts_at_window": "cohorts"})
    climbing = suggestion.get("final_marginal_bps") or 0
    options_md = (
            "`captured (%)` is measured the same way the window was chosen: "
            "against each cohort's OWN eventual bads, pooled over the cohorts "
            "deep enough to have reached that age. `median cohort (%)` is the "
            "middle cohort of those, and it lands on the basis by construction "
            "-- that is the fixed point checking itself.\n\n"
            "`pooled curve would say` is the same capture read off the pooled "
            "curve instead. It runs far longer, because the set of cohorts "
            "contributing changes with age, so a share of the pooled total "
            "tracks the edge of the panel rather than the book. That gap is why "
            "the window is not read that way.\n\n"
            + (md_table(opts, floatfmt="{:.1f}") if len(opts)
               else "_(no curve available)_")
            + "\n\n- **75% capture** is the recommendation.\n"
              "- **50% capture** is the shorter option. "
              "Take it when the cost of the longer window is the binding constraint\n\n"
            + (f"\n> **The recommendation is crowding the horizon.** {chosen} months "
               f"is {100 * chosen / horizon:.0f}% of the {horizon}-month panel, so "
               f"this number says more about how deep the data goes than about how "
               f"long the book takes to go bad -- a longer window could not have "
               f"been reported even if it were the right one. Season the cohorts "
               f"further before defending it.\n"
               if suggestion.get("window_crowds_horizon") else ""))

    if seed:
        stage_md = (
            "1. **This run.** Count a loan bad at the conventional 90+ DPD and "
            "measure how long the cumulative bad rate takes to stop climbing.\n"
            "2. **Split the window in half.** The first half is what you could "
            "have seen when you scored the loan; the second is the outcome you "
            "were predicting.\n"
            "3. **`make rollrate`** compares the worst rung a loan reaches in the "
            "first half with the worst it reaches in the second, and reads the "
            "real bad threshold off that transition.\n"
            "4. **`make vintage`** re-measures this curve at that threshold. "
            "*That* window goes into the label.\n\n")
    else:
        fixed_point = ""
        if seed_window and chosen and int(seed_window) != int(chosen):
            fixed_point = (
                f"\n\n> **The window moved.** Step 1 measured {seed_window} months "
                f"at 90+ DPD and the roll rates were read off halves cut out of "
                f"that ({int(seed_window) // 2} observed + "
                f"{int(seed_window) - int(seed_window) // 2} predicted). At "
                f"{30 * bad_dpd}+ DPD the window is {chosen} months, so those are "
                f"not the halves this definition implies.")
        elif seed_window and chosen:
            fixed_point = (
                f"\n\n> **The window did not move.** Step 1 measured "
                f"{seed_window} months at 90+ DPD and this run measures the same "
                f"{chosen} at {30 * bad_dpd}+ DPD, so the halves the roll rates "
                f"were read off are the halves this definition implies. The pair "
                f"(threshold, window) is a fixed point.")
        from_rollrate = bad_dpd_source.startswith("roll-rate analysis")
        step3 = ("3. `make rollrate` compared the worst rung reached in each half "
                 f"and concluded that bad means **{30 * bad_dpd}+ DPD**.\n"
                 if from_rollrate else
                 "3. `make rollrate` **did not produce a threshold** "
                 f"({bad_dpd_source}), so the curve below still counts a loan bad "
                 f"at the conventional {30 * bad_dpd}+ DPD.\n")
        step4 = ("4. **This run** re-measures the curve at that threshold. The "
                 "window in *The decision* above is measured against the "
                 "definition the label actually uses, which is what makes it "
                 "quotable."
                 if from_rollrate else
                 "4. **This run** therefore repeats step 1 rather than completing "
                 "the argument. Both halves of the label are convention until the "
                 "roll-rate analysis produces a threshold -- fix that first, and "
                 "do not quote the window below as evidence.")
        stage_md = (
                ("**This is STEP 4 of four, and this is the window the label uses.**"
                 if from_rollrate else
                 "**This is STEP 4 of four, and the argument did not complete.**")
                + "\n\n"
                  "1. `make vintage-seed` measured a window at the conventional 90+ DPD "
                  "-- see `docs/vintage_analysis_seed.md`.\n"
                  "2. That window was split into an observation half and an outcome "
                  "half.\n"
                + step3 + step4 + fixed_point)

    cohort_desc = (f"the reporting month ({stats.get('cohort_freq', 'monthly')}) ")
    criteria = pd.DataFrame([
        {"setting": "stage",
         "value": ("seed (step 1) -- measures the window the roll rates split"
                   if seed else
                   "final (step 4) -- measures the window the label uses")},
        {"setting": "cohort", "value": cohort_desc},
        {"setting": "age axis",
         "value": ("months watched forward from the cohort month")},
        {"setting": "bad definition",
         "value": f"delinquency status >= {bad_dpd:02d} ({30 * bad_dpd}+ DPD)"},
        {"setting": "bad definition from", "value": stats.get("bad_dpd_source", "--")},
        {"setting": "analysis period (cohort months)",
         "value": f"{period.label} ({period.months} months, inclusive)"},
        {"setting": "as-of date",
         "value": (f"{pd.Timestamp(observed).date()} -- the panel's last month, "
                   f"{period.lookforward_months(pd.Timestamp(observed).date())} "
                   f"month(s) past the last cohort. Cohorts are watched forward "
                   f"into it.")},
        {"setting": "horizon",
         "value": ("no configured maximum -- a loan is followed until it "
                   "vanishes from the data, so each cohort runs to the as-of "
                   f"date ({int(pivot.index.max())} months for the oldest)"
                   if len(pivot.index) else "no cohorts")},
        {"setting": "horizon read over",
         "value": (f"the full {horizon} months -- every cohort at every age it "
                   f"reached ({suggestion.get('cohorts')} cohorts; "
                   f"{suggestion.get('cohorts_at_horizon')} still contributing at "
                   f"the deepest age)")},
        {"setting": "window criterion", "value": suggestion.get("criterion", "--")},
        {"setting": "still climbing at the horizon",
         "value": f"{suggestion.get('final_marginal_bps') or 0:.2f} bps in the "
                  f"final month"},
        {"setting": "pooled bad rate at horizon",
         "value": f"{100 * (suggestion.get('pooled_bad_rate_at_horizon') or 0):.3f}%"},
    ])
    ages = [x for x in (3, 6, 9, 12, 18, 24, 36, 48, 60, 72, 84, 96, 120, 180,
                        240, 300) if x in pivot.index]
    if len(orig_table) and len(orig_pooled):
        o_ages = [x for x in ages if x <= int(orig_pooled["months_on_book"].max())]
        o_pivot = (orig_table.pivot(index="months_on_book", columns="vintage",
                                    values="cum_bad_rate") * 100).round(3)
        o_pivot.index.name = "months on book"

        side = (pooled[["months_on_book", "cohorts", "at_risk", "cum_bad_rate"]]
                .rename(columns={"cohorts": "cohorts (obs)", "at_risk": "at_risk (obs)",
                                 "cum_bad_rate": "rate (obs) %"})
                .merge(orig_pooled[["months_on_book", "cohorts", "at_risk", "cum_bad_rate"]]
                       .rename(columns={"cohorts": "cohorts (orig)",
                                        "at_risk": "loans (orig)",
                                        "cum_bad_rate": "rate (orig) %"}),
                       on="months_on_book", how="outer"))
        side["rate (obs) %"] *= 100
        side["rate (orig) %"] *= 100
        side = side[side["months_on_book"].isin(o_ages)]

        o_opts = pd.DataFrame([
            {"basis": o["basis"], "months": o["months"],
             "captured": _round(o.get("captured")),
             "cohorts at window": o.get("cohorts_at_window")}
            for o in orig_suggestion.get("window_options", [])])

        orig_md = (
            "The curve above keys every loan-MONTH to the month it was observed "
            "in, so a single loan joins many cohorts and the age axis means "
            "'months since we looked'. This one keys each LOAN to the quarter it "
            "was ORIGINATED in, and the age axis is the loan's own age. It is the "
            "conventional vintage curve, and it answers 'of the loans written in "
            "2021Q3, what share had gone bad by month 18'.\n\n"
            "Two differences follow from that, and both matter when reading the "
            "two tables against each other:\n\n"
            "- **One span per loan, not per loan-month.** Events are TIME TO "
            "FIRST here. The observation-month curve is recurrent, because a "
            "cured loan genuinely rejoins the population at a fresh observation "
            "month; a vintage curve asks when a cohort first deteriorates, so a "
            "loan is counted once and stays counted.\n"
            f"- **Left truncation.** {orig_stats.get('loans_originated_before_period', 0):,} "
            f"of the {orig_stats.get('loans_in_panel', 0):,} loans in the panel "
            "were originated before the analysis period. The panel is floored at "
            "the period start, so their early ages are simply absent -- we cannot "
            "know whether a loan went bad at age 4 when age 4 is not in the data "
            "-- and keeping them would count them as having survived ages they "
            "were never observed at. They are excluded, leaving "
            f"**{orig_stats.get('origination_loans', 0):,} loans** across "
            f"**{orig_stats.get('origination_cohorts', 0)} quarterly cohorts**, of "
            f"which {orig_stats.get('origination_loans_ever_bad', 0):,} ever reach "
            f"{30 * bad_dpd}+ DPD.\n\n"
            "Origination month is derived as `period - loan_age` rather than "
            "joined from the origination table.\n\n"
            "**Window options read off the origination curve**\n\n"
            + (md_table(o_opts) if len(o_opts) else "_(none)_") + "\n\n"
            "These are reported for comparison only. The window the label uses "
            "is still the one above: the roll-rate analysis reads "
            "`suggested_window_months`, which stays the observation-month "
            "figure.\n\n"
            "**The two curves side by side**\n\n"
            + md_table(side.round(3)) + "\n\n"
            "`at_risk (obs)` counts loan-months and `loans (orig)` counts loans, "
            "so the levels are not comparable -- the shapes are.\n\n"
            "**Cumulative bad rate by origination quarter, selected ages (%)**\n\n"
            + md_table(o_pivot.loc[[x for x in o_ages if x in o_pivot.index]],
                       index=True, floatfmt="{:.3f}"))
    else:
        orig_md = ("_Not computed -- `--no-origination` was passed, or no loan in "
                   "the panel was originated inside the analysis period._")

    sections = [
        ("The decision", verdict + "\n\n" + md_table(criteria)),
        ("Where this sits in the argument", stage_md),
        ("Window options", options_md),
        ("Pooled curve over the full horizon",
         "This is the curve the window is read off: every cohort at every age "
         "it actually reached. `at_risk` is the members observed to that age, "
         "so the rate is correct at each age however ragged the panel is.\n\n"
         "**Read `cohorts` before trusting the tail.** It is how many cohorts "
         "are still contributing at that age, and it falls away as the age "
         "grows -- the deepest ages are carried by the oldest cohorts alone, so "
         "the curve there mixes seasoning with cohort composition.\n\n"
         "`marginal_bps` is the increment each extra month of waiting buys you "
         "-- reported as a diagnostic, not used as a test.\n\n"
         + (md_table(pooled.assign(**{"cum_bad_rate (%)": 100 * pooled["cum_bad_rate"]})
                     [["months_on_book", "cohorts", "at_risk", "cum_bad",
                       "cum_bad_rate (%)", "marginal_bps"]],
                     floatfmt="{:.3f}")
            if len(pooled) else "_(no curve available)_")),
        ("Cumulative bad rate by vintage, selected ages (%)",
         f"Rows are months on book, columns are 'reporting months' "
         "Blank cells are ages a cohort has not reached yet.\n\n"
         + md_table(pivot.loc[ages], index=True, floatfmt="{:.3f}")),
        ("Vintage by origination quarter", orig_md),
        ("Cumulative bad rate by vintage, every age (%)",
         md_table(pivot, index=True, floatfmt="{:.3f}")),
        ("At-risk denominators",
         "The counts behind the rates above.\n\n"
         + md_table(table.pivot(index="months_on_book", columns="vintage",
                                values="at_risk").loc[ages],
                    index=True, floatfmt="{:.0f}")),
        ("The analysis period",
         period.describe(pd.Timestamp(observed).date())
         + ("\n\n" + md_table(pd.DataFrame([
             {"loan-months read": f"inside {period.label}",
              "rows": period_stats.get("panel_rows_in_period", 0)},
             {"loan-months read": "after the period",
              "rows": period_stats.get("panel_rows_after_period", 0)},
             {"loan-months read": "total opened by this run",
              "rows": period_stats.get("panel_rows_read", 0)},
         ]), floatfmt="{:.0f}")
            if "panel_rows_read" in period_stats else "")
         + (f"\n\nMonths before the period were **never opened**. The floor is "
            f"applied as {period_stats.get('panel_floor_applied_as')}, so "
            f"{period_stats.get('panel_partitions_skipped')} of "
            f"{period_stats.get('panel_partitions_total')} monthly partitions "
            f"were left out of the read entirely."
            if "panel_partitions_total" in period_stats else "")
         + "\n\nA cohort is a month inside the period."),
        ("The analysed population", caveat(population)),
        ("What is in a cohort",
         ("A cohort is a reporting month, and a loan belongs to every month it "
           "was on the book.\n\n"
           + md_table(pd.DataFrame([
                     {"population": "loan-months in the panel",
                      "rows": stats.get("panel_rows", 0)},
                     {"population": "cohort memberships (the denominator base)",
                      "rows": stats.get("cohort_members", 0)},
                     {"population": "dropped -- already past the threshold that month",
                      "rows": stats.get("dropped_already_bad", 0)},
                     {"population": "distinct loans behind them",
                      "rows": stats.get("loans", 0)},
                 ]), floatfmt="{:.0f}")
           + "\n\nA loan already past the threshold when the month opens is not "
             "a member of that month's cohort."
             "\n\nEvents are **recurrent**: a loan that goes bad, cures and "
             "relapses rejoins the population the month it cures, and its next "
             "default is counted by every cohort that opened while it was "
             "performing. Only the months it is actually bad are excluded.")),
        ("How to read this",
         "- The **at-risk denominator** is the part people get wrong. Recent "
         "cohorts have not had time to season, so a naive share-of-cohort "
         "understates their bad rate and flattens the curve. Here the "
         "denominator at age *t* is the loans actually observed to age *t*, "
         "which is why young vintages stop rather than bend down.\n"
         "- **Differences in shape between cohorts are the finding**, not "
         "noise. They are the argument for out-of-time validation: a model "
         "trained on one credit environment and validated randomly across all "
         "of them will look better than it is.\n"
         "- The window is a **decision, not an output**. This file is the "
         "evidence for whichever number you put in the label spec's window; "
         "if you override it, say why here.\n"
         + ("- **The threshold is not decided here, and 90+ is a seed.** These "
            "curves count a loan bad at 90+ DPD so that a window exists for the "
            "roll-rate analysis to split; that analysis decides what bad means, "
            "and step 4 re-measures this curve against its answer. Do not quote "
            "the window below as the label's window."
            if seed else
            "- **The threshold is not decided here.** These curves count a loan "
            f"bad at {30 * bad_dpd}+ DPD because that is what the roll-rate "
            "analysis concluded; this module measures how long to watch, not "
            "what to watch for.")),
    ]

    out = write_report(
        Path(a.out),
        "Vintage Analysis -- Seed Window (90+ DPD)" if seed else "Vintage Analysis",
        "src/jobs/vintage_main.py", sections,
        preamble=(f"Cumulative bad rate by 'reporting-month'\n\n"
                  + ("**Seed run (step 1).** Bad is the conventional 90+ DPD, held "
                     "fixed so that a window exists for the roll-rate analysis to "
                     "split in half."
                     if seed else
                     f"**Final run (step 4).** Bad is {30 * bad_dpd}+ DPD, from "
                     f"{bad_dpd_source}. "
                     + ("The window below is the one the label uses."
                        if bad_dpd_source.startswith("roll-rate analysis")
                        else "The roll-rate analysis did not set a threshold, so "
                             "this repeats the seed run and the window below is "
                             "not evidence for anything the label can quote."))))
    log.info("Wrote %s", out)
    log.info("Curve data in %s", csv_dir)


if __name__ == "__main__":
    main()