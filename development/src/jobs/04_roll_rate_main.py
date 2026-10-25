import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from helpers.config import settings
from helpers.population import caveat, restrict, workout_condition
from helpers.report import md_table, write_report
from helpers.spark_io import curated_path, spark_session
from helpers.period import add_arguments as add_period_arguments
from helpers.period import from_args as period_from_args, read_panel_spark
from helpers.period import observation_mask_spark
from helpers.layout import CREDIT_EVENT_CODES, PREPAID_CODES
from pyspark.sql import functions as F, Window, DataFrame

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s")
log = logging.getLogger("roll_rate")

DELINQUENT_STATES = ["Current", "30 DPD", "60 DPD", "90+ DPD"]
TERMINAL_STATES = ["Prepaid", "Credit event", "Other exit"]
STATE_ORDER = DELINQUENT_STATES + TERMINAL_STATES

RUNG_STATES = {0: "Current", 1: "30 DPD", 2: "60 DPD", 3: "90+ DPD"}

PREPAID = sorted(PREPAID_CODES)
CREDIT_EVENT = sorted(CREDIT_EVENT_CODES)

CANDIDATE_STATES: Dict[str, int] = {"30 DPD": 1, "60 DPD": 2, "90+ DPD": 3}

ROLL_DOMINATES = 0.75

SEED_WINDOW_JSON = settings.data_curated / "vintage_seed" / "suggested_window.json"

def _short(path: Path) -> str:
    """path relative to the working directory when it is under it."""
    try:
        return str(Path(path).resolve().relative_to(Path.cwd()))
    except ValueError:
        return str(path)

def split_window(window_months: int) -> Tuple[int, int]:
    """The observation and outcome halves of a window, in months (the outcome half takes the odd month)."""
    if window_months < 2:
        raise ValueError("Window months must be at least 2")
    first = window_months // 2
    return first, window_months - first

SEED_BASES = ("observation", "origination")


def _window_from_vintage(vt: dict, basis: str = "observation") -> Tuple[Optional[int], str]:
    """The window the seed vintage measured, on whichever time axis was asked for.

    The seed JSON carries BOTH readings. At the top level is the
    observation-month one: every loan-month is a cohort member, the age axis is
    months since that observation, and events are recurrent. Under
    `origination_vintage` is the conventional one: each loan is counted once,
    the cohort is the quarter it was originated in, the age axis is the loan's
    own age, and events are time-to-first.

    They measure the same book and do not have to agree. The observation-month
    curve pools cohorts whose membership changes with age, so its capture point
    drifts with the panel's depth; the origination curve holds each cohort
    fixed but can only use loans originated inside the analysis period, because
    anything older is left-truncated by the panel floor. Neither is the right
    answer by construction -- which is why this is a choice and not a default.
    """
    if basis not in SEED_BASES:
        raise SystemExit(f"--seed-basis must be one of {', '.join(SEED_BASES)}")
    node = vt
    if basis == "origination":
        node = vt.get("origination_vintage") or {}
        if not node:
            return None, ("the seed vintage carries no origination-quarter reading. "
                          "Re-run src/jobs/vintage_main.py --stage seed WITHOUT "
                          "--no-origination, or pass --seed-basis observation")
    months = node.get("suggested_window_months")
    if not months:
        return None, f"the seed vintage analysis produced no {basis} window at all"
    return int(months), node.get("criterion", "capture criterion not recorded")


def resolve_window(explicit: Optional[int], seed_json: Path,
                   basis: str = "observation") -> Tuple[int, str, dict]:
    """The performance window, where it came from and the seed vintage JSON: --window-months, else the seed vintage
    analysis on the chosen basis."""
    if explicit is not None:
        return explicit, f'--window-months {explicit} (explicit override)', {}
    if seed_json.exists():
        try:
            vt = json.loads(seed_json.read_text())
        except json.JSONDecodeError:
            vt = {}
        months, criterion = _window_from_vintage(vt, basis)
        if months:
            log.info("Performance window from the seed vintage analysis "
                     "(%s basis): %s months -- %s", basis, months, criterion)
            other, _ = _window_from_vintage(
                vt, "origination" if basis == "observation" else "observation")
            if other and other != months:
                log.info("For reference, the %s basis would have given %s months.",
                         "origination" if basis == "observation" else "observation",
                         other)
            return (months,
                    f"seed vintage analysis, {basis} basis "
                    f"({_short(seed_json)}): {criterion}",
                    vt)
        raise SystemExit(
            f"The seed vintage analysis at {seed_json} produced no window "
            f"on the {basis} basis: {criterion}")
    raise SystemExit(
        f"No seed window at {seed_json}.")

def _rung_state(col):
    """The delinquency rung (0-3) as a state name."""
    return (
        F.when(col == 0, F.lit('Current'))
        .when(col == 1, F.lit('30 DPD'))
        .when(col == 2, F.lit('60 DPD'))
        .when(col == 3, F.lit('90+ DPD'))
    )

def _prepare(perf, workout_as_deepest: bool = False) -> Tuple[DataFrame, int, int]:
    """The panel reduced to rung, month index and exit flags (credit event, prepaid, other exit), the last month
    index and the number of rows dropped as unreadable (no status and no zero-balance code). Workouts are read
    in the EVENT sense, the month the workout happened only, or a loan modified in 2012 would read 90+ DPD for a
    decade of performing payments. With workout_as_deepest a workout month is pushed to the deepest rung: the
    servicer capitalised the arrears and returned the status to current, so read literally a modification is a
    cure, and the deepest delinquencies are the ones most often worked out. Scoring it as 90+ DPD says the loan
    did not recover, it was rewritten, and agrees with the warehouse build, which labels a workout bad."""
    df = perf.select('loan_identifier', 'period', 'dpd', 'zero_balance_code',
                     F.when(workout_condition(perf, include_prior=False), 1)
                      .otherwise(0).alias('workout'))

    dpd = F.col('dpd')
    zbc = F.lpad(F.trim(F.col('zero_balance_code')), 2, "0")
    zbc = F.when(
        F.col('zero_balance_code').isNull() |
        (F.trim(F.col('zero_balance_code')) == ''), None
    ).otherwise(zbc)

    rung = (F.when(dpd == 0, F.lit(0)).when(dpd == 1, F.lit(1))
            .when(dpd == 2, F.lit(2)).when(dpd >= 3, F.lit(3)))
    if workout_as_deepest:
        rung = F.when(F.col('workout') == 1, F.lit(3)).otherwise(rung)
    period = F.col('period')

    df = (df.withColumn('zbc', zbc)
          .withColumn('rung', rung)
          .withColumn('month_idx', F.year(period) * 12 + F.month(period))
          .withColumn('credit_event',
                      F.when(F.col('zbc').isin(CREDIT_EVENT), 1).otherwise(0))
          .withColumn('prepaid',
                      F.when(F.col('zbc').isin(PREPAID), 1).otherwise(0))
          .withColumn('other_exit',
                      F.when(F.col('zbc').isNotNull()
                             & ~F.col('zbc').isin(CREDIT_EVENT + PREPAID), 1).otherwise(0))
          )

    unreadable = F.col('rung').isNull() & F.col('zbc').isNull()
    unknown = df.filter(unreadable).count()
    df = df.filter(~unreadable)

    panel_max = df.agg(F.max('month_idx')).collect()[0][0]
    return df, int(panel_max or 0), unknown

def build_window_transitions(perf, windows: Tuple[int, ...], period=None,
                             workout_as_deepest: bool = False):
    """One (from_state, to_state) pair per anchor month and window: the worst rung in the observation half against
    the worst rung or exit in the outcome half, with the counts kept, censored and gapped. Only anchors whose
    whole window closes inside the panel are kept. Exempting loans that reached an absorbing state would admit
    nearly every refinance, which resolves in one month, and only the defaults that had already run to REO,
    while dropping a loan sitting at 90+ DPD, so later anchors would enter conditioned on their own outcome. An
    anchor whose outcome half holds no rows (it exited inside the observation half) takes its outcome from the
    terminal flags over the whole window and is flagged obs_half_exit so the matrix can be read without it; at
    24 months that is 9.2% of all pairs and 32% of the 90+ DPD row."""
    df, panel_max, unknown = _prepare(perf, workout_as_deepest)

    frames, excluded = [], {}
    for w in windows:
        obs_months, out_months = split_window(w)
        part = Window.partitionBy('loan_identifier').orderBy('month_idx')
        first = part.rangeBetween(1, obs_months)
        second = part.rangeBetween(obs_months + 1, w)
        whole = part.rangeBetween(1, w)

        step = (df
                .withColumn('n_first', F.count(F.lit(1)).over(first))
                .withColumn('n_second', F.count(F.lit(1)).over(second))
                .withColumn('rung_first', F.max('rung').over(first))
                .withColumn('rung_second', F.max('rung').over(second))
                .withColumn('ce', F.max('credit_event').over(whole))
                .withColumn('pp', F.max('prepaid').over(whole))
                .withColumn('oe', F.max('other_exit').over(whole))
                )

        terminated = (F.col('ce') + F.col('pp') + F.col('oe')) > 0

        mature = (F.col('month_idx') + F.lit(w)) <= F.lit(panel_max)

        to_state = (F.when(F.col('ce') == 1, F.lit('Credit event'))
                    .when(F.col('rung_second') >= 1, _rung_state(F.col('rung_second')))
                    .when(F.col('pp') == 1, F.lit('Prepaid'))
                    .when(F.col('oe') == 1, F.lit('Other exit'))
                    .when(F.col('n_second') > 0, F.lit('Current')))

        anchor = F.col('zbc').isNull()
        if period is not None:
            anchor = anchor & observation_mask_spark(period)
        step = (step
                .filter(anchor)
                .withColumn('from_state', _rung_state(F.col('rung_first')))
                .withColumn('to_state', to_state)
                .withColumn('mature', mature)
        )
        usable = F.col('from_state').isNotNull() & F.col('to_state').isNotNull()

        counts = step.agg(
            F.count(F.lit(1)).alias('anchors'),
            F.sum(F.when(F.col('mature') & usable, 1).otherwise(0)).alias('kept'),
            F.sum(F.when(~F.col('mature'), 1).otherwise(0)).alias('censored'),
            F.sum(F.when(F.col('mature') & ~usable,1).otherwise(0)).alias('gapped')
        ).collect()[0]
        excluded[w] = {"kept": int(counts["kept"]), "censored": int(counts["censored"]),
                       "gapped": int(counts["gapped"]), "anchors": int(counts["anchors"]),
                       "unknown_status": unknown,
                       "observation_half_months": obs_months,
                       "outcome_half_months": out_months}

        log.info("%2s-month window (%s observed + %s predicted): %s transitions kept "
                 "| %s censored (window had not closed), %s with nothing observed "
                 "in a half", w, obs_months, out_months, f"{counts['kept']:,}",
                 f"{counts['censored']:,}", f"{counts['gapped']:,}")

        frames.append(step.filter(F.col("mature") & usable)
                      .withColumn("window_months", F.lit(w))
                      .withColumn("year", F.year("period"))
                      .withColumn("obs_half_exit", F.col("n_second") == 0)
                      .select("loan_identifier", "period", "year",
                              "window_months", "from_state", "to_state",
                              "obs_half_exit"))

    pairs = frames[0]
    for f in frames[1:]:
        pairs = pairs.unionByName(f)
    if unknown:
        log.info("%s rows have an unparseable status and no zero balance code",
                 f"{unknown:,}")
    return pairs, excluded

def transition_counts(pairs) -> pd.DataFrame:
    """Every count the report needs, in ONE pass over `pairs`.

    `pairs` sits on top of window functions over the whole panel and is far too
    large to cache, so each action against it recomputes that pipeline. Grouping
    on year as well means the by-year table can be summed out of this frame
    instead of triggering a second full recomputation -- at 1.4B loan-months the
    difference is the whole job's spill budget.
    """
    return (pairs.groupBy("window_months", "year", "from_state", "to_state",
                          "obs_half_exit")
            .count().toPandas().rename(columns={"count": "n"}))


def collapse_years(counts: pd.DataFrame) -> pd.DataFrame:
    """Sum the year dimension out, for the matrices that pool over time."""
    keys = [c for c in ("window_months", "from_state", "to_state", "obs_half_exit")
            if c in counts.columns]
    return counts.groupby(keys, as_index=False)["n"].sum()

def transition_matrix(counts: pd.DataFrame, window: int,
                      drop_obs_half_exits: bool = False) -> pd.DataFrame:
    """Row = worst rung in the observation half, column = where it got to.

    `drop_obs_half_exits` is the sensitivity read: it removes the anchors whose
    outcome half was empty, so every remaining pair was measured across both
    halves. The headline matrix keeps them -- they are real outcomes, just ones
    that resolved early -- and the two are reported side by side.
    """
    c = collapse_years(counts)
    c = c[c["window_months"] == window]
    if drop_obs_half_exits and "obs_half_exit" in c.columns:
        c = c[~c["obs_half_exit"].astype(bool)]
    if c.empty:
        return pd.DataFrame(index=[], columns=STATE_ORDER, dtype=int)
    m = (c.pivot_table(index="from_state", columns="to_state", values="n",
                       aggfunc="sum")
         .reindex(index=DELINQUENT_STATES, columns=STATE_ORDER).fillna(0).astype(int))
    return m.loc[[s for s in DELINQUENT_STATES if m.loc[s].sum() > 0]]

def as_rates(matrix: pd.DataFrame) -> pd.DataFrame:
    """A count matrix as row percentages."""
    totals = matrix.sum(axis=1).replace(0, pd.NA)
    return (100 * matrix.div(totals, axis=0)).astype(float)

def rates_by_year_from_counts(counts: pd.DataFrame, moves: List[tuple],
                             window: int) -> pd.DataFrame:
    """The by-year table, summed out of `counts` rather than recollected."""
    c = counts[(counts["window_months"] == window)
               & counts["from_state"].isin([m[0] for m in moves])]
    rows = []
    for year, block in c.groupby("year"):
        row = {"year": str(int(year))}
        for frm, to in moves:
            b = block[block["from_state"] == frm]
            tot = float(b["n"].sum())
            hit = float(b.loc[b["to_state"] == to, "n"].sum())
            row[f"{frm} -> {to}"] = (100 * hit / tot) if tot else None
            row[f"n({frm})"] = int(tot)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("year")
    ordered = ["year"] + [c2 for m in moves for c2 in (f"{m[0]} -> {m[1]}", f"n({m[0]})")]
    return out[[c2 for c2 in ordered if c2 in out.columns]]


def rates_by_year(pairs, moves: List[tuple], window: int) -> pd.DataFrame:
    """Per year, the share of loans from each move's origin state that reached its destination, read straight from
    the pairs."""
    wanted = (pairs.filter(F.col("window_months") == window)
              .filter(F.col("from_state").isin([m[0] for m in moves])))
    agg = (wanted.groupBy("year", "from_state")
           .agg(F.count("*").alias("n"),
                *[F.sum(F.when(F.col("to_state") == to, 1).otherwise(0)).alias(f"to_{i}")
                  for i, (_, to) in enumerate(moves)])
           .toPandas())

    rows = []
    for year, block in agg.groupby("year"):
        row = {"year": str(int(year))}
        for i, (frm, to) in enumerate(moves):
            b = block[block["from_state"] == frm]
            if len(b) and b["n"].iloc[0]:
                row[f"{frm} -> {to}"] = 100 * b[f"to_{i}"].iloc[0] / b["n"].iloc[0]
                row[f"n({frm})"] = int(b["n"].iloc[0])
            else:
                row[f"{frm} -> {to}"] = None
                row[f"n({frm})"] = 0
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("year")
    ordered = ["year"] + [c for m in moves for c in (f"{m[0]} -> {m[1]}", f"n({m[0]})")]
    return out[[c for c in ordered if c in out.columns]]

def movement_table(matrix: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    """Per delinquent rung: cured, stayed, deteriorated and other-exit shares, and the roll share of movers.
    Staying is held out of the roll share at Current only, where it is the good outcome and carries no
    direction. At a delinquent rung staying is a failure to recover: a loan 60 DPD in the observation half and
    still 60 DPD at the end has been delinquent for the whole window, and holding it out understates how sticky
    delinquency is. This matters most at the deepest rung, where nothing is worse and the only other route to
    deterioration is a completed credit event, which foreclosure timelines (18-36 months) usually push outside
    the outcome half."""
    rows = []
    for state in DELINQUENT_STATES:
        if state not in rates.index:
            continue
        i = DELINQUENT_STATES.index(state)
        better = [c for c in DELINQUENT_STATES[:i] if c in rates.columns]
        worse = [c for c in DELINQUENT_STATES[i + 1:] if c in rates.columns]

        def total(cols: List[str]) -> float:
            cols = [c for c in cols if c in rates.columns]
            return float(rates.loc[state, cols].sum()) if cols else 0.0

        stayed = float(rates.loc[state, state]) if state in rates.columns else 0.0
        cured = total(better + ["Prepaid"])
        worsened = total(worse + ["Credit event"])

        counts_stayed = state != DELINQUENT_STATES[0]
        rolled = worsened + stayed if counts_stayed else worsened
        moved = cured + rolled
        rows.append({
            "state": state,
            "pairs": int(matrix.loc[state].sum()),
            "cured (%)": cured,
            "stayed (%)": stayed,
            "deteriorated (%)": worsened,
            "other exit (%)": total(["Other exit"]),
            "roll share of movers (%)": 100 * rolled / moved if moved else float("nan"),
            "roll share counts": "moved + stayed" if counts_stayed else "moved",
        })
    return pd.DataFrame(rows)

def _stays_count(row) -> bool:
    """Does staying at this rung count against the loan rather than being held out?"""
    return str(row.get("roll share counts", "moved")) == "moved + stayed"


def _scope(row) -> str:
    """Which loans the row's roll share is computed over, in words."""
    if _stays_count(row):
        return (f"of the loans that either moved off {row['state']} or sat at it "
                f"for the whole outcome half")
    return (f"of the loans whose worst rung in the outcome half differed from "
            f"{row['state']} in the observation half")


def recommend_definition(movement: pd.DataFrame, dominates: float = ROLL_DOMINATES) -> Tuple[dict, pd.DataFrame]:
    """The bad threshold: the shallowest candidate rung whose roll share reaches dominates. When none does, the
    deepest readable rung is returned as an inconclusive floor. Returns the recommendation and the candidate
    table."""
    cand = movement[movement['state'].isin(CANDIDATE_STATES)].copy()
    cand['dpd_threshold'] = cand['state'].map(CANDIDATE_STATES)
    cand['point_of_no_return'] = cand['roll share of movers (%)'] >= 100 * dominates
    cand = cand.sort_values('dpd_threshold', ignore_index=True)

    usable = cand[cand['roll share of movers (%)'].notna()]
    chosen = cand[cand['point_of_no_return']]

    rec: dict = {
        'rule': (f"the shallowest rung at which at least {dominates:.0%} of the "
                 f"loans FAILED TO RECOVER across the two halves of the "
                 f"performance window -- ending at a worse rung, at a credit "
                 f"event, or still at the same delinquent rung -- rather than "
                 f"recovering to a better rung or prepaying. Staying put counts "
                 f"against the loan at every delinquent rung, because a loan that "
                 f"sat at 60 DPD for the whole window did not cure; only at "
                 f"`Current`, where staying is the good outcome and carries no "
                 f"direction, is it held out"),
        'roll_dominates': float(dominates),
        'evidence': cand.round(4).to_dict("records"),
    }

    if not chosen.empty:
        row = chosen.iloc[0]
        rec.update({
            "conclusive": True,
            "state": str(row["state"]),
            "dpd_threshold": int(row["dpd_threshold"]),
            "roll_share_of_movers": round(float(row["roll share of movers (%)"]) / 100, 4),
            "reason": (f"{_scope(row)}, "
                       f"{row['roll share of movers (%)']:.1f}% "
                       f"{'failed to recover' if _stays_count(row) else 'were worse rather than better'}"
                       f" -- the shallowest rung where deterioration outweighs "
                       f"cure, and therefore the earliest point that is a real "
                       f"default rather than a missed payment most borrowers recover "
                       f"from"),
        })
    elif not usable.empty:
        row = usable.iloc[-1]
        rec.update({
            "conclusive": False,
            "state": str(row["state"]),
            "dpd_threshold": int(row["dpd_threshold"]),
            "roll_share_of_movers": round(float(row["roll share of movers (%)"]) / 100, 4),
            "reason": (f"INCONCLUSIVE -- no rung reached {dominates:.0%} deterioration "
                       f"among its movers, so on this book every measured rung still "
                       f"cures more often than it rolls across the window. Quoted is "
                       f"the deepest readable rung ({row['state']}, "
                       f"{row['roll share of movers (%)']:.1f}%), which is a floor on "
                       f"the threshold rather than a recommendation. A book observed "
                       f"over a benign window will do this"),
        })
    else:
        rec.update({
            "conclusive": False, "state": None, "dpd_threshold": None,
            "roll_share_of_movers": None,
            "reason": "no candidate rung has any loans moving out of it",
        })
    return rec, cand

def roll_share_by_window(movements: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """The roll share of each rung for every window length, side by side."""
    rows = []
    for state in DELINQUENT_STATES:
        row = {"state": state}
        for w, mv in sorted(movements.items()):
            hit = mv[mv["state"] == state]
            row[f"{w}m (%)"] = (float(hit["roll share of movers (%)"].iloc[0])
                                if len(hit) else None)
            row[f"n({w}m)"] = int(hit["pairs"].iloc[0]) if len(hit) else 0
        rows.append(row)
    return pd.DataFrame(rows)

def main() -> None:
    p = argparse.ArgumentParser(
        description="Roll-rate analysis"
    )
    p.add_argument("--performance", default=curated_path('performance'))
    p.add_argument('--out', default="docs/roll_rate_analysis.md")
    p.add_argument('--csv-dir', default=str(settings.data_curated / 'roll_rate'))
    p.add_argument('--window-months', type=int)
    p.add_argument('--seed-window-json', default=str(SEED_WINDOW_JSON))
    p.add_argument('--seed-basis', choices=list(SEED_BASES), default='observation',
                   help='which of the seed vintage\'s two readings sets the '
                        'performance window: the observation-month curve '
                        '(default) or the origination-quarter one. Ignored '
                        'when --window-months is given')
    p.add_argument('--compare-windows', default='',
                   help="extra window lengths to report beside the chosen one, comma separated")
    p.add_argument('--roll-dominates', type=float, default=ROLL_DOMINATES)
    p.add_argument('--keep-workouts', action='store_true')
    p.add_argument('--workout-as-90plus', action='store_true',
                   help='treat a workout month as the deepest rung (90+ DPD) '
                        'regardless of its reported dpd. A modification resets '
                        'the status to 00, so without this a workout reads as a '
                        'cure at exactly the rung the threshold is read off. '
                        'Use with --keep-workouts.')
    p.add_argument('--keep-disaster', action='store_true')
    add_period_arguments(p)
    a = p.parse_args()

    period = period_from_args(a)
    window, window_source, seed = resolve_window(a.window_months, Path(a.seed_window_json), a.seed_basis)
    obs_months, out_months = split_window(window)
    windows = tuple(sorted({window} | {int(w) for w in a.compare_windows.split(',') if w.strip()}))

    for w in windows:
        split_window(w)

    spark = spark_session('roll_rate')
    try:
        perf, period_stats = read_panel_spark(spark, a.performance, period)
        perf, population = restrict(perf, not a.keep_workouts, not a.keep_disaster)
        pairs, excluded = build_window_transitions(perf, windows, period,
                                                  a.workout_as_90plus)

        counts = transition_counts(pairs)
        moves = [('Current', '30 DPD'), ('30 DPD', '60 DPD'), ('60 DPD', '90+ DPD')]
        by_year = rates_by_year_from_counts(counts, moves, window)
    finally:
        spark.stop()

    matrices = {w: transition_matrix(counts, w) for w in windows}
    rates = {w: as_rates(m) for w, m in matrices.items()}
    movements = {w: movement_table(matrices[w], rates[w]) for w in windows}
    recs = {w: recommend_definition(movements[w], a.roll_dominates) for w in windows}
    persistence = roll_share_by_window(movements)

    matrices_ex = {w: transition_matrix(counts, w, drop_obs_half_exits=True)
                   for w in windows}
    movements_ex = {w: movement_table(matrices_ex[w], as_rates(matrices_ex[w]))
                    for w in windows if len(matrices_ex[w])}
    recs_ex = {w: recommend_definition(movements_ex[w], a.roll_dominates)
               for w in windows if w in movements_ex}
    sensitivity = pd.DataFrame([
        {"window": f"{w}m",
         "pairs (all)": int(matrices[w].to_numpy().sum()),
         "pairs (both halves)": int(matrices_ex[w].to_numpy().sum()) if len(matrices_ex[w]) else 0,
         "dropped (%)": (100 * (1 - matrices_ex[w].to_numpy().sum()
                                / max(matrices[w].to_numpy().sum(), 1))
                         if len(matrices_ex[w]) else float("nan")),
         "state (all)": recs[w][0].get("state"),
         "state (both halves)": recs_ex[w][0].get("state") if w in recs_ex else None,
         "roll share (all) %": recs[w][0].get("roll_share_of_movers"),
         "roll share (both halves) %": (recs_ex[w][0].get("roll_share_of_movers")
                                        if w in recs_ex else None)}
        for w in windows])
    stable_vs_sensitivity = bool(
        len(sensitivity) and (sensitivity["state (all)"]
                              == sensitivity["state (both halves)"]).all())
    log.info("Sensitivity (empty outcome halves removed): threshold %s across all "
             "windows.", "HOLDS" if stable_vs_sensitivity else "MOVES")

    rec, candidates = recs[window]
    rec['window_months'] = window
    rec['observation_half_months'] = obs_months
    rec['outcome_half_months'] = out_months
    rec['window_source'] = window_source
    rec['seed_window_dpd_threshold'] = seed.get('bad_dpd', 3)
    rec['workout_as_90plus'] = bool(a.workout_as_90plus)
    rec['population_workouts_kept'] = bool(a.keep_workouts)
    if a.workout_as_90plus and not a.keep_workouts:
        log.warning('--workout-as-90plus with workouts EXCLUDED from the '
                    'population: every loan it would touch has already been '
                    'dropped, so the flag does nothing. Add --keep-workouts.')
    rec['windows_reported'] = list(windows)
    rec['by_window'] = {str(w): {'state': r[0].get('state'),
                                 'dpd_threshold': r[0].get('dpd_threshold'),
                                 'conclusive': r[0].get('conclusive')}
                        for w, r in sorted(recs.items())}
    agreed = {v['dpd_threshold'] for v in rec['by_window'].values()}
    rec['stable_across_windows'] = len(agreed) == 1
    rec['excluded'] = {str(w): excluded[w] for w in sorted(excluded)}
    rec['sensitivity_empty_outcome_half'] = {
        'stable': stable_vs_sensitivity,
        'note': ('the same matrices with anchors whose outcome half held no rows '
                 'removed, so every remaining pair spans both halves'),
        'by_window': {r['window']: {k: v for k, v in r.items() if k != 'window'}
                      for r in sensitivity.round(4).to_dict('records')},
        'by_window_excluded': {
            str(w): movements_ex[w].round(4).to_dict('records')
            for w in sorted(movements_ex)},
    }
    rec.update(period_stats)
    rec['analysed_population'] = population
    rec['source'] = 'src/jobs/roll_rate_main.py'
    rec['generated_at'] = datetime.now(timezone.utc).isoformat(timespec='seconds')

    csv_dir = Path(a.csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)
    counts.to_csv(csv_dir / 'transition_counts.csv', index=False)
    pd.concat([r.round(4).assign(window_months=w).reset_index().rename(columns={'index': 'from_state'})
               for w, r in sorted(rates.items())]).to_csv(csv_dir / 'transition_rates.csv', index=False)
    by_year.to_csv(csv_dir / 'roll_rates_by_year.csv', index=False)
    pd.concat([m.round(4).assign(window_months=w)
               for w, m in sorted(movements.items())]).to_csv(csv_dir / 'movement.csv', index=False)
    sensitivity.round(4).to_csv(csv_dir / 'sensitivity_empty_outcome_half.csv', index=False)
    if movements_ex:
        pd.concat([m.round(4).assign(window_months=w)
                   for w, m in sorted(movements_ex.items())]
                  ).to_csv(csv_dir / 'movement_both_halves.csv', index=False)
    rec_path = csv_dir / 'recommended_definition.json'
    rec_path.write_text(json.dumps(rec, indent=2))

    if rec.get("dpd_threshold"):
        log.info("BAD DEFINITION from the %s-month window (%s observed + %s "
                 "predicted): %s+ DPD (status %s) -- %s", window, obs_months,
                 out_months, 30 * int(rec["dpd_threshold"]), rec["dpd_threshold"],
                 "conclusive" if rec["conclusive"] else "INCONCLUSIVE")
        if not rec["stable_across_windows"]:
            log.warning("The window lengths disagree about the threshold: %s.",
                        ", ".join(f"{w}m -> {v['state']}"
                                  for w, v in rec["by_window"].items()))
    else:
        log.warning("No bad definition could be read from the roll rates: %s",rec["reason"])
    log.info("Recommendation written to %s", rec_path)
    log.info("NEXT: `make vintage` re-measures the window at %s+ DPD.",
             30 * int(rec["dpd_threshold"]) if rec.get("dpd_threshold") else "?")

    if rec.get("dpd_threshold"):
        verdict = (f"**The bad definition is {30 * int(rec['dpd_threshold'])}+ days past "
                   f"due** (Freddie delinquency status `{int(rec['dpd_threshold']):02d}`, "
                   f"the `{rec['state']}` rung).\n\n"
                   f"{rec['reason'][0].upper()}{rec['reason'][1:]}.")
        if not rec["conclusive"]:
            verdict = ("**No rung on this book is a point of no return.**\n\n"
                       + rec["reason"] + ".")
    else:
        verdict = "**The roll rates cannot set a definition.** " + rec["reason"] + "."

    ev = candidates.rename(columns={"roll share of movers (%)": "roll share (%)"})
    ev = ev[["state", "dpd_threshold", "pairs", "cured (%)", "stayed (%)",
             "deteriorated (%)", "roll share (%)", "roll share counts",
             "point_of_no_return"]]

    halves = pd.DataFrame([
        {"part": "performance window", "months": window,
         "ages covered": f"1 .. {window}",
         "what it is": "the span the label's `window_months` covers"},
        {"part": "observation half", "months": obs_months,
         "ages covered": f"1 .. {obs_months}",
         "what it is": "the FROM state is the worst rung reached in it"},
        {"part": "outcome half", "months": out_months,
         "ages covered": f"{obs_months + 1} .. {window}",
         "what it is": "the TO state is the worst rung reached in it"},
    ])

    from_seed = window_source.startswith("seed vintage analysis")
    window_md = (
        ("The window is **not decided here**. It comes from the seed vintage "
         "analysis, which measured the shortest window holding most of the bads "
         "visible anywhere in its panel, using the conventional 90+ DPD as a SEED "
         "It is a floor: the panel's total is the bads "
         "visible by its horizon, not the bads that eventually arrive."
         if from_seed else
         "**The window did not come from the seed vintage analysis.** It is "
         "supposed to: step 1 measures how long a loan has to be watched before "
         "its cumulative bad rate stops climbing, at the conventional 90+ DPD, and "
         "this module splits that answer. Everything below is still a measurement, "
         "but the halves it is cut into are not evidence-based, so the threshold "
         "inherits whatever the window inherited.")
        + f"\n\nSource: {window_source}.\n\n"
        + md_table(halves) + "\n\n"
        "The odd month, when the window is odd, goes to the **outcome** half: the "
        "thing being predicted should never have less time to happen in than the "
        "observation had, or the matrix would flatter itself.\n\n"
        "Both halves are **calendar offsets from the anchor month**, not row "
        "offsets, so a loan that skips three months of reporting contributes the "
        "months it actually reported rather than quietly shifting its halves "
        "forward. An anchor with nothing observed in a half is dropped and counted "
        "under *What was excluded*.\n\n"
        "Age 0 -- the anchor month itself -- is in neither half.")

    hz = f"{window}-month window ({obs_months} observed + {out_months} predicted)"
    stability_md = (
        ("Only one window length was computed, so there is nothing to compare. "
         "Pass `--compare-windows` with a couple of alternatives to find out "
         "whether the threshold depends on the window step 1 handed over."
         if len(windows) == 1 else
         ("Every window length picks the same rung, so the threshold does not "
          "depend on how long step 1 said to watch."
          if rec["stable_across_windows"] else
          "**The window lengths disagree**."))
        + "\n\n"
        + ", ".join(f"`{w}m` -> {v['state'] or 'none'}"
                    for w, v in rec["by_window"].items())
        + "\n\n" + md_table(persistence.round(2)) + "\n\n"
        "Read a row left to right: of the loans whose outcome half reached a "
        "different rung from that one, the share that reached a worse one. `n` is "
        "the transitions behind each figure -- they shrink as the window "
        "lengthens, because a longer window needs more months of data after the "
        "anchor before it has closed.")

    exclusion_rows = []
    for w in sorted(excluded):
        e = excluded[w]
        tot = e["kept"] + e["censored"] + e["gapped"]
        exclusion_rows.append({
            "window": f"{w}m",
            "halves": f"{e['observation_half_months']}+{e['outcome_half_months']}",
            "kept": e["kept"],
            "censored (window had not closed)": e["censored"],
            "nothing observed in a half": e["gapped"],
            "kept (%)": 100 * e["kept"] / tot if tot else None})
    unknown_status = excluded[sorted(excluded)[0]]["unknown_status"]

    sections = [
        ("The bad definition",
         verdict + "\n\n"
         "Each row is a rung reached in the **observation half**; the columns are "
         "where the loan got to in the **outcome half**. `deteriorated` folds in "
         "credit events and `cured` folds in prepayment, because a loan that "
         "leaves the book owing nothing has cured and one that leaves through a "
         "loss has not. `roll share` is the decisive column: of the loans that "
         "resolved in a direction between the halves, the share that went the "
         "wrong way. Loans whose outcome half reached the same rung are normally "
         "excluded from it -- the window told you nothing new about direction -- "
         "and `roll share counts` says where that exclusion applied.\n\n"
         "**`Current` is the only rung where staying put is held out.** There it "
         "is the good outcome and says nothing about direction. At every "
         "delinquent rung, a loan that sits at the same rung through the whole "
         "outcome half has NOT cured -- it has been delinquent for the entire "
         "window -- so it counts as the failure to recover that it is. This "
         "matters most at the deepest rung, which has nowhere worse to go: its "
         "only route into `deteriorated` is a COMPLETED credit event, and "
         "foreclosure runs 18-36 months from the delinquency, longer than most "
         "outcome halves. Held out, that rung would be scored on the one outcome "
         "the servicing calendar forbids and could never reach the threshold on "
         "any book. Every rung is eligible however few transitions sit behind it, "
         "so read `pairs` before trusting a roll share.\n\n"
         + md_table(ev.round(2)) + "\n\n"
         f"Rule applied: {rec['rule']}.\n\n"
         f"Read over the **{hz}**. The ladder stops at `90+ DPD`: deeper rungs "
         "measure a servicer's foreclosure timeline rather than a borrower's "
         "credit, and splitting them spreads the deepest rows over several thin "
         "buckets.\n\n"
         "This threshold is half of the label. The other half -- how long to "
         "watch -- is what the vintage analysis measures, and it is measured "
         "twice: once at 90+ DPD to produce the window split here, and again at "
         "this threshold to produce the window that goes into the label.\n\n"
         f"Computed over {rec.get('performance_rows', 0):,} performance rows, "
         f"{rec.get('period_min')} to {rec.get('period_max')}, anchored in the "
         f"analysis period {period.label}, release "
         f"{', '.join(rec.get('release_labels') or ['(unstamped)'])}."),
        ("The analysis period",
         md_table(pd.DataFrame([
             {"loan-months read": f"inside {period.label}",
              "rows": period_stats.get("panel_rows_in_period", 0)},
             {"loan-months read": "after the period",
              "rows": period_stats.get("panel_rows_after_period", 0)},
             {"loan-months read": "total opened by this run",
              "rows": period_stats.get("panel_rows_read", 0)},
         ]), floatfmt="{:.0f}")
         + (f"\n\nMonths before the period were **never opened**. The floor is "
            f"applied as {period_stats.get('panel_floor_applied_as')}, so "
            f"{period_stats.get('panel_partitions_skipped')} of "
            f"{period_stats.get('panel_partitions_total')} monthly partitions "
            f"were left out of the read entirely."
            if "panel_partitions_total" in period_stats else "")
         + "\n\nAn anchor must be inside the period; the two halves it opens "
           "need not be. An anchor in the last month of the period is a real "
           "observation whose outcome half falls after it, and dropping it "
           "would throw away the newest transitions in the matrix.\n\n"
           f"*Censored* in the exclusions below means an anchor whose window "
           "would close after that."),
        ("The window and its two halves", window_md),
        ("Does the threshold hold at other window lengths?", stability_md),
        ("Sensitivity: anchors whose outcome half was empty",
         "`rung_second` is measured over the OUTCOME half, but the terminal "
         "flags (`ce`, `pp`, `oe`) are measured over the whole window. So a loan "
         "that reached an absorbing state inside the OBSERVATION half has no "
         "rows in the outcome half at all -- `rung_second` is NULL, the DPD "
         "branch cannot fire -- and still receives a definite `to_state` from "
         "first-half evidence. Its `from_state` is read off a truncated half "
         "too: a mean of 5.9 months of a nominal 12 at the 24-month window, and "
         "9.5 of 27 at the 54-month one.\n\n"
         "These pairs are **kept**, classified by the worst rung reached before "
         "the payoff, because the outcome itself is real -- 97.6% of voluntary "
         "payoffs are non-delinquent within six months of exit, so they are "
         "genuine cures. What is wrong is the WINDOW: a transition that resolved "
         "in ten months is being pooled with ones that took twenty-four. So the "
         "matrices are read a second time without them, and if the threshold is "
         "the same in both, the early-resolving pairs are not driving it.\n\n"
         + ("**The threshold HOLDS with them removed.**\n\n"
            if stable_vs_sensitivity else
            "**The threshold MOVES with them removed** -- which is itself the "
            "finding, and the headline number should not be quoted without "
            "this table beside it.\n\n")
         + md_table(sensitivity.round(2)) + "\n\n"
         "The concentration is what makes this worth a section: the affected "
         "share of each row climbs with the rung, because a deeply delinquent "
         "loan leaves the book far more often than a current one. At the "
         "24-month window it is 9.2% of `Current` and 32.2% of `90+ DPD`; at "
         "54 months, 26.6% and 51.3%. The deepest rung is the one the threshold "
         "is read off.\n\n"
         "`movement_both_halves.csv` carries the full folded table for the "
         "restricted population."),
        ("The analysed population", caveat(population)),
        ("Transition matrices, row %",
         "Read one row at a time: *given* the worst rung a loan reached in the "
         "observation half, where did it get to in the outcome half? Rows sum to "
         "100%.\n\n"
         "A terminal state is absorbing, so a loan that prepaid in the first half "
         "is `Prepaid` in the outcome half rather than being dropped as "
         "unobserved.\n\n"
         + "\n\n".join(
             f"**{w}-month window ({excluded[w]['observation_half_months']} observed "
             f"+ {excluded[w]['outcome_half_months']} predicted)**\n\n"
             + md_table(rates[w].round(2), index=True) for w in sorted(rates))),
        ("Transition matrices, counts",
         "\n\n".join(f"**{w}-month window**\n\n"
                     + md_table(matrices[w], index=True, floatfmt="{:.0f}")
                     for w in sorted(matrices))),
        ("Cure, inertia, deterioration",
         "The same matrices folded into the directions a loan can move. An "
         "`Other exit` -- Freddie repurchasing or selling the loan -- is "
         "counted as neither cure nor deterioration, because it is a decision "
         "of Freddie's rather than an outcome the borrower produced.\n\n"
         + "\n\n".join(
             f"**{w}-month window**\n\n"
             + md_table(movements[w].round(2)) for w in sorted(movements))),
        ("Headline transitions by year",
         f"One matrix pooled over the whole panel is an average of several "
         f"credit environments. This is where they separate. Years are the "
         f"ANCHOR month's year, over the {hz}.\n\n" + md_table(by_year.round(2))),
        ("What was excluded",
         "Per window, because the longer the window the more of the panel runs "
         "out before it closes.\n\n"
         + md_table(pd.DataFrame(exclusion_rows).round(2))
         + f"\n\nRows with an unparseable delinquency status (`XX` or blank) and no "
           f"Zero Balance Code: **{unknown_status:,}**. They take part in nothing\n\n"
           "**Censored** anchors are the price of a finite as-of date, not an "
           "error. An anchor is kept only when every month of its window exists "
           "in the panel: a loan that reached an absorbing state early is NOT "
           "exempt, because prepayment resolves in a month and default only "
           "after foreclosure, so exempting terminations would select the "
           "matrix on its own outcome.\n\n"
           "**Nothing observed in a half** is different."),
    ]

    out = write_report(
        Path(a.out), "Roll-Rate Analysis", "src/jobs/roll_rate_main.py", sections,
        preamble=(f"The worst delinquency a loan reaches in the FIRST half of the "
                  f"performance window, beside the worst it reaches in the SECOND "
                  f"half -- over the {hz}, anchored in {period.label}. "
                  f"The two halves an anchor opens may run past the end of that "
                  f"period; the panel is what bounds them. The window comes from "
                  f"{window_source}; this module splits it and reads the bad "
                  f"definition off the result. Terminations (Zero Balance Code) "
                  f"are states in their own right and absorbing, so a loan that "
                  f"pays off or goes to REO is an outcome rather than a "
                  f"disappearance."))
    log.info("Wrote %s", out)
    log.info("Underlying tables in %s", csv_dir)


if __name__ == "__main__":
    main()
