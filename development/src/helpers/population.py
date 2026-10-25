import logging
from typing import Dict, Tuple
from helpers.layout import (DISASTER_FLAG, DISASTER_VALUES, SPECIAL_MENTION_FLAGS,
                            SPECIAL_MENTION_POSITIVE_UPB, SPECIAL_MENTION_PRIOR)
from pyspark.sql import functions as F

log = logging.getLogger('population')

def _any_of(columns, flags: Dict[str, set]):
    """True where any of the present columns holds one of its flagged values."""
    cond = F.lit(False)
    for col, values in flags.items():
        if col in columns:
            cond = cond | F.col(col).isin(sorted(values))
    return cond

def workout_condition(perf, include_prior: bool = True):
    """True where the loan-month carries a workout marker.

    `include_prior` picks which of the two senses Freddie encodes in these
    columns (see SPECIAL_MENTION_PRIOR in layout.py):

      True   EVER modified -- the intervention happened at any point, this month
             or earlier ('Y'/'C' or 'P'). This is the sense for EXCLUDING a loan
             and for deciding whether a month may open a fresh observation: a
             modification is a permanent change in the regime the loan is
             managed under, so the months after it are shaped by it too. It also
             handles left truncation correctly -- a loan modified in 2012 and
             flagged 'P' throughout the panel is never eligible, rather than
             being counted as a NEW bad at the start of the window.

      False  the intervention happened THIS MONTH ('Y'/'C'). This is the sense
             for anything that needs a date, or a rung: pushing every post-
             modification month to the deepest rung would read a decade of
             performing payments as permanent default.

    Exported so the code that excludes workouts and the code that scores them
    read one definition rather than two copies that drift.
    """
    cols = set(perf.columns)
    cond = _any_of(cols, SPECIAL_MENTION_FLAGS)
    if include_prior:
        cond = cond | _any_of(cols, SPECIAL_MENTION_PRIOR)
    if SPECIAL_MENTION_POSITIVE_UPB in cols:
        cond = cond | (F.col(SPECIAL_MENTION_POSITIVE_UPB) > 0)
    return cond


def restrict(perf, exclude_workouts: bool = True, exclude_disaster: bool = True) -> Tuple[object, Dict[str, object]]:
    """perf without the loans ever in a workout or ever disaster-delinquent (each optional), and the counts of
    loans held out."""
    cols = set(perf.columns)
    workout = workout_condition(perf)
    disaster = (F.col(DISASTER_FLAG).isin(sorted(DISASTER_VALUES)) if DISASTER_FLAG in cols else F.lit(False))

    per_loan = perf.groupBy('loan_identifier').agg(
        F.max(F.when(workout, 1).otherwise(0)).alias('workout'),
        F.max(F.when(disaster, 1).otherwise(0)).alias('disaster')
    )

    drop = F.lit(False)
    if exclude_workouts:
        drop = drop | (F.col('workout') == 1)
    if exclude_disaster:
        drop = drop | (F.col('disaster') == 1)

    counted = per_loan.agg(
        F.count(F.lit(1)).alias('loans'),
        F.sum('workout').alias('workout'),
        F.sum('disaster').alias('disaster'),
        F.sum(F.when(drop, 1).otherwise(0)).alias('dropped')
    ).collect()[0]

    stats = {
        'loan_before': int(counted['loans']),
        'loans_with_workout': int(counted['workout'] or 0),
        'loans_with_disaster': int(counted['disaster'] or 0),
        'loans_dropped': int(counted['dropped'] or 0),
        'loan_analysed': int(counted['loans']) - int(counted['dropped'] or 0),
        'excluded_workouts': exclude_workouts,
        'excluded_disasters': exclude_disaster,
    }
    log.info("Analysed population: %s of %s loans. Held out %s -- %s ever in a workout, "
             "%s ever disaster-delinquent.",
             f'{stats['loan_analysed']:,}', f'{stats['loan_before']:,}', f'{stats['loans_dropped']:,}',
             f'{stats['loans_with_workout']:,}', f'{stats['loans_with_disaster']:,}', )

    keep = per_loan.filter(~drop).select('loan_identifier')
    return perf.join(keep, 'loan_identifier'), stats

def caveat(stats: Dict[str, object]) -> str:
    """The population exclusion in words for a report."""
    if not stats.get('loans_dropped'):
        return ('No loan carries a workout or disaster flag')
    return (
        f"{stats['loans_dropped']:,} of {stats['loan_before']:,} loans are held out of this analysis. "
        f"{stats['loan_analysed']:,} remain. "
        f"{stats['loans_with_workout']:,} were ever in a workout (modification, payment deferral, "
        f"forbearance or repayment plan, or principal made non-interest-bearing) and {stats['loans_with_disaster']:,} "
        f"were ever delinquent due to a declared disaster. "
        f"These loans were removed only during the bad definition analysis. "
        f"These loans were included in the model training with tags of special mentioned events."
    )