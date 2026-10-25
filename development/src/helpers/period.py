import argparse
import logging
from dataclasses import dataclass
from datetime import date
import time
from typing import Callable, Dict, Optional, Set, Tuple, List
import pandas as pd
from helpers.config import settings
from pyspark.sql import DataFrame, functions as F
from helpers.spark_io import resolve_path, read_curated
import pyarrow.dataset as ds

log = logging.getLogger('period')

PERIOD_COLUMN = 'period'

def _month(value, what: str) -> pd.Timestamp:
    """The first of the month of value, exiting with a message when it is not a date."""
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Could not convert {value} to timestamp: {exc}")
    if pd.isna(ts):
        raise SystemExit(f"Could not convert {value} to timestamp: {ts}")
    return ts.normalize().replace(day=1)

@dataclass(frozen=True)
class AnalysisPeriod:
    """The first and last observation month, and where they came from (config or command line)."""
    start: pd.Timestamp
    end: pd.Timestamp
    source: str = 'config'

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise SystemExit("End date must be after start date")

    @property
    def label(self) -> str:
        return f"{self.start:%Y-%m}-{self.end:%Y-%m}"

    @property
    def months(self):
        return (self.end.to_period('M') - self.start.to_period('M')).n + 1

    def lookforward_months(self, asof: Optional[date]) -> int:
        """How many months the data runs past the last observation month (0 without asof)."""
        if asof is None:
            return 0
        return max(0, (pd.Timestamp(asof).to_period('M') - self.end.to_period('M')).n)

    def describe(self, asof: Optional[date] = None, window_months: Optional[int] = None) -> str:
        """The period in words for a report, and whether the data past it is long enough to close a window_months
        performance window for every observation month."""
        tail = self.lookforward_months(asof)
        body = (
            f"Analysis period: {self.label} ({self.months} observation months), from {self.source}. \n\n"
        )
        if asof is None:
            return body
        if tail:
            body += (
                f"\n\nThe data runs to {asof}, which is {tail} month(s) past the last observation month"
            )
            if window_months:
                covered = self.end + pd.DateOffset(months=window_months)
                if covered <= pd.Timestamp(asof):
                    body += ", and it is long enough. Every observation month in the period can be labelled"
                else:
                    short = (covered.to_period('M') - pd.Timestamp(asof).to_period('M')).n
                    body += (f", and it is {short} month(s) short. "
                             f"The last {short} observation month(s) of the period have no closed window")
            else:
                body += '.'
        else:
            body += (
                f"\n\nThe panel ends at {asof}, with no months past the period at all"
            )
        return body

    def to_dict(self) -> Dict[str, object]:
        return {"analysis_period": self.label,
                "analysis_start": str(self.start.date()),
                "analysis_end": str(self.end.date()),
                "analysis_months": self.months,
                "analysis_bounds": "observation months only",
                "analysis_period_source": self.source}

def resolve(start=None, end=None) -> AnalysisPeriod:
    """The analysis period from the given start and end, each defaulting to OBSERVATION_START / OBSERVATION_END."""
    if start is None:
        start, s = settings.observation_start, "OBSERVATION_START"
    else:
        s = "--period-start"
    if end is None:
        end, e = settings.observation_end, "OBSERVATION_END"
    else:
        e = "--period-end"
    return AnalysisPeriod(_month(start, '--period-start'), _month(end, '--period-end'),
                          source=s if s == e else f"{s} / {e}")


def add_arguments(p: argparse.ArgumentParser) -> None:
    """Add --period-start and --period-end to a job's parser."""
    g = p.add_argument_group(
        'Analysis period',
        "The months an observation may come from. Outcomes are resolved against "
        "months after the period, so the end does not have to leave room for a "
        "performance window"
    )
    g.add_argument("--period-start", metavar="YYYY-MM",
                   help=f"First observation month (default: "
                        f"{settings.observation_start:%Y-%m}, from OBSERVATION_START")
    g.add_argument("--period-end", metavar="YYYY-MM",
                   help=f"Last observation month (default: "
                        f"{settings.observation_end:%Y-%m}, from OBSERVATION_END")

def from_args(a: argparse.Namespace) -> AnalysisPeriod:
    """The analysis period from a job's parsed arguments."""
    period = resolve(getattr(a, 'period_start', None), getattr(a, 'period_end', None))
    log.info("Analysis period: %s (%s observation months) -- from %s. Outcomes "
             "resolve against months after it.",
             period.label, period.months, period.source)
    return period

PARTITION_COLUMNS = ('year', 'month')

def _has_partitions(names) -> bool:
    return all(c in set(names) for c in PARTITION_COLUMNS)

_UNREACHABLE = ("ConnectException", "Connection refused", "UnknownHostException",
                "NoRouteToHostException", "ConnectTimeoutException",
                "SocketTimeoutException", "Operation category READ is not supported",
                "Call From")


def _unreachable(exc: Exception) -> bool:
    """Whether the exception says the filesystem cannot be reached. That is a different failure from a panel with
    an odd directory layout and must not be handled the same way; _UNREACHABLE holds the markers Hadoop puts in
    the message."""
    text = f"{type(exc).__name__}: {exc}"
    return any(m in text for m in _UNREACHABLE)


def _why(exc: Exception) -> str:
    """The one informative line out of a Java stack trace.

    Py4J's first line is always "An error occurred while calling oNN.method",
    which says nothing. The line worth printing is the one carrying the cause,
    so find that and drop the hundred frames after it.
    """
    for line in str(exc).splitlines():
        line = line.strip()
        if any(m in line for m in _UNREACHABLE):
            return line[:240]
    return str(exc).splitlines()[0][:240] if str(exc) else type(exc).__name__


def _has_partitions(names) -> bool:
    return all(c in set(names) for c in PARTITION_COLUMNS)


def partition_floor_spark(period: AnalysisPeriod):
    """A partition filter keeping year/month from the period's first month on."""
    return (
        (F.col('year') > period.start.year) |
        (F.col('year') == period.start.year) & (F.col('month') >= period.start.month)
    )

def month_partitions(spark, path: str) -> Optional[list]:
    """(year, month, uri) of every year=/month= folder under path, sorted, or None when it cannot be listed. An
    unreachable filesystem exits here with one line: returning None would send the caller down the fallback
    path, which retries the same location through spark.read.parquet and buries the cause under a Java stack
    trace."""
    try:
        jvm = spark._jvm
        hpath = jvm.org.apache.hadoop.fs.Path(resolve_path(path))
        fs = hpath.getFileSystem(spark._jsc.hadoopConfiguration())
        found: List[Tuple[int, int, str]] = []
        for y in fs.listStatus(hpath):
            name = y.getPath().getName()
            if not (y.isDirectory() and name.startswith('year=')):
                continue
            year = int(name.split('=', 1)[1])
            for m in fs.listStatus(y.getPath()):
                mname = m.getPath().getName()
                if m.isDirectory() and mname.startswith('month='):
                    found.append((year, int(mname.split('=', 1)[1]), m.getPath().toString()))
        return sorted(found)
    except Exception as exc:
        if _unreachable(exc):
            raise SystemExit(
                f"Cannot reach the filesystem holding {path}.\n"
                f"  {_why(exc)}\n"
                f"  -- if HDFS is meant to be up, start it and check "
                f"`hdfs dfs -ls {path}`;\n"
                f"  -- to read a local copy instead, pass --performance with a "
                f"local directory.") from exc
        log.warning("Could not list partitions under %s (%s). Falling back to "
                    "reading the root and pruning with a partition filter.",
                    path, exc)
        return None

def read_period(spark, path: str, period: AnalysisPeriod) -> DataFrame:
    """Only the year=/month= folders inside the period, as feature_engineering_main
    reads its panel -- Spark never lists or opens the months outside it."""
    parts = month_partitions(spark, path)
    if not parts:
        raise SystemExit(f'{path} is not partitioned by year/month')
    lo = period.start.year * 100 + period.start.month
    hi = period.end.year * 100 + period.end.month
    uris = [uri for y, m, uri in parts if lo <= y * 100 + m <= hi]
    if not uris:
        raise SystemExit(f'{path} has no months in {period.label}')
    log.info('Reading %s of %s months from %s', len(uris), len(parts), path)
    return spark.read.option('basePath', resolve_path(path)).parquet(*uris)

def complete_months(spark, path: str) -> Set[Tuple[int, int]]:
    """year=/month= folders under path holding a committed write: at least one
    parquet file and no _temporary left by a write that died mid-way."""
    jvm = spark._jvm
    root = jvm.org.apache.hadoop.fs.Path(resolve_path(path))
    fs = root.getFileSystem(spark._jsc.hadoopConfiguration())
    if not fs.exists(root):
        return set()
    done: Set[Tuple[int, int]] = set()
    for y, m, uri in month_partitions(spark, path) or []:
        names = [st.getPath().getName() for st in fs.listStatus(jvm.org.apache.hadoop.fs.Path(uri))]
        if '_temporary' not in names and any(n.endswith('.parquet') for n in names):
            done.add((y, m))
    return done

def write_by_month(spark, src: str, out: str, build: Callable[[DataFrame, int, int], DataFrame],
                   resume: bool = False, files_per_month: int = 0) -> Dict[str, object]:
    """Row-wise transform of a year/month-partitioned dataset, one month at a time.

    Each month is read from its own folder and written straight into
    out/year=Y/month=M -- no partitionBy, so no shuffle and no sort before the
    write. A task holds one input split and one parquet writer, whatever the
    month's size. partitionBy with a repartition on 6-month chunks ran the
    10g local heap out of memory on the 2021 months.

    build(df, year, month) gets the month WITHOUT year/month columns (they
    live in the path) and returns what to write. files_per_month > 0
    coalesces (no shuffle) to at most that many files; 0 keeps one file per
    input split.
    """
    parts = month_partitions(spark, src)
    if not parts:
        raise SystemExit(f'{src} is not partitioned by year/month')
    done: Set[Tuple[int, int]] = set()
    if resume:
        done = complete_months(spark, out) & {(y, m) for y, m, _ in parts}
        log.info('Resuming: %s of %s months already written', len(done), len(parts))
    todo = [(y, m, uri) for y, m, uri in parts if (y, m) not in done]
    base = out.rstrip('/')
    t0 = time.time()
    for i, (y, m, uri) in enumerate(todo, 1):
        t = time.time()
        df = build(spark.read.parquet(uri), y, m)
        if files_per_month > 0:
            df = df.coalesce(files_per_month)
        df.write.mode('overwrite').parquet(f'{base}/year={y}/month={m}')
        log.info('[%s/%s] %s-%02d written in %.0fs', i, len(todo), y, m, time.time() - t)
    return {'months': len(parts), 'months_written': len(todo), 'months_skipped': len(done),
            'seconds': round(time.time() - t0)}


def observation_mask_spark(period: AnalysisPeriod, column: str = PERIOD_COLUMN):
    """Rows whose period falls inside the analysis period."""
    return (
        (F.col(column) >= F.lit(period.start.date())) &
        (F.col(column) <= F.lit(period.end.date()))
    )

def _row_census(df: DataFrame, period: AnalysisPeriod, column: str = PERIOD_COLUMN) -> Dict[str, object]:
    """Rows read, rows inside the period and rows after it."""
    counted = df.agg(
        F.count(F.lit(1)).alias('read'),
        F.sum(F.when(observation_mask_spark(period, column), 1).otherwise(F.lit(0))).alias('anchors')
    ).collect()[0]
    read, anchors = int(counted['read'] or 0), int(counted['anchors'] or 0)
    return {"panel_rows_read": read,
            "panel_rows_in_period": anchors,
            "panel_rows_remaining": read - anchors}

def _require_anchors(stats: Dict[str, object], period: AnalysisPeriod) -> None:
    """Exit when no row falls inside the analysis period."""
    if not stats.get('panel_rows_in_period'):
        raise SystemExit("No panel_rows_in_period")

def clip_panel_spark(df: DataFrame, period: AnalysisPeriod, column: str = PERIOD_COLUMN, count: bool = True,
                     path: Optional[str] = None) -> Tuple[DataFrame, Dict[str, object]]:
    """The panel from the period's first month on, by a partition filter when it has year/month columns and a row
    filter on period otherwise, plus the row and partition counts when count is set. Months after the period are
    kept as outcome evidence."""
    stats: Dict[str, object] = dict(period.to_dict())

    if _has_partitions(df.columns):
        floor = partition_floor_spark(period)
        stats['panel_floor_applied_as'] = 'partition filter on year/month'
    elif column in df.columns:
        log.warning("The panel has no year/month partition columns, so the "
                    "period floor has to be a row filter on %s", column)
        floor = F.col(column) >= F.lit(period.start.date())
        stats['panel_floor_applied_as'] = f'row filter on {column}'
    else:
        raise SystemExit(
            f"The panel carries neither year/month partition columns nor "
            f"{column!r}, so the analysis period cannot be applied"
        )

    clipped = df.filter(floor)

    if count:
        stats.update(_row_census(clipped, period, column))
        if path is not None:
            parts = month_partitions(df.sparkSession, path)
            if parts is not None:
                floor_key = (period.start.year, period.start.month)
                kept = [q for q in parts if (q[0], q[1]) >= floor_key]
                stats.update({"panel_partitions_total": len(parts),
                              "panel_partitions_read": len(kept),
                              "panel_partitions_skipped": len(parts) - len(kept)})
        log.info("Analysis period %s: read %s loan-months -- %s inside the "
                 "period, %s after it as outcome evidence.", period.label,
                 f"{stats['panel_rows_read']:,}",
                 f"{stats['panel_rows_in_period']:,}",
                 f"{stats['panel_rows_remaining']:,}")
        _require_anchors(stats, period)
    return clipped, stats


def read_panel_spark(spark, path: str, period: AnalysisPeriod,
                     count: bool = True) -> Tuple[DataFrame, Dict[str, object]]:
    """The panel from the period's first month on, reading only those year=/month= folders (falling back to
    clip_panel_spark when the path is not partitioned), plus the row and partition counts when count is set."""
    parts = month_partitions(spark, path)
    if not parts:
        if parts is not None:
            log.warning("No partitions found for %s", path)
        return clip_panel_spark(read_curated(spark, path), period, count=count, path=path)

    floor = (period.start.year, period.start.month)
    kept = [q for q in parts if (q[0], q[1]) >= floor]
    if not kept:
        raise SystemExit(f"No reporting months at or after {period.start:%Y-%m} under {path}")

    root = resolve_path(path)
    df = spark.read.option('basePath', root).parquet(*[q[2] for q in kept])
    log.info("Opened %s of %s monthly partitions (%s .. %s)", len(kept), len(parts),
             f"{kept[0][0]}-{kept[0][1]:02d}", f"{kept[-1][0]}-{kept[-1][1]:02d}")

    stats: Dict[str, object] = dict(period.to_dict())
    stats.update({"panel_floor_applied_as": "partition paths named at read time",
                  "panel_partitions_total": len(parts),
                  "panel_partitions_read": len(kept),
                  "panel_partitions_skipped": len(parts) - len(kept)})

    if count:
        stats.update(_row_census(df, period))
        log.info("Analysis period %s: read %s loan-months -- %s inside the "
                 "period, %s after it as outcome evidence.", period.label,
                 f"{stats['panel_rows_read']:,}",
                 f"{stats['panel_rows_in_period']:,}",
                 f"{stats['panel_rows_remaining']:,}")
        _require_anchors(stats, period)
    return df, stats

def arrow_panel_filter(period: AnalysisPeriod, schema_names, column: str = PERIOD_COLUMN):
    """The pyarrow dataset filter equivalent of the period floor."""
    if _has_partitions(schema_names):
        return (
            (ds.field('year') > period.start.year) |
            (ds.field('year') == period.start.year) &
            (ds.field('month') >= period.start.month)
        )
    if "year" in schema_names:
        return ds.field('year') >= period.start.year
    if column in schema_names:
        log.warning("The panel has no year/month partition columns, so the "
                    "period floor has to be a row filter on %s", column)
        return ds.field(column) >= period.start.to_pydatetime()
    raise SystemExit(f"The panel carries neither year/month partition columns nor {column!r}")

def observation_mask_pandas(df: pd.DataFrame, period: AnalysisPeriod, column: str = PERIOD_COLUMN) -> pd.Series:
    """Rows whose period falls inside the analysis period (all rows when there is no period column)."""
    if column not in df.columns:
        return pd.Series(True, index=df.index)
    per = pd.to_datetime(df[column])
    return (per >= period.start) & (per <= period.end)

def resolve_asof(explicit: Optional[str], observed: Optional[date], period: AnalysisPeriod) -> Tuple[date, str]:
    """The as-of date and how it was chosen: the explicit --asof, else the last reporting month in the panel."""
    if explicit and explicit != 'auto':
        return date.fromisoformat(explicit), f"--asof {explicit} (explicit override)"
    if observed is None:
        raise SystemExit(f"No reporting months found in the panel")
    tail = period.lookforward_months(observed)
    if tail == 0:
        log.warning("The panel ends at %s, which is not past the last observation "
                    "month (%s).", observed, f"{period.end:%Y-%m}")
    return observed, (f"the last reporting month in the panel, {tail} month(s) "
                      f"past the last observation month ({period.end:%Y-%m})")


