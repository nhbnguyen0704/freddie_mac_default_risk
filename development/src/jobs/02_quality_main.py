import argparse
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from pyspark.sql import DataFrame, SparkSession, functions as F
from helpers.config import settings
from helpers.report import md_table, write_report
from helpers.layout import DELINQUENCY_MISSING, ZERO_BALANCE_DOMAIN
from helpers.period import month_partitions
from helpers.spark_io import resolve_path, spark_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s")
log = logging.getLogger("quality")

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"

PERF_REQUIRED = ['loan_identifier', 'period']
ORIG_SENTINELS = {'classic_fico': 9999, 'original_debt_to_income_ratio': 999, 'original_loan_to_value': 999,
                  'original_combined_loan_to_value': 999, 'mortgage_insurance_percentage': 999}

@dataclass
class Check:
    """One check's outcome: PASS, WARN or FAIL, a one-line summary, and optional detail rows and notes."""
    name: str
    status: str
    summary: str
    details: Optional[pd.DataFrame] = None
    notes: List[str] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return self.status == FAIL

@dataclass
class Curated:
    """The curated root and its listing: performance month partitions and files, and origination files. Files are
    listed through the Hadoop FileSystem API, so the same code reads hdfs:// and file:// roots; only listing and
    footers happen on the driver, and every row-level check is a Spark aggregation."""
    spark: SparkSession
    root: str
    perf_parts: List[Tuple[int, int, str]]
    perf_files: List[str]
    orig_files: List[str]

    @property
    def perf_root(self) -> str:
        return f"{self.root}/performance"

    @property
    def orig_root(self) -> str:
        return f"{self.root}/origination"

    def read(self, paths: List[str], base: str) -> DataFrame:
        return self.spark.read.option("basePath", base).parquet(*paths)

    def read_perf(self, parts: List[Tuple[int, int, str]]) -> DataFrame:
        return self.read([d for _, _, d in parts], self.perf_root)

    def read_orig(self, files: Optional[List[str]] = None) -> DataFrame:
        return self.read(self.orig_files if files is None else files, self.orig_root)

def _list_parquet(spark: SparkSession, root: str) -> List[str]:
    """Every .parquet file under root, recursively (empty when root is missing)."""
    jvm = spark._jvm
    path = jvm.org.apache.hadoop.fs.Path(root)
    fs = path.getFileSystem(spark._jsc.hadoopConfiguration())
    if not fs.exists(path):
        return []
    found, it = [], fs.listFiles(path, True)
    while it.hasNext():
        p = it.next().getPath().toString()
        if p.endswith(".parquet"):
            found.append(p)
    return sorted(found)

def _footer_schema(spark: SparkSession, path: str) -> Dict[str, str]:
    """Column name -> Parquet type, from one file's footer."""
    jvm = spark._jvm
    hfile = jvm.org.apache.parquet.hadoop.util.HadoopInputFile.fromPath(
        jvm.org.apache.hadoop.fs.Path(path), spark._jsc.hadoopConfiguration())
    reader = jvm.org.apache.parquet.hadoop.ParquetFileReader.open(hfile)
    try:
        return {t.getName(): t.toString() for t in reader.getFileMetaData().getSchema().getFields()}
    finally:
        reader.close()

def _tail(path: str) -> str:
    """`year=.../month=.../<file>` -- the part of a path worth printing."""
    m = re.search(r"year=[^/]+/[^/]+=[^/]+/[^/]+$", path)
    return m.group(0) if m else path

def _stem(path: str) -> str:
    """The file name without .parquet: the origination quarter."""
    return path.rsplit("/", 1)[-1][:-len(".parquet")]

def _file_stem():
    """The source file's name without .parquet, per row."""
    return F.regexp_extract(F.col("_metadata.file_name"), r"^(.*)\.parquet$", 1)

def _file_tail():
    """The year=/<part>=/<file> tail of the source file's path, per row."""
    return F.regexp_extract(F.col("_metadata.file_path"), r"(year=[^/]+/[^/]+=[^/]+/[^/]+)$", 1)

def _n(cond):
    return F.sum(F.when(cond, 1).otherwise(0))

def _sample(items: List, n: Optional[int], seed: int = 42) -> List:
    """A seeded sample of n items, sorted, or all of them."""
    if not n or n >= len(items):
        return items
    return sorted(random.Random(seed).sample(items, n))

def check_partition_content(cur: Curated, sample: Optional[int] = None) -> Check:
    """Every performance row's period falls in its year=/month= folder, and every origination row is stamped with
    its file's quarter. `year` and `month` are the partition columns Spark infers from the path."""
    problems: List[str] = []
    parts = _sample(cur.perf_parts, sample)
    dirs = {d for _, _, d in parts}
    checked = sum(f.rsplit("/", 1)[0] in dirs for f in cur.perf_files) + len(cur.orig_files)

    if parts:
        per = cur.read_perf(parts)
        off = F.col("period").isNotNull() & ((F.year("period") != F.col("year"))
                                             | (F.month("period") != F.col("month")))
        bad = (per.groupBy(_file_tail().alias("file"), "year", "month")
               .agg(_n(off).alias("off"), _n(F.col("period").isNull()).alias("nulls"))
               .filter((F.col("off") > 0) | (F.col("nulls") > 0))
               .orderBy("file").collect())
        for r in bad:
            if r["off"]:
                problems.append(f"performance/{r['file']}: {r['off']:,} row(s) outside "
                                f"{r['year']}-{r['month']:02d}")
            if r["nulls"]:
                problems.append(f"performance/{r['file']}: {r['nulls']:,} null reporting period(s)")

    if cur.orig_files:
        want = F.format_string("%dQ%d", F.col("year"), F.col("quarter"))
        bad = (cur.read_orig()
               .withColumn("want", want)
               .groupBy(_file_tail().alias("file"), "want")
               .agg(_n(~F.col("origination_quarter").eqNullSafe(F.col("want"))).alias("wrong"))
               .filter(F.col("wrong") > 0)
               .orderBy("file").collect())
        for r in bad:
            problems.append(f"origination/{r['file']}: {r['wrong']:,} row(s) not stamped {r['want']}")

    return Check("partition content", FAIL if problems else PASS,
                 f"{checked:,} file(s) checked; every partition column matches its path"
                 if not problems else f"{len(problems)} partition(s) disagree with their path",
                 notes=problems[:10])

def check_keys(cur: Curated, sample: Optional[int] = None) -> Check:
    """No duplicate loan within a performance month or an origination file."""
    rows, worst = [], 0
    parts = _sample(cur.perf_parts, sample)
    if parts:
        dup = (cur.read_perf(parts)
               .groupBy("year", "month", "loan_identifier").count()
               .groupBy("year", "month")
               .agg(F.sum("count").alias("rows"), F.sum(F.col("count") - 1).alias("dup"))
               .filter(F.col("dup") > 0)
               .orderBy("year", "month").collect())
        for r in dup:
            worst = max(worst, int(r["dup"]))
            rows.append({"partition": f"year={r['year']}/month={r['month']}",
                         "rows": int(r["rows"]), "duplicate keys": int(r["dup"])})

    if cur.orig_files:
        dup = (cur.read_orig()
               .groupBy(_file_tail().alias("file"), "loan_identifier").count()
               .groupBy("file")
               .agg(F.sum("count").alias("rows"), F.sum(F.col("count") - 1).alias("dup"))
               .filter(F.col("dup") > 0)
               .orderBy("file").collect())
        for r in dup:
            worst = max(worst, int(r["dup"]))
            rows.append({"partition": r["file"], "rows": int(r["rows"]), "duplicate keys": int(r["dup"])})

    return Check("keys", FAIL if rows else PASS,
                 "no duplicate keys" if not rows
                 else f"{len(rows)} partition(s) hold duplicate keys (worst: {worst:,})",
                 details=pd.DataFrame(rows) if rows else None)

def check_coverage(cur: Curated) -> Check:
    """The performance months run without gaps from the first to the last."""
    months = [pd.Period(f"{y}-{m}", freq="M") for y, m, _ in cur.perf_parts]

    if not months:
        return Check("coverage", FAIL, "no performance partitions at all")
    lo, hi = min(months), max(months)
    missing = sorted(set(pd.period_range(lo, hi, freq="M")) - set(months))
    return Check("coverage", WARN if missing else PASS,
                 f"{lo} .. {hi}, {len(months)} month(s)"
                 + (f", {len(missing)} MISSING: {', '.join(str(m) for m in missing[:6])}"
                    if missing else ", contiguous"))

def check_referential(cur: Curated, sample: Optional[int] = None) -> Check:
    """Every performance row has an origination row. A performance file is named for the origination quarter of its
    loans, so <quarter>.parquet under any month joins to origination/.../<quarter>.parquet."""
    orig_by_quarter = {_stem(f): f for f in cur.orig_files}
    perf_by_quarter: Dict[str, List[str]] = {}
    for p in cur.perf_files:
        perf_by_quarter.setdefault(_stem(p), []).append(p)

    rows = []
    no_orig = sorted(set(perf_by_quarter) - set(orig_by_quarter))
    if no_orig:
        counts = (cur.read([p for q in no_orig for p in perf_by_quarter[q]], cur.perf_root)
                  .groupBy(_file_stem().alias("quarter")).count().collect())
        for r in sorted(counts, key=lambda r: r["quarter"]):
            rows.append({"quarter": r["quarter"], "origination loans": 0,
                         "performance rows": int(r["count"]), "orphan rows": int(r["count"])})

    quarters = [q for q in _sample(sorted(orig_by_quarter), sample) if q in perf_by_quarter]
    if quarters:
        orig = (cur.read_orig([orig_by_quarter[q] for q in quarters])
                .select("loan_identifier", _file_stem().alias("quarter")).distinct())
        n_orig = {r["quarter"]: int(r["count"]) for r in orig.groupBy("quarter").count().collect()}
        perf = (cur.read([p for q in quarters for p in perf_by_quarter[q]], cur.perf_root)
                .select("loan_identifier", _file_stem().alias("quarter")))
        found = (perf.join(orig.withColumn("_hit", F.lit(1)), ["quarter", "loan_identifier"], "left")
                 .groupBy("quarter")
                 .agg(F.count(F.lit(1)).alias("rows"), _n(F.col("_hit").isNull()).alias("orphans"))
                 .filter(F.col("orphans") > 0)
                 .orderBy("quarter").collect())
        for r in found:
            rows.append({"quarter": r["quarter"], "origination loans": n_orig.get(r["quarter"], 0),
                         "performance rows": int(r["rows"]), "orphan rows": int(r["orphans"])})
    return Check("referential integrity", FAIL if rows else PASS,
                 "every performance row has an origination row" if not rows
                 else f"{len(rows)} quarter(s) have orphan performance rows",
                 details=pd.DataFrame(rows) if rows else None)

def check_domains(cur: Curated, sample: Optional[int] = None) -> Check:
    """No null keys, delinquency statuses and zero-balance codes in the documented domain, and no origination
    sentinel (FICO 9999, ratios 999) left in place. Origination values are read with try_cast, as a plain cast
    raises under ANSI mode on a non-numeric string."""
    problems, stats = [], []
    parts = _sample(cur.perf_parts, sample)
    n = n_null_key = n_unparsed = n_stray_zb = 0
    zb_seen = set()
    if parts:
        t = cur.read_perf(parts)
        cols = set(t.columns)
        agg = [F.count(F.lit(1)).alias("n")]
        agg += [F.sum(F.col(c).isNull().cast("long")).alias(f"null_{c}") for c in PERF_REQUIRED if c in cols]
        if {'dpd', 'current_loan_delinquency_status'} <= cols:
            raw = F.upper(F.trim(F.col('current_loan_delinquency_status').cast("string")))
            agg.append(_n(F.col('dpd').isNull() & raw.isNotNull()
                          & ~raw.isin(sorted(DELINQUENCY_MISSING))).alias("unparsed"))
        if 'zero_balance_code' in cols:
            zb = F.col('zero_balance_code').cast("string")
            agg.append(_n(zb.isNotNull() & ~zb.isin(sorted(ZERO_BALANCE_DOMAIN))).alias("stray_zb"))
            agg.append(F.collect_set(zb).alias("zb_seen"))
        r = t.agg(*agg).collect()[0].asDict()
        n = int(r["n"] or 0)
        n_null_key = sum(int(r.get(f"null_{c}") or 0) for c in PERF_REQUIRED)
        n_unparsed = int(r.get("unparsed") or 0)
        n_stray_zb = int(r.get("stray_zb") or 0)
        zb_seen = set(r.get("zb_seen") or [])

    if n_null_key:
        problems.append(f"{n_null_key:,} null value(s) in loan_identifier or "
                        f"period")

    if n_unparsed:
        problems.append(f"{n_unparsed:,} delinquency status value(s) parsed to null "
                        f"but are not the documented missing codes")

    if n_stray_zb:
        problems.append(f"{n_stray_zb:,} zero balance code(s) outside the documented "
                        f"domain: {sorted(zb_seen - ZERO_BALANCE_DOMAIN)[:6]}")
    stats.append({"metric": "performance rows scanned", "value": f"{n:,}"})
    stats.append({"metric": "zero balance codes seen", "value": ", ".join(sorted(zb_seen)) or "--"})

    leaked = []
    if cur.orig_files:
        o = cur.read_orig()
        have = [c for c in ORIG_SENTINELS if c in o.columns]
        if have:
            hits = (o.groupBy(_file_stem().alias("quarter"))
                    .agg(*[_n(F.col(c).try_cast("double") == float(ORIG_SENTINELS[c])).alias(c)
                           for c in have])
                    .orderBy("quarter").collect())
            for r in hits:
                for c in have:
                    if r[c]:
                        leaked.append(f"{c}={ORIG_SENTINELS[c]} survives {r[c]:,} time(s) "
                                      f"in {r['quarter']}")
    problems += leaked[:5]

    status = FAIL if (n_null_key or n_stray_zb or leaked) else (WARN if n_unparsed else PASS)
    return Check("value domains", status,
                 "nulls and domains as documented" if status == PASS
                 else f"{len(problems)} domain problem(s)",
                 details=pd.DataFrame(stats), notes=problems)

def check_schema(cur: Curated, sample: Optional[int] = None) -> Check:
    """One schema per grain. Per-file schemas come from the Parquet footers, read on the driver, because Spark only
    reports the one schema it chose for the whole read."""
    problems, n_mismatch = [], 0
    for label, files in (("performance", _sample(cur.perf_files, sample)),
                         ("origination", cur.orig_files)):
        base = base_file = None
        for f in files:
            fields = _footer_schema(cur.spark, f)
            if base is None:
                base, base_file = fields, f
                continue
            if fields != base:
                added = set(fields) - set(base)
                lost = set(base) - set(fields)
                retyped = {k for k in set(fields) & set(base) if fields[k] != base[k]}
                n_mismatch += 1
                if len(problems) < 10:
                    problems.append(f"{label}: {_tail(f)} differs from {_tail(base_file)} "
                                    f"(added {sorted(added) or '-'}, missing "
                                    f"{sorted(lost) or '-'}, retyped {sorted(retyped) or '-'})")
    return Check("schema consistency", FAIL if n_mismatch else PASS,
                 "one schema per grain" if not n_mismatch
                 else f"{n_mismatch} file(s) differ from the first schema of their grain",
                 notes=problems)

def open_curated(spark: SparkSession, root: str) -> Curated:
    """The curated root with its partitions and files listed. month_partitions exits with one line if the
    filesystem is unreachable."""
    root = resolve_path(root).rstrip("/")
    parts = month_partitions(spark, f"{root}/performance") or []
    return Curated(spark, root, parts,
                   _list_parquet(spark, f"{root}/performance"),
                   _list_parquet(spark, f"{root}/origination"))

def _guarded(name: str, fn, *args) -> Check:
    """Run a check, turning an exception into a failed check. Spark reads a grain under one schema, so a file whose
    column type drifted fails the whole read; that is reported as a failed check rather than losing the report."""
    try:
        return fn(*args)
    except Exception as exc:
        cause = next((l.strip() for l in str(exc).splitlines() if l.strip()), type(exc).__name__)
        log.exception("%s could not run", name)
        return Check(name, FAIL, "could not read the data -- see schema consistency",
                     notes=[cause[:300]])

def run(cur: Curated, sample: Optional[int] = None) -> List[Check]:
    """Every check, schema consistency first."""
    schema = check_schema(cur, sample)
    return [_guarded("partition content", check_partition_content, cur, sample),
            _guarded("keys", check_keys, cur, sample),
            check_coverage(cur),
            _guarded("referential integrity", check_referential, cur, sample),
            _guarded("value domains", check_domains, cur, sample),
            schema]

def main():
    p = argparse.ArgumentParser(description="Data quality check")
    p.add_argument("--curated", "--out-dir", dest="curated", default=settings.hdfs_curated_url,
                   help="curated root holding performance/ and origination/ "
                        "(hdfs:// URL, or a local directory)")
    p.add_argument("--sample-partitions", type=int, default=0,
                   help="check this many performance partitions; 0 to check all")
    p.add_argument("--report", default="docs/data_quality.md")
    p.add_argument("--warn-only", action="store_true")
    a = p.parse_args()

    sample = a.sample_partitions or None
    spark = spark_session("quality")
    try:
        cur = open_curated(spark, a.curated)
        log.info("Checking %s%s", cur.root,
                 f" (sampling {sample} performance partitions)" if sample else " (all partitions)")
        log.info("Found %s performance partition(s), %s performance file(s), %s origination file(s)",
                 f"{len(cur.perf_parts):,}", f"{len(cur.perf_files):,}", f"{len(cur.orig_files):,}")
        checks = run(cur, sample)
    finally:
        spark.stop()

    for c in checks:
        (log.error if c.status == FAIL else
         log.warning if c.status == WARN else log.info)(
            "%-22s %-4s %s", c.name, c.status, c.summary)
        for n in c.notes:
            log.info("    - %s", n)

    summary = pd.DataFrame([{"check": c.name, "status": c.status, "result": c.summary}
                            for c in checks])
    sections = [("Summary", md_table(summary))]
    for c in checks:
        body = c.summary
        if c.details is not None and len(c.details):
            body += "\n\n" + md_table(c.details)
        if c.notes:
            body += "\n\n" + "\n".join(f"- {n}" for n in c.notes)
        sections.append((f"{c.name} -- {c.status}", body))

    n_fail = sum(c.failed for c in checks)
    n_warn = sum(c.status == WARN for c in checks)
    out = write_report(
        Path(a.report), "Curated Data Quality", "src/jobs/quality_main.py", sections,
        preamble=(f"{len(checks)} checks on `{cur.root}`: **{n_fail} failed**, "
                  f"{n_warn} warned. "))
    log.info("Wrote %s", out)
    if n_fail:
        log.error("Data Quality: FAIL (%s check(s)). -- read %s.", n_fail, a.report)
        raise SystemExit(0 if a.warn_only else 1)
    log.info("Data Quality: PASS%s", f" ({n_warn} warning(s))" if n_warn else "")

if __name__ == "__main__":
    main()
