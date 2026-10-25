import argparse
import json
import logging
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from helpers.config import settings
from helpers.archives import QuarterArchive, QuarterSource, check_quarters, discover
from helpers.layout import (DATE_FORMAT, DELINQUENCY_MISSING, DELINQUENCY_SPECIAL, N_ORIG, N_PERF, ORIG_DATE_COLS,
                            PERF_DATE_COLS, REO_ACQUISITION_DPD, ZERO_BALANCE_DOMAIN, resolve_columns)

log = logging.getLogger("ingestion")

PARTITION_ON = 'period'

PERF_COLS = ['loan_identifier', 'period', 'current_actual_upb', 'current_loan_delinquency_status', 'loan_age',
             'remaining_months_to_legal_maturity', 'modification_flag', 'zero_balance_code',
             'zero_balance_effective_date', 'current_interest_rate', 'current_non_interest_bearing_upb',
             'interest_rate_step_indicator', 'payment_deferral_flag', 'estimated_loan_to_value',
             'delinquency_due_to_disaster', 'borrower_assistance_plan', 'servicer_name']

ORIG_COLS = ['loan_identifier', 'classic_fico', 'first_payment_date', 'first_time_homebuyer_indicator', 'maturity_date',
             'metropolitan_statistical_area_or_metropolitan_division', 'mortgage_insurance_percentage',
             'number_of_units', 'occupancy_status', 'original_combined_loan_to_value', 'original_debt_to_income_ratio',
             'original_upb', 'original_loan_to_value', 'original_interest_rate', 'channel',
             'prepayment_penalty_indicator', 'amortization_type', 'property_state', 'property_type', 'postal_code',
             'loan_purpose', 'original_loan_term', 'number_of_borrowers', 'seller_name', 'super_conforming_flag',
             'special_eligibility_program', 'harp_indicator', 'property_valuation_method', 'interest_only_indicator']

NUMERIC_COLS = ['classic_fico', 'mortgage_insurance_percentage', 'number_of_units', 'original_combined_loan_to_value',
                'original_debt_to_income_ratio', 'original_upb', 'original_loan_to_value', 'original_interest_rate',
                'original_loan_term', 'number_of_borrowers', 'current_actual_upb', 'loan_age',
                'remaining_months_to_legal_maturity', 'current_interest_rate', 'current_non_interest_bearing_upb',
                'estimated_loan_to_value']

SENTINELS: Dict[str, Set[float]] = {
    'classic_fico': {9999.0},
    'original_debt_to_income_ratio': {999.0},
    'original_loan_to_value': {999.0},
    'original_combined_loan_to_value': {999.0},
    'mortgage_insurance_percentage': {999.0},
    'estimated_loan_to_value': {999.0},
    'number_of_units': {99.0},
    'number_of_borrowers': {99.0},
}

_PERF_FIELDS = {
    'loan_identifier': pa.string(),
    'period': pa.timestamp('us'),
    'current_actual_upb': pa.float64(),
    'current_loan_delinquency_status': pa.string(),
    'dpd': pa.float64(),
    'loan_age': pa.float64(),
    'remaining_months_to_legal_maturity': pa.float64(),
    'modification_flag': pa.string(),
    'zero_balance_code': pa.string(),
    'zero_balance_effective_date': pa.timestamp('us'),
    'current_interest_rate': pa.float64(),
    'current_non_interest_bearing_upb': pa.float64(),
    'interest_rate_step_indicator': pa.string(),
    'payment_deferral_flag': pa.string(),
    'estimated_loan_to_value': pa.float64(),
    'delinquency_due_to_disaster': pa.string(),
    'borrower_assistance_plan': pa.string(),
    'servicer_name': pa.string()
}

def _peek_columns(arc: QuarterArchive, grain: str) -> int:
    """The number of columns in the first line of a grain's file."""
    with arc.open(grain) as fd:
        head = pd.read_csv(fd, sep='|', header=None, dtype=str, nrows=1, na_values=[''], keep_default_na=False)
    return head.shape[1]

def parse_delinquency(s: pd.Series) -> pd.Series:
    """Delinquency status as a number of months past due: 'RA' (REO acquisition) becomes REO_ACQUISITION_DPD, 'XX'
    and anything unparsable missing."""
    txt = s.astype(str).str.strip().str.upper()
    out = pd.to_numeric(txt.where(~txt.isin(DELINQUENCY_SPECIAL | DELINQUENCY_MISSING)), errors='coerce')
    return out.mask(txt.isin(DELINQUENCY_SPECIAL), REO_ACQUISITION_DPD)

def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Typed columns: dates parsed, numerics as float64 with sentinels (FICO 9999, 999, 99) made missing, dpd added
    and the zero-balance code padded to two digits. Numerics stay float64 even when a quarter has no gaps, so
    the origination schema does not drift to int64 quarter by quarter."""
    for c in ORIG_DATE_COLS + PERF_DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c].astype(str).str.strip(), format=DATE_FORMAT, errors='coerce')
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float64')
            bad = SENTINELS.get(c)
            if bad:
                df[c] = df[c].mask(df[c].isin(bad))
    if 'current_loan_delinquency_status' in df.columns:
        df['dpd'] = parse_delinquency(df['current_loan_delinquency_status'])
    if 'zero_balance_code' in df.columns:
        zb = df['zero_balance_code'].astype('string').str.strip()
        df['zero_balance_code'] = zb.mask(zb.eq('').fillna(False)).str.zfill(2)
    return df

def verify(src: QuarterSource) -> bool:
    """Check one quarter's raw files: both grains present, column counts, unique loans and loan-months, FICO in
    range, first payment + term = maturity, zero-balance codes and delinquency statuses in the documented
    domain, and parsable periods."""
    ok = True
    with QuarterArchive(src) as arc:
        log.info("Checking quarter: %s", src.quarter)
        grains = arc.grains()
        if set(grains) != {'origination', 'performance'}:
            log.error("Found grains: %s", grains)
            return False

        n = _peek_columns(arc, 'origination')
        names = resolve_columns('origination', n)
        with arc.open('origination') as fd:
            df = pd.read_csv(fd, sep='|', header=None, names=names, dtype=str, na_values=[""],
                             keep_default_na=False)
        log.info("Origination  %s rows x %s cols  %s", f"{len(df):,}", n, src.label)
        if n != N_ORIG:
            log.error("Expected %s columns, found %s", N_ORIG, str(n))
            ok = False
        dup = int(df['loan_identifier'].duplicated().sum()) if 'loan_identifier' in df else -1
        if dup > 0:
            log.error("Found duplicate loan identifiers: %s", dup)
            ok = False
        if 'classic_fico' in df.columns:
            cs = pd.to_numeric(df['classic_fico'], errors='coerce')
            inrange = cs.between(300, 850).mean()
            if inrange < 0.5:
                log.error("Proportion of classic fico in range is too low: %s", inrange)
                ok = False

        if {'first_payment_date', 'maturity_date', 'original_loan_term'} <= set(df.columns):
            fp = pd.to_datetime(df['first_payment_date'].astype(str).str.strip(), format=DATE_FORMAT,
                                errors='coerce').dt.to_period('M')
            mat = pd.to_datetime(df['maturity_date'].astype(str).str.strip(), format=DATE_FORMAT,
                                 errors='coerce').dt.to_period('M')
            term = pd.to_numeric(df['original_loan_term'], errors='coerce')
            usable = fp.notna() & mat.notna() & term.notna()
            if usable.any():
                agree = float((fp[usable] + (term[usable].astype('int64') - 1) == mat[usable]).mean())
                if agree < 0.99:
                    log.error("Relationship between first_payment_date, maturity_date and original_loan_term doesn't hold")
                    ok = False

        n = _peek_columns(arc, 'performance')
        names = resolve_columns('performance', n)
        with arc.open('performance') as fd:
            df = pd.read_csv(fd, sep='|', header=None, names=names, dtype=str, na_values=[""],
                             keep_default_na=False)
        log.info("performance  %s rows x %s cols", f"{len(df):,}", n)
        if n != N_PERF:
            log.error("Expected %s columns, found %s", N_PERF, str(n))
            ok = False

        vals = set(df['zero_balance_code'].dropna().astype(str).str.strip()
                   .str.zfill(2).unique()) if 'zero_balance_code' in df else set()
        stray = vals - ZERO_BALANCE_DOMAIN
        if stray:
            log.error("Found zero balance codes outside of documented ones: %s", stray)
            ok = False

        if 'current_loan_delinquency_status' in df.columns:
            raw = df['current_loan_delinquency_status'].astype(str).str.strip().str.upper()
            unparsed = raw[parse_delinquency(raw).isna() & ~raw.isin(DELINQUENCY_MISSING)].unique()
            if len(unparsed) > 0:
                log.error("Found unparsed loan delinquency status: %s", unparsed)
                ok = False

        if 'period' in df.columns:
            per = pd.to_datetime(df['period'].astype(str).str.strip(), format=DATE_FORMAT, errors='coerce')
            if per.isna().any():
                log.error("Found nan periods")
                ok = False

            dup = int(df.duplicated(subset=['loan_identifier', 'period']).sum())
            if dup > 0:
                log.error("Found duplicate loan identifier - period pairs: %s", dup)
                ok = False

    log.info("Layout verification: %s", "Passed" if ok else "Failed")
    return ok

class PartitionWriter:
    """Writes one quarter's performance rows to year=/month= parquet files, through a .writing temp file per month
    that replaces the final file only when the whole quarter succeeded."""
    WRITING_SUFFIX = '.writing'

    def __init__(self, root: Path, quarter: str, schema: pa.Schema):
        self.root, self.quarter, self.schema = root, quarter, schema
        self._writers: Dict[pd.Timestamp, pq.ParquetWriter] = {}
        self._tmp: Dict[pd.Timestamp, Path] = {}
        self.paths: List[Path] = []

    def _path(self, period: pd.Timestamp) -> Path:
        d = self.root / f'year={period.year:04d}' / f'month={period.month}'
        d.mkdir(parents=True, exist_ok=True)
        return d / f'{self.quarter}.parquet'

    def write(self, period: pd.Timestamp, frame: pd.DataFrame) -> None:
        w = self._writers.get(period)
        if w is None:
            final = self._path(period)
            tmp = final.with_suffix(final.suffix + self.WRITING_SUFFIX)
            w = pq.ParquetWriter(tmp, self.schema, compression='snappy')
            self._writers[period] = w
            self._tmp[period] = tmp
            self.paths.append(final)
        w.write_table(pa.Table.from_pandas(frame, schema=self.schema, preserve_index=False))

    def close(self, commit: bool = True) -> None:
        """Close every writer; with commit, move the temp files into place, otherwise delete them."""
        for period, w in self._writers.items():
            try:
                w.close()
            except Exception:
                commit = False
        for period, tmp in self._tmp.items():
            if commit and tmp.exists():
                tmp.replace(self._path(period))
            elif tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    log.warning("Left a partial file behind: %s", tmp)
        self._writers.clear()
        self._tmp.clear()

    @staticmethod
    def partition_key(period: pd.Series) -> pd.Series:
        return period.dt.to_period('M').dt.to_timestamp()

    @classmethod
    def sweep(cls, root: Path) -> int:
        """Delete the temp files a crashed run left under root; returns how many."""
        if not root.exists():
            return 0
        n = 0
        for tmp in root.rglob(f'*{cls.WRITING_SUFFIX}'):
            try:
                tmp.unlink()
                n += 1
            except OSError:
                pass
        return n

def origination_path(orig_root: Path, quarter: str) -> Path:
    """The origination parquet file of a quarter: year=/quarter=/<quarter>.parquet."""
    per = pd.Period(quarter.upper(), freq='Q')
    return orig_root / f'year={per.year:04d}' / f'quarter={per.quarter}' / f'{quarter}.parquet'

def _prune_empty_dirs(root: Path) -> None:
    for d in sorted((d for d in root.rglob('*') if d.is_dir()), key=lambda d: len(d.parts), reverse=True):
        try:
            d.rmdir()
        except OSError:
            pass

def _drop_stale_files(root: Path, quarter: str, keep: Set[Path]) -> int:
    """Delete the quarter's files under root that this run did not write (months no longer in it), then prune empty
    directories."""
    if not root.exists():
        return 0
    removed = 0
    for path in root.rglob(f'{quarter}.parquet'):
        if path not in keep:
            try:
                path.unlink()
                removed += 1
            except OSError:
                log.warning("Left a partial file behind: %s", path)
    if removed:
        _prune_empty_dirs(root)
    return removed

def convert_quarter(src: QuarterSource, out_root: Path, chunk_size: int, min_period: Optional[pd.Timestamp] = None,
                    max_period: Optional[pd.Timestamp] = None) -> Dict[str, object]:
    """Convert one quarter: origination to one parquet file and performance, read in chunks and clipped to
    [min_period, max_period], to monthly partitions. Returns row and partition counts."""
    quarter = src.quarter
    with QuarterArchive(src) as arc:
        n_cols = _peek_columns(arc, 'origination')
        o_names = resolve_columns('origination', n_cols)
        with arc.open('origination') as fh:
            orig = pd.read_csv(fh, sep='|', header=None, names=o_names, dtype=str, low_memory=False, na_values=[''],
                               keep_default_na=False)
        orig = _coerce(orig)[[c for c in ORIG_COLS if c in o_names]]
        orig['origination_quarter'] = quarter

        dup = int(orig['loan_identifier'].duplicated().sum())
        if dup > 0:
            log.error("Duplicated loan identifiers: %s", dup)

        orig_file = origination_path(out_root / 'origination', quarter)
        orig_file.parent.mkdir(parents=True, exist_ok=True)
        text = orig.select_dtypes(['object']).columns
        if len(text):
            orig = orig.astype({c: 'string' for c in text})
        orig.to_parquet(orig_file, index=False, coerce_timestamps='us', allow_truncated_timestamps=True)

        n_cols = _peek_columns(arc, 'performance')
        p_names = resolve_columns('performance', n_cols)
        kept = [c for c in PERF_COLS if c in p_names] + ['dpd']
        schema = pa.schema([(c, _PERF_FIELDS[c]) for c in kept if c in _PERF_FIELDS])

        writer = PartitionWriter(out_root / 'performance', quarter, schema)
        n_read = n_kept = 0
        lo = hi = None
        ok = False
        try:
            with arc.open('performance') as fh:
                reader = pd.read_csv(fh, sep='|', header=None, names=p_names, dtype=str, chunksize=chunk_size,
                                     low_memory=False, on_bad_lines='warn', na_values=[''], keep_default_na=False)
                for i, chunk in enumerate(reader, 1):
                    n_read += len(chunk)
                    chunk = _coerce(chunk)
                    per = chunk[PARTITION_ON]
                    mask = per.notna()
                    if min_period is not None:
                        mask &= per >= min_period
                    if max_period is not None:
                        mask &= per <= max_period
                    chunk = chunk.loc[mask]
                    if chunk.empty:
                        continue
                    chunk = chunk[[c for c in schema.names]]
                    p = chunk[PARTITION_ON]
                    lo = p.min() if lo is None else min(lo, p.min())
                    hi = p.max() if hi is None else max(hi, p.max())
                    n_kept += len(chunk)
                    for period, part in chunk.groupby(PartitionWriter.partition_key(p), sort=False):
                        writer.write(period, part)
            ok = True
        finally:
            writer.close(commit=ok)

    stale = (_drop_stale_files(out_root / 'performance', quarter, set(writer.paths))
             + _drop_stale_files(out_root / 'origination', quarter, {orig_file}))
    if stale:
        log.info("Found stale files: %s", stale)
    if n_kept == 0:
        log.info("%s: no performance rows inside the observation window. "
                 "The vintage is kept for its origination rows only.", quarter)
    else:
        log.info("performance %s of %s rows -> %s partitions",
                 f"{n_kept:,}", f"{n_read:,}", len(writer.paths))

    return {'quarter': quarter,
            'origination_rows': int(len(orig)),
            'performance_rows': int(n_kept),
            'performance_rows_in_file': int(n_read),
            'min_period': None if lo is None else str(lo.date()),
            'max_period': None if hi is None else str(hi.date()),
            'partitions': len(writer.paths),
            'source': src.label}

def _hdfs_url(hdfs_dir: str, *parts: str) -> str:
    path = '/'.join((hdfs_dir.rstrip('/'), *parts))
    if '://' in path:
        return path
    return f'{settings.hdfs_uri.rstrip("/")}/{path.lstrip("/")}'

def _hdfs(*args: str) -> None:
    """Run `hdfs dfs` with the arguments, raising with its output when it fails."""
    cli = shutil.which(os.getenv("HDFS_CLI", 'hdfs'))
    if cli is None:
        raise RuntimeError(
            "The 'hdfs' command is not on PATH. Put the Hadoop client tools on "
            "PATH, point HDFS_CLI at the binary, or pass --no-hdfs to write "
            "local parquet only.")
    proc = subprocess.run([cli, "dfs", *args], capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise RuntimeError(f"hdfs dfs {' '.join(args)} failed "
                           f"(exit {proc.returncode}):\n{detail}")

def hdfs_prepare(hdfs_dir: str) -> None:
    _hdfs('-mkdir', '-p', *[_hdfs_url(hdfs_dir, g) for g in ('performance', 'origination')])

def hdfs_push_partitions(grain: str, part_files: Set[Path], hdfs_dir: str) -> None:
    """Copy the local partition files of a grain to the same year=/month= folders on HDFS."""
    if not part_files:
        return
    remote = _hdfs_url(hdfs_dir, grain)
    by_dir: Dict[str, List[Path]] = {}
    for f in part_files:
        remote_dir = f"{remote}/{f.parent.parent.name}/{f.parent.name}"
        by_dir.setdefault(remote_dir, []).append(f)

    _hdfs("-mkdir", "-p", *sorted(by_dir))
    for remote_dir, files in sorted(by_dir.items()):
        _hdfs("-put", "-f", *[str(f) for f in sorted(files)], f"{remote_dir}/")

def _period(value: Optional[str]) -> Optional[pd.Timestamp]:
    """The first day of a YYYY-MM month, or None."""
    if not value:
        return None
    return pd.Timestamp(pd.Period(str(value).replace('-', ''), freq='M').start_time)

def quarter_range(start: str, end: str) -> List[str]:
    """Every quarter from start to end, as 'YYYYQn'."""
    per = pd.period_range(start=start.upper(), end=end.upper(), freq='Q')
    return [f'{p.year}Q{p.quarter}' for p in per]

def _wanted(arg: Optional[str]) -> Optional[List[str]]:
    """The quarters asked for by --quarters: a comma-separated list or a start:end range."""
    if not arg:
        return None
    return quarter_range(*arg.split(':', 1)) if ':' in arg else [q.strip().upper() for q in arg.split(',')]

def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

    p = argparse.ArgumentParser(description='Ingestion')
    p.add_argument('command', choices=['verify', 'convert'])
    p.add_argument('--input-dir', default=str(settings.data_raw),
                   help='folder holding historical_data_YYYY.zip')
    p.add_argument('--out-dir', default=str(settings.data_curated))
    p.add_argument('--chunk_size', type=int, default=1_000_000)
    p.add_argument('--quarters', help='comma-separated list, or a range like 2004Q1:2025Q4')
    p.add_argument('--min-period', default='1999-01')
    p.add_argument('--max-period', default='2026-03')
    p.add_argument('--hdfs-dir', default=settings.hdfs_curated_dir)
    p.add_argument('--no-hdfs', action='store_true')
    a = p.parse_args()

    settings.ensure_dirs()
    in_dir, out_root = Path(a.input_dir), Path(a.out_dir)

    sources = discover(in_dir)
    if not sources:
        log.error(f"no sources found in {in_dir}")
        raise SystemExit(1)

    wanted = _wanted(a.quarters)
    available = sorted(sources)
    log.info("%s quarters available: %s .. %s", len(available), available[0], available[-1])
    problems = check_quarters(sources, wanted)
    for msg in problems:
        log.error(" %s", msg)

    todo = [q for q in available if wanted is None or q in wanted]
    if not todo:
        log.error("No quarters match --quarters %s", a.quarters)
        raise SystemExit(1)

    if a.command == 'verify':
        for q in todo:
            verify(sources[q])
        return

    if problems:
        log.error("Refusing to proceed: quarters named in --quarters are missing from "
                  "the download.")
        raise SystemExit(1)

    min_p, max_p = _period(a.min_period), _period(a.max_period)

    hdfs_dir = None if (a.no_hdfs or not settings.hdfs_enabled) else a.hdfs_dir
    try:
        if hdfs_dir:
            hdfs_prepare(hdfs_dir)
        else:
            log.warning("HDFS push disabled -- writing local parquet only.")

        swept = PartitionWriter.sweep(out_root / "performance")

        for i, q in enumerate(todo, 1):
            perf_files: Set[Path] = set()
            orig_files: Set[Path] = set()

            log.info("[%s/%s] %s", i, len(todo), sources[q].label)
            stats = convert_quarter(sources[q], out_root, a.chunk_size, min_p, max_p)
            orig_files.add(origination_path(out_root / "origination", q))
            perf_root = out_root / "performance"
            if perf_root.exists():
                perf_files.update(perf_root.rglob(f"{q}.parquet"))

            if hdfs_dir:
                log.info("Pushing to %s", _hdfs_url(hdfs_dir))
                hdfs_push_partitions("performance", perf_files, hdfs_dir)
                hdfs_push_partitions("origination", orig_files, hdfs_dir)
    except RuntimeError as exc:
        log.error("%s", exc)
        raise SystemExit(1)

if __name__ == '__main__':
    main()