import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from optbinning import OptimalBinning
from pyspark.sql import Column, DataFrame, SparkSession, functions as F
from pyspark.sql.types import NumericType, StringType
from helpers.config import settings
from helpers.period import (AnalysisPeriod, add_arguments as add_period_arguments, from_args as period_from_args,
                            month_partitions, read_period, write_by_month)
from helpers.report import md_table, write_report
from helpers.spark_io import curated_path, resolve_path, spark_session

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s | %(message)s')
log = logging.getLogger('woe_binning_main')

TARGET = 'label'
WEIGHT = 'sample_weight'
KEYS: tuple = ('loan_identifier', 'period')
NOT_FEATURES: tuple = KEYS + (TARGET, 'observation_date', 'year', 'month')

DEV_END = '2022-09'

GOODS_PER_BAD = 2

WOE_SUFFIX = '_woe'


def pick_features(df: DataFrame, include: Optional[List[str]], exclude: List[str]) -> Dict[str, str]:
    """The features to bin, numeric as 'numerical' and strings as 'categorical', restricted to include and without
    exclude."""
    kinds: Dict[str, str] = {}
    for f in df.schema.fields:
        if f.name in NOT_FEATURES or f.name in exclude:
            continue
        if include and f.name not in include:
            continue
        if isinstance(f.dataType, NumericType):
            kinds[f.name] = 'numerical'
        elif isinstance(f.dataType, StringType):
            kinds[f.name] = 'categorical'
        else:
            log.info('Skipping %s: %s is neither numerical nor categorical', f.name, f.dataType.simpleString())
    if include:
        missing = sorted(set(include) - set(kinds))
        if missing:
            raise SystemExit(f"--features not in the training set (or not binnable): {', '.join(missing)}")
    return kinds

def build_sample(dev: DataFrame, period: AnalysisPeriod, features: Dict[str, str], goods_per_bad: float,
                seed: int, out: str) -> Dict[str, object]:
    """Write the binning sample: every bad of the development window and enough goods for goods_per_bad goods per
    bad, each good weighted by the inverse of its sampling fraction so bins are sized on the population. Returns
    the sample's record."""
    c = dev.agg(F.count(F.lit(1)).alias('n'), F.sum(TARGET).alias('bad')).collect()[0]
    n, bad = int(c['n']), int(c['bad'] or 0)
    good = n - bad
    if not bad or not good:
        raise SystemExit(f'Unable to bin because development window has {bad:,} bads and {good:,} goods')
    f = min(1, goods_per_bad * bad / good)
    log.info('Development window %s: %s rows, %s bads (%.3f%%). Keeping every bad and %.4f of goods.',
             period.label, f'{n:,}', f'{bad:,}', 100 * bad / n, f)

    cols = [F.col(c).cast('double').alias(c) if k == 'numerical' else F.col(c) for c, k in features.items()]
    sample = (dev.sampleBy(TARGET, {0:f, 1:1}, seed=seed)
              .select(F.col(TARGET).cast('tinyint').alias(TARGET),
                      F.when(F.col(TARGET) == 1, F.lit(1)).otherwise(F.lit(1 / f)).alias(WEIGHT),
                      *cols))
    sample.coalesce(16).write.mode('overwrite').parquet(out)

    k = dev.sparkSession.read.parquet(out).agg(F.count(F.lit(1)).alias('n'),
                                               F.sum(TARGET).alias('bad')).collect()[0]
    kept, kept_bad = int(k['n']), int(k['bad'] or 0)
    log.info('Sample: %s rows, %.1f%% bad, at %s', f'{kept:,}', 100 * kept_bad / max(kept, 1), out)

    return {'period': period.to_dict(), 'dev_rows': n, 'dev_bads': bad, 'dev_bad_rate': bad / n, 'good_fraction': f,
            'goods_per_bad': goods_per_bad, 'seed': seed, 'sample_rows': kept,
            'sample_bads': kept_bad, 'sample_path': out}

def _num(v) -> Optional[float]:
    """v as a float, or None when it is not finite."""
    v = float(v)
    return v if np.isfinite(v) else None

def binning_spec(optb: OptimalBinning, kind: str) -> Dict[str, object]:
    """A fitted binning as JSON: per bin its count, event rate, WoE and IV, the missing-value WoE, and the splits
    (numerical) or category groups (categorical)."""
    bt = optb.binning_table.build()
    n_bins = len(optb.splits) + (0 if kind == 'categorical' else 1)
    regular = bt.iloc[:n_bins]
    bins = [
        {
            'bin': str(r['Bin']),
            'count': _num(r['Count']),
            'event_rate': _num(r['Event rate']),
            'woe': _num(r['WoE']),
            'iv': _num(r['IV'])
        }
        for _, r in regular.iterrows()
    ]
    spec: Dict[str, object] = {
        'dtype': kind, 'status': optb.status,
        'iv': _num(bt.loc['Totals', 'IV']),
        'bins': bins,
        'woe_missing': _num(bt.iloc[n_bins + 1]['WoE']),
        'woe_unknown': 0
    }
    if kind == 'numerical':
        spec['splits'] = [float(s) for s in optb.splits]
    else:
        spec['categories'] = [[str(v) for v in group] for group in optb.splits]
    return spec

def fit_features(spark: SparkSession, sample_path: str, features: Dict[str, str],
                params: Dict[str, object]) -> Dict[str, dict]:
    """Fit a weighted optimal binning per feature on the sample, one column at a time; a feature that is all
    missing or fails to fit is recorded with its status instead."""
    sample = spark.read.parquet(sample_path)
    specs: Dict[str, dict] = {}
    for i, (c, kind) in enumerate(features.items(), 1):
        t = time.time()
        pdf = sample.select(c, TARGET, WEIGHT).toPandas()
        y, w = pdf[TARGET].to_numpy(), pdf[WEIGHT].to_numpy()
        x = pdf[c].to_numpy(dtype=float, na_value=np.nan) if kind == 'numerical' else pdf[c].to_numpy(dtype=object)
        del pdf
        if pd.isna(x).all():
            specs[c] = {'dtype': kind, 'status': 'ALL_MISSING'}
            log.warning('[%s/%s] %s is missing on every sampled row', i, len(features), c)
            continue
        kw = dict(params)
        if kind == 'categorical':
            kw.pop('monotonic_trend', None)
        try:
            optb = OptimalBinning(name=c, dtype=kind, **kw).fit(x, y, sample_weight=w)
            specs[c] = binning_spec(optb, kind)
            log.info('[%s/%s] %s (%s): %s, %s bins, IV %.4f in %.1fs', i, len(features), c, kind,
                     optb.status, len(specs[c]['bins']), specs[c]['iv'] or 0, time.time() - t)
        except Exception as exc:
            specs[c] = {'dtype': kind, 'status': f'FAILED: {exc}'}
            log.warning('[%s/%s] %s failed: %s', i, len(features), c, exc)
    return specs

def woe_column(c: str, spec: dict) -> Column:
    """A Spark expression mapping a feature to its bin's WoE: missing to the missing WoE, an unseen category to
    woe_unknown."""
    x = F.col(c)
    woe = [b['woe'] if b['woe'] is not None else spec['woe_unknown'] for b in spec['bins']]
    if spec['dtype'] == 'numerical':
        x = x.cast('double')
        expr = F.when(x.isNull() | F.isnan(x), F.lit(spec['woe_missing']))
        for edge, v in zip(spec['splits'], woe):
            expr = expr.when(x < F.lit(edge), F.lit(v))
        expr = expr.otherwise(F.lit(woe[-1]))
    else:
        pairs = [F.lit(v) for group, wv in zip(spec['categories'], woe)
                 for cat in group for v in (cat, wv)]
        lookup = F.create_map(*pairs)[x] if pairs else F.lit(None)
        expr = F.when(x.isNull(), F.lit(spec['woe_missing'])).otherwise(
            F.coalesce(lookup, F.lit(spec['woe_unknown']))
        )
    return expr.cast('double').alias(f'{c}{WOE_SUFFIX}')

def apply_woe(spark: SparkSession, training_set: str, specs: Dict[str, dict], out: str,
              files_per_month: int = 0, resume: bool = False) -> Dict[str, object]:
    """Write the WoE of every binned feature for every month of the training set, and check the row count is
    conserved."""
    usable = {c: s for c, s in specs.items() if 'bins' in s}
    parts = month_partitions(spark, training_set)
    if not parts:
        raise SystemExit(f'{training_set} is not partitioned by year/month')
    root = resolve_path(training_set)
    months = [(y, m) for y, m, _ in parts]

    cols = spark.read.parquet(parts[0][2]).columns
    missing = sorted(set(usable) - set(cols))
    if missing:
        raise SystemExit(f'Binned features not in {training_set}: {', '.join(missing)}')
    woe_cols = [woe_column(c, s) for c, s in usable.items()]
    log.info('WoE for %s features over %s months', len(usable), len(months))
    run = write_by_month(spark, training_set, out,
                         lambda df, y, m: df.select(*KEYS, TARGET, *woe_cols),
                         resume=resume, files_per_month=files_per_month)
    written = run['months_written']

    src = spark.read.parquet(root).count()
    dst = spark.read.parquet(out).where(
        (F.col('year') * 100 + F.col('month')).isin([y * 100 + m for y, m in months])
    ).count()
    if src != dst:
        log.warning('ROW COUNT MISMATCH: %s training rows but %s WoE rows (%+d)',
                    f'{src:,}', f'{dst:,}', dst - src)
    else:
        log.info('Row count conserved: %s rows', f'{src:,}')
    return {'features': len(usable), 'months': len(months), 'months_written': written, 'training_rows': src,
            'woe_rows': dst}

def write_doc(path: Path, specs: Dict[str, dict], meta: Dict[str, object]) -> None:
    rows = [{'feature': c, 'dtype': s['dtype'], 'status': s['status'],
             'bins': len(s.get('bins', [])), 'IV': s.get('iv')} for c, s in specs.items()]
    iv = pd.DataFrame(rows).sort_values('IV', ascending=False, na_position='last').reset_index(drop=True)
    s = meta['sample']
    preamble = (f"Optimal binning (optbinning, weighted) fitted on the development window "
                f"**{meta['period']['analysis_period']}**: {s['dev_rows']:,} rows, {s['dev_bads']:,} bads "
                f"({s['dev_bad_rate']:.3%}). The fit used every bad and {s['good_fraction']:.4f} of goods "
                f"({s['sample_rows']:,} rows), each good weighted by 1/{s['good_fraction']:.4f} so the bins "
                f"are sized on the population. "
                f"Counts and event rates below are weighted estimates of the development window.")
    sections = [('Information value', md_table(iv, floatfmt='{:.4f}'))]
    for c, spec in sorted(specs.items(), key=lambda kv: -(kv[1].get('iv') or -1)):
        if 'bins' not in spec:
            continue
        t = pd.DataFrame(spec['bins'] + [{'bin': 'Missing', 'woe': spec['woe_missing']}])
        t['count'] = t['count'].round().astype('Int64')
        sections.append((f"{c} (IV {spec['iv'] or 0:.4f})", md_table(t, floatfmt='{:.4f}')))
    write_report(path, 'WoE Binning', 'src/jobs/woe_binning_main.py', sections, preamble)

def main() -> None:
    p = argparse.ArgumentParser(description='woe_binning_main')
    add_period_arguments(p)
    p.set_defaults(period_end=DEV_END)
    p.add_argument('--stage', choices=('all', 'sample', 'fit', 'apply'), default='all',
                   help='sample -> fit -> apply')
    p.add_argument('--training-set', default=curated_path('training_set'))
    p.add_argument('--sample-out', default=curated_path('binning_sample'))
    p.add_argument('--sample-meta', default=str(settings.data_curated / 'binning_sample_meta.json'))
    p.add_argument('--binning-json', default=str(settings.model_dir / 'woe_binning.json'))
    p.add_argument('--doc', default=str(Path(__file__).resolve().parents[2] / 'docs' / 'woe_binning.md'))
    p.add_argument('--out', default=curated_path('training_set_woe'))
    p.add_argument('--files-per-month', type=int, default=0,
                   help='apply: at most this many files per month (0 = one per input split)')
    p.add_argument('--resume', action='store_true', help='apply: skip months already in --out')
    p.add_argument('--features', help='comma-separated; default: every numeric and string column')
    p.add_argument('--exclude', default='', help='comma-separated columns to leave out')
    p.add_argument('--goods-per-bad', type=float, default=GOODS_PER_BAD)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-n-bins', type=int, default=10)
    p.add_argument('--min-bin-size', type=float, default=0.01)
    p.add_argument('--min-prebin-size', type=float, default=0.01)
    p.add_argument('--monotonic-trend', default='auto_asc_desc',
                   help="numerical features only; 'none' to switch off")
    p.add_argument('--cat-cutoff', type=float, default=0.01)
    p.add_argument('--time-limit', type=int, default=300, help='solver seconds per feature')
    a = p.parse_args()

    period = period_from_args(a)
    sample_path = a.sample_out
    meta_file = Path(a.sample_meta)
    json_path = Path(a.binning_json)
    include = [c.strip() for c in a.features.split(',') if c.strip()] if a.features else None
    exclude = [c.strip() for c in a.exclude.split(',') if c.strip()]
    run_sample, run_fit, run_apply = (a.stage in ('all', s) for s in ('sample', 'fit', 'apply'))

    meta: Dict[str, object] = {}
    features: Dict[str, str] = {}

    spark = spark_session('woe_binning', conf={'spark.driver.maxResultSize': '4g',
                                               'spark.sql.execution.arrow.pyspark.enabled': 'true'})

    try:
        if run_sample:
            dev = read_period(spark, a.training_set, period)
            features = pick_features(dev, include, exclude)
            log.info('%s features to bin', len(features))
            meta['sample'] = build_sample(dev, period, features, a.goods_per_bad, a.seed, sample_path)
            meta['features'] = features
            meta_file.parent.mkdir(parents=True, exist_ok=True)
            meta_file.write_text(json.dumps(meta, indent=2))

        if run_fit:
            if not meta:
                if not meta_file.exists():
                    raise SystemExit(f'No sample record at {meta_file}')
                meta = json.loads(meta_file.read_text())
                if meta['sample']['sample_path'] != sample_path:
                    log.warning('%s describes the sample at %s, not --sample-out %s; fitting on %s',
                                meta_file, meta['sample']['sample_path'], sample_path, meta['sample']['sample_path'])
                sample_path = meta['sample']['sample_path']
            features = meta['features']
            if include or exclude:
                features = {c: k for c, k in features.items()
                            if (not include or c in include) and c not in exclude}
            params = dict(solver='cp', max_n_bins=a.max_n_bins, min_bin_size=a.min_bin_size,
                          min_prebin_size=a.min_prebin_size, cat_cutoff=a.cat_cutoff, time_limit=a.time_limit)
            if a.monotonic_trend.lower() != 'none':
                params['monotonic_trend'] = a.monotonic_trend
            specs = fit_features(spark, sample_path, features, params)
            failed = [c for c, s in specs.items() if 'bins' not in s]
            if failed:
                log.warning('No binning for: %s', ', '.join(failed))
            if meta['sample'].get('period', {}).get('analysis_period') != period.label and a.stage == 'fit':
                log.info('Fitting on the sample drawn for %s', meta['sample']['period']['analysis_period'])
            meta.update({'period': meta['sample']['period'], 'target': TARGET, 'params': params,
                         'generated': datetime.now().isoformat(timespec='seconds'),
                         'source': 'src/jobs/08_lr_woe_binning_main.py'})
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps({'meta': meta, 'features': specs}, indent=2))
            log.info('Binning for %s features in %s', len(specs) - len(failed), json_path)
            write_doc(Path(a.doc), specs, meta)
            log.info('Report in %s', a.doc)

        if run_apply:
            if not json_path.exists():
                raise SystemExit(f'No binning at {json_path}')
            specs = json.loads(json_path.read_text())['features']
            if include:
                specs = {c: s for c, s in specs.items() if c in include}
            stats = apply_woe(spark, a.training_set, specs, a.out, a.files_per_month, a.resume)
            log.info('Apply: %s', stats)
    finally:
        spark.stop()


if __name__ == '__main__':
    main()