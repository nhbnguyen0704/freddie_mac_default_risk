import argparse
import json
import logging
import time
from datetime import datetime
from functools import reduce
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession, functions as F
from helpers.config import settings
from helpers.period import (AnalysisPeriod, add_arguments as add_period_arguments, from_args as period_from_args,
                            month_partitions, read_period, write_by_month)
from helpers.report import md_table, write_report
from helpers.spark_io import curated_path, spark_session

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s | %(message)s')
log = logging.getLogger('vif_selection_main')

TARGET = 'label'
KEYS: tuple = ('loan_identifier', 'period')
WOE_SUFFIX = '_woe'

DEV_END = '2022-09'

MIN_IV = 0.01
CORR_THRESHOLD = 0.8
VIF_THRESHOLD = 5.0

MIN_VARIANCE = 1e-12

def moments(df: DataFrame, cols: List[str]) -> Tuple[int, np.ndarray, np.ndarray]:
    """Row count, column sums and the cross-product matrix of cols, in one pass over df."""
    p = len(cols)

    def partial(batches):
        n, s, g = 0, np.zeros(p), np.zeros((p, p))
        for pdf in batches:
            x = pdf.to_numpy(dtype=np.float64)
            n += len(x)
            s += x.sum(axis=0)
            g += x.T @ x
        yield pd.DataFrame({'n': [n], 's': [s.tolist()], 'g': [g.ravel().tolist()]})

    parts = (df.select(*cols)
             .mapInPandas(partial, 'n long, s array<double>, g array<double>')
             .collect())
    n = sum(int(r['n']) for r in parts)
    s = reduce(np.add, (np.array(r['s']) for r in parts))
    g = reduce(np.add, (np.array(r['g']) for r in parts)).reshape(p, p)
    return n, s, g

def correlation(n: int, s: np.ndarray, g: np.ndarray, cols: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    """The correlation matrix and the variances from the moments."""
    cov = (g - np.outer(s, s) / n) / (n-1)
    var = pd.Series(np.diag(cov), index=cols)
    sd = np.sqrt(np.clip(np.diag(cov), MIN_VARIANCE, None))
    corr = pd.DataFrame(cov / np.outer(sd, sd), index=cols, columns=cols)
    return corr, var

def corr_filter(corr: pd.DataFrame, iv: pd.Series, threshold: float) -> Tuple[List[str], pd.DataFrame]:
    """While any pair has |corr| above threshold, drop the feature with the highest max |corr|, then the highest
    median |corr|, then the lowest IV. Returns the kept features and the rounds."""
    a = corr.abs().mask(np.eye(len(corr), dtype=bool))
    keep = list(corr.columns)
    steps: List[dict] = []
    while len(keep) > 1:
        sub = a.loc[keep, keep]
        mx = sub.max()
        if mx.max() <= threshold:
            break
        cand = mx.index[mx > threshold]
        rank = pd.DataFrame({'max_corr': mx[cand], 'median_corr': sub[cand].median(), 'iv': iv[cand]})
        worst = rank.sort_values(['max_corr', 'median_corr', 'iv'], ascending=[False, False, True]).index[0]
        partner = sub[worst].idxmax()
        steps.append({'round': len(steps) + 1, 'dropped': worst, 'max_corr': float(rank.loc[worst, 'max_corr']),
                      'with': partner, 'corr': float(corr.loc[worst, partner]),
                      'median_corr': float(rank.loc[worst, 'median_corr']), 'iv': float(iv[worst]),
                      'partner_iv': float(iv[partner]), 'features_left': len(keep) - 1})
        log.info('Corr round %s: drop %s (|corr| %.3f with %s, median %.3f, IV %.4f vs %.4f)', len(steps), worst,
                 rank.loc[worst, 'max_corr'], partner, rank.loc[worst, 'median_corr'], iv[worst], iv[partner])
        keep.remove(worst)
    return keep, pd.DataFrame(steps)

def vifs(r: np.ndarray) -> np.ndarray:
    """The variance inflation factor of each feature, from the correlation matrix."""
    p = len(r)
    out = np.empty(p)
    for j in range(p):
        o = np.arange(p) != j
        if not o.any():
            out[j] = 1
            continue
        b, *_ = np.linalg.lstsq(r[np.ix_(o, o)], r[o, j], rcond=None)
        r2 = float(r[j, o] @ b)
        out[j] = np.inf if r2 >= 1 - 1e-10 else 1.0 / (1.0 - r2)
    return out

def vif_filter(corr: pd.DataFrame, iv: pd.Series,
               threshold: float) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    """While any VIF is above threshold, drop the highest VIF (lowest IV on a tie). Returns the kept features, the
    rounds and the final VIF table."""
    keep = list(corr.columns)
    steps: List[dict] = []
    while True:
        v = pd.Series(vifs(corr.loc[keep, keep].to_numpy()), index=keep)
        if len(keep) <= 1 or v.max() <= threshold:
            break
        rank = pd.DataFrame({'vif': v, 'iv': iv[keep]})
        rank['_v'] = rank['vif'].round(6)
        worst = rank.sort_values(['_v', 'iv'], ascending=[False, True]).index[0]
        tied = int((rank['_v'] == rank.loc[worst, '_v']).sum())
        steps.append({'round': len(steps) + 1, 'dropped': worst, 'vif': float(v[worst]),
                      'iv': float(iv[worst]), 'tied_at_max': tied, 'features_left': len(keep) - 1})
        log.info('VIF round %s: drop %s (VIF %.2f, IV %.4f%s)', len(steps), worst, v[worst], iv[worst],
                 f', tie of {tied} broken by IV' if tied > 1 else '')
        keep.remove(worst)
    final = pd.DataFrame({'vif': v, 'iv': iv[keep]}).sort_values('vif', ascending=False)
    return keep, pd.DataFrame(steps), final

def write_selected(spark: SparkSession, woe_path: str, kept: List[str], train: AnalysisPeriod, out: str,
                   files_per_month: int, resume: bool) -> Dict[str, object]:
    """Write the kept WoE columns month by month, with a sample column: 'train' up to the end of the train period,
    'test' after it."""
    end = train.end.year * 100 + train.end.month

    def build(df: DataFrame, y: int, m: int) -> DataFrame:
        split = 'train' if y * 100 + m <= end else 'test'
        return df.select(*KEYS, TARGET, F.lit(split).alias('sample'), *kept)

    run = write_by_month(spark, woe_path, out, build, resume=resume, files_per_month=files_per_month)
    parts = month_partitions(spark, woe_path)
    n_train = sum(1 for y, m, _ in parts if y * 100 + m <= end)
    return {**run, 'train_months': n_train, 'test_months': len(parts) - n_train}


def write_doc(path: Path, result: dict) -> None:
    m = result['meta']
    n_corr, n_vif = len(result['corr_steps']), len(result['vif_steps'])
    preamble = (f"Feature selection on the WoE features, fitted on the **train** months only "
                f"({m['train_period']['analysis_period']}, {m['train_rows']:,} rows) and applied unchanged "
                f"to the test months. Starting from {m['binned']} binned features:\n\n"
                f"0. **IV**: drop IV < {m['min_iv']:g} ({len(result['low_iv'])} dropped) and features with a "
                f"constant WoE on train ({len(result['constant'])} dropped).\n"
                f"1. **Correlation**: while any pair has |corr| > {m['corr_threshold']:g}, drop the feature with "
                f"the highest max |corr|, then the highest median |corr|, then the lowest IV "
                f"({n_corr} dropped).\n"
                f"2. **VIF**: while any VIF > {m['vif_threshold']:g}, drop the highest VIF, lowest IV on a tie "
                f"({n_vif} dropped).\n\n"
                f"**{len(result['kept'])} of {m['binned']} features kept.**")

    def rounds(rows, what):
        return md_table(pd.DataFrame(rows), floatfmt='{:.4f}') if rows else f'_(none: {what})_'

    sections = [('Kept features', md_table(pd.DataFrame(result['final']), floatfmt='{:.4f}')),
                ('Correlation filter', rounds(result['corr_steps'], 'no pair was above the threshold')),
                ('VIF filter', rounds(result['vif_steps'], 'every VIF was already within the threshold'))]
    if result['low_iv']:
        low = pd.DataFrame(result['low_iv']).sort_values('iv', ascending=False)
        sections.append((f"Dropped for IV < {m['min_iv']:g}", md_table(low, floatfmt='{:.4f}')))
    if result['constant']:
        sections.append(('Dropped as constant',
                         'Constant WoE on the train months (one bin), so no correlation or VIF: '
                         + ', '.join(f'`{c}`' for c in result['constant'])))
    write_report(path, 'Feature Selection: Correlation and VIF', 'src/jobs/09_lr_vif_selection_main.py',
                 sections, preamble)

def main() -> None:
    p = argparse.ArgumentParser(description='lr_vif_selection_main')
    add_period_arguments(p)
    p.set_defaults(period_end=DEV_END)
    p.add_argument('--stage', choices=('all', 'select', 'write'), default='all')
    p.add_argument('--woe', default=curated_path('training_set_woe'),
                   help='output of 08_lr_woe_binning_main')
    p.add_argument('--binning-json', default=str(settings.model_dir / 'woe_binning.json'))
    p.add_argument('--min-iv', type=float, default=MIN_IV)
    p.add_argument('--corr-threshold', type=float, default=CORR_THRESHOLD)
    p.add_argument('--vif-threshold', type=float, default=VIF_THRESHOLD)
    p.add_argument('--corr-csv', default=str(settings.model_dir / 'woe_corr_train.csv'))
    p.add_argument('--reuse-corr', action='store_true')
    p.add_argument('--selection-json', default=str(settings.model_dir / 'vif_selection.json'))
    p.add_argument('--doc', default=str(Path(__file__).resolve().parents[2] / 'docs' / 'vif_selection.md'))
    p.add_argument('--out', default=curated_path('training_set_woe_selected'))
    p.add_argument('--files-per-month', type=int, default=0, help='0 = one per input split')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()

    train = period_from_args(a)
    sel_path = Path(a.selection_json)
    corr_path = Path(a.corr_csv)
    spark = spark_session('vif_selection', conf={'spark.sql.execution.arrow.pyspark.enabled': 'true'})
    try:
        if a.stage in ('all', 'select'):
            specs = json.loads(Path(a.binning_json).read_text())['features']
            iv = pd.Series({f'{c}{WOE_SUFFIX}': s.get('iv') or 0 for c, s in specs.items() if 'bins' in s})
            low_iv = iv[iv < a.min_iv].sort_values()
            log.info('IV filter: %s of %s binned features have IV >= %g',
                     len(iv) - len(low_iv), len(iv), a.min_iv)
            if len(low_iv):
                log.info('Dropped for IV < %g: %s', a.min_iv,
                         ', '.join(f'{c} ({v:.4f})' for c, v in low_iv.items()))
            strong = [c for c in iv.index if c not in low_iv.index]

            if a.reuse_corr:
                if not corr_path.exists():
                    raise SystemExit(f'No correlation matrix at {corr_path}: run without --reuse-corr')
                corr = pd.read_csv(corr_path, index_col=0)
                var, n = corr.pop('_variance'), int(corr.pop('_n').iloc[0])
                log.info('Reusing the correlation matrix in %s', corr_path)
                missing = sorted(set(strong) - set(corr.columns))
                if missing:
                    log.warning('Not in the saved matrix, left out: %s', ', '.join(missing))
            else:
                woe = read_period(spark, a.woe, train)
                cols = [c for c in woe.columns if c in strong]
                absent = sorted(set(strong) - set(cols))
                if absent:
                    log.warning('In the binning but not in %s: %s', a.woe, ', '.join(absent))
                t = time.time()
                n, s, g = moments(woe, cols)
                corr, var = correlation(n, s, g, cols)
                log.info('Correlation of %s features over %s train rows in %.0fs', len(cols), f'{n:,}',
                         time.time() - t)
                corr_path.parent.mkdir(parents=True, exist_ok=True)
                corr.assign(_variance=var, _n=n).to_csv(corr_path)

            var = var[[c for c in var.index if c in strong]]
            constant = sorted(var.index[var < MIN_VARIANCE])
            if constant:
                log.info('Constant on train, dropped before the loop: %s', ', '.join(constant))
            live = [c for c in corr.columns if c in strong and c not in constant]
            after_corr, corr_steps = corr_filter(corr.loc[live, live], iv, a.corr_threshold)
            log.info('Correlation filter: %s of %s features kept', len(after_corr), len(live))
            kept, vif_steps, final = vif_filter(corr.loc[after_corr, after_corr], iv, a.vif_threshold)
            final['max_abs_corr'] = corr.loc[kept, kept].abs().mask(np.eye(len(kept), dtype=bool)).max()
            log.info('VIF filter: %s of %s features kept; max VIF %.2f', len(kept), len(after_corr),
                     final['vif'].max())

            result = {
                'meta': {
                    'train_period': train.to_dict(), 'train_rows': n, 'min_iv': a.min_iv,
                    'corr_threshold': a.corr_threshold, 'vif_threshold': a.vif_threshold,
                    'binned': len(iv), 'candidates': len(live),
                    'corr_rule': 'highest max |corr|, then highest median |corr|, then lowest IV',
                    'vif_rule': 'highest VIF, lowest IV on a tie',
                    'woe': a.woe, 'binning_json': a.binning_json, 'corr_csv': str(corr_path),
                    'generated': datetime.now().isoformat(timespec='seconds'),
                    'source': 'src/jobs/09_lr_vif_selection_main.py'
                },
                'kept': kept,
                'low_iv': [{'feature': c, 'iv': float(v)} for c, v in low_iv.items()],
                'constant': constant,
                'after_corr': after_corr,
                'corr_steps': corr_steps.to_dict('records'),
                'vif_steps': vif_steps.to_dict('records'),
                'final': final.reset_index(names='feature').to_dict('records'),
            }
            sel_path.parent.mkdir(parents=True, exist_ok=True)
            sel_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=float))
            write_doc(Path(a.doc), result)
            log.info('Selection in %s, report in %s', sel_path, a.doc)

        if a.stage in ('all', 'write'):
            if not sel_path.exists():
                raise SystemExit(f'No selection at {sel_path}: run --stage select first.')
            result = json.loads(sel_path.read_text())
            if result['meta']['train_period']['analysis_period'] != train.label:
                log.warning('The selection was made on %s, not %s; the split column follows the selection',
                            result['meta']['train_period']['analysis_period'], train.label)
                train = AnalysisPeriod(pd.Timestamp(result['meta']['train_period']['analysis_start']),
                                       pd.Timestamp(result['meta']['train_period']['analysis_end']))
            stats = write_selected(spark, a.woe, result['kept'], train, a.out, a.files_per_month, a.resume)
            log.info('Written to %s: %s', a.out, stats)
    finally:
        spark.stop()

if __name__ == '__main__':
    main()