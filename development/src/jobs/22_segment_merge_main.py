import argparse
import importlib
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from helpers.config import settings
from helpers.gbm import SPLIT, average_precision, load_split, with_split
from helpers.report import md_table, write_report
from helpers.spark_io import curated_path, resolve_path, spark_session

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s | %(message)s')
log = logging.getLogger('segment_merge_main')

g15 = importlib.import_module('jobs.15_gbm_tuning_main')
g20 = importlib.import_module('jobs.20_regional_features_main')

CLEAN_MODEL = 'clean_regional'
DELINQUENT_MODEL = f'all_rows_{g20.VERSION}'
OFFSETS = np.round(np.arange(-3.0, 3.0001, 0.1), 2)

os.environ.setdefault('PYSPARK_PYTHON', sys.executable)


def score_tables(spark, a, cfg: dict, feats, reg, kinds, cats) -> pd.DataFrame:
    """(version, split, m) -> n, bad: clean_regional on the clean valid / test rows,
    the current model on the delinquent ones."""
    df = with_split(spark.read.parquet(resolve_path(a.training_set)), cfg).where(F.col(SPLIT).isin('valid', 'test'))
    info = json.loads((Path(a.gbm_models_dir) / f'{g20.VERSION}.json').read_text())
    fitted = json.loads((Path(a.models_dir) / 'fitted.json').read_text())[CLEAN_MODEL]
    t = time.time()
    d = g15.score_all(df.where(~g20.clean()), {DELINQUENT_MODEL: (
        (Path(a.gbm_models_dir) / f'{g20.VERSION}.ubj').read_bytes(), int(info['best_iteration']))},
        feats, kinds, cats).assign(segment='delinquent')
    log.info('Delinquent rows scored in %.0f min', (time.time() - t) / 60)
    t = time.time()
    c = g15.score_all(g20.with_regional(spark, df.where(g20.clean()), a),
                      {CLEAN_MODEL: (Path(fitted['path']).read_bytes(), int(fitted['best_iteration']))},
                      feats + reg, kinds, cats).assign(segment='clean')
    log.info('Clean rows scored in %.0f min', (time.time() - t) / 60)
    return pd.concat([c, d], ignore_index=True)

def platt(t: pd.DataFrame) -> Tuple[float, float]:
    """p = sigmoid(a + b m), fitted on a (m, n, bad) table."""
    x = np.r_[t['m'], t['m']].reshape(-1, 1)
    y = np.r_[np.ones(len(t)), np.zeros(len(t))]
    w = np.r_[t['bad'], t['n'] - t['bad']].astype(float)
    keep = w > 0
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(x[keep], y[keep], sample_weight=w[keep])
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])

def merged_ap(c: pd.DataFrame, d: pd.DataFrame, sc: np.ndarray, sd: np.ndarray) -> float:
    """AP of the clean and delinquent rows ranked together on the given scores."""
    return average_precision(np.r_[sc, sd], np.r_[c['bad'], d['bad']].astype(float),
                             np.r_[c['n'], d['n']].astype(float))

def merge(s: pd.DataFrame) -> dict:
    """Two-segment scoring: job 20's clean_regional on clean rows and job 15's recency_12 on delinquent rows,
    merged into one ranking. The two models' margins are on different scales (different training rows), so the
    merge is measured raw, after a per-segment Platt fit on valid, after one fitted on test (an oracle), and
    with a sweep of offsets added to the clean margin (an upper bound for any shift, chosen on test). Also the
    AP within each segment, the bad rates and the calibration (mean predicted against actual bad rate) per
    segment."""
    tab = {(seg, sp): g.reset_index(drop=True) for (seg, sp), g in s.groupby(['segment', SPLIT])}
    seg_ap = {f'{sp}/{seg}': average_precision(g['m'].to_numpy(), g['bad'].to_numpy(float), g['n'].to_numpy(float))
              for (seg, sp), g in tab.items()}
    rates = {f'{sp}/{seg}': float(g['bad'].sum() / g['n'].sum()) for (seg, sp), g in tab.items()}
    pc, pd_ = platt(tab['clean', 'valid']), platt(tab['delinquent', 'valid'])
    oc, od = platt(tab['clean', 'test']), platt(tab['delinquent', 'test'])
    out = {'segment_ap': seg_ap, 'bad_rate': rates,
           'platt_valid': {'clean': pc, 'delinquent': pd_}, 'platt_test': {'clean': oc, 'delinquent': od}}
    for sp in ('valid', 'test'):
        c, d = tab['clean', sp], tab['delinquent', sp]
        mc, md = c['m'].to_numpy(), d['m'].to_numpy()
        r = {'raw': merged_ap(c, d, mc, md),
             'platt_valid': merged_ap(c, d, pc[0] + pc[1] * mc, pd_[0] + pd_[1] * md),
             'platt_oracle': merged_ap(c, d, oc[0] + oc[1] * mc, od[0] + od[1] * md)}
        sweep = {float(o): merged_ap(c, d, mc + o, md) for o in OFFSETS}
        best = max(sweep, key=sweep.get)
        r |= {'offset_best': sweep[best], 'offset_best_at': best, 'sweep': sweep}
        r['mean_pred'] = {'clean_raw': float(np.sum(expit(mc) * c['n']) / c['n'].sum()),
                          'clean_platt_valid': float(np.sum(expit(pc[0] + pc[1] * mc) * c['n']) / c['n'].sum()),
                          'delinquent_raw': float(np.sum(expit(md) * d['n']) / d['n'].sum()),
                          'delinquent_platt_valid': float(np.sum(expit(pd_[0] + pd_[1] * md) * d['n'])
                                                          / d['n'].sum())}
        out[sp] = r
    return out

def write_doc(path: Path, r: dict) -> None:
    m = r['merge']
    ref = r['reference']
    rows = [{'ranking': f'{DELINQUENT_MODEL} on every row (current)', 'valid AP': ref['valid'], 'test AP': ref['test']},
            {'ranking': 'merge, raw margins', 'valid AP': m['valid']['raw'], 'test AP': m['test']['raw']},
            {'ranking': 'merge, Platt per segment fitted on valid', 'valid AP': m['valid']['platt_valid'],
             'test AP': m['test']['platt_valid']},
            {'ranking': 'merge, Platt per segment fitted on test (oracle)', 'valid AP': m['valid']['platt_oracle'],
             'test AP': m['test']['platt_oracle']},
            {'ranking': 'merge, best offset on the clean margin (oracle)', 'valid AP': m['valid']['offset_best'],
             'test AP': m['test']['offset_best']}]
    t = pd.DataFrame(rows)
    t['test vs current'] = [f"{x - ref['test']:+.4f}" for x in t['test AP']]
    seg = pd.DataFrame([{'rows': k, 'bad rate': f"{m['bad_rate'][k]:.4%}",
                         'model': CLEAN_MODEL if k.endswith('clean') else DELINQUENT_MODEL,
                         'AP within segment': v} for k, v in sorted(m['segment_ap'].items())])
    cal = pd.DataFrame([{'split': sp, 'segment': s, 'actual bad rate': f"{m['bad_rate'][f'{sp}/{s}']:.4%}",
                         'mean p, raw': f"{m[sp]['mean_pred'][f'{s}_raw']:.4%}",
                         'mean p, Platt (valid)': f"{m[sp]['mean_pred'][f'{s}_platt_valid']:.4%}"}
                        for sp in ('valid', 'test') for s in ('clean', 'delinquent')])
    sw = pd.DataFrame({'clean offset': list(m['test']['sweep']), 'valid AP': list(m['valid']['sweep'].values()),
                       'test AP': list(m['test']['sweep'].values())})
    sw = sw[np.isclose(sw["clean offset"] * 2, np.round(sw["clean offset"] * 2))]
    pre = (f"One ranking from two models: **{CLEAN_MODEL}** (job 20, fitted on clean rows, with the regional "
           f"features) scores the clean rows, **{DELINQUENT_MODEL}** (job 15, the current model) the delinquent rows. "
           f"Clean = no 30+ DPD in the last 3 months and no 60+ in the last 12. Every valid and test row is scored.\n\n"
           f"- The two margins come from models fitted on different rows, so they need not be on one scale. "
           f"**Raw** merges them as they are; **Platt (valid)** maps each segment's margin to a probability with a "
           f"logistic fit on that segment's valid rows, the deployable version. The two **oracle** rows are fitted "
           f"or chosen on test and only bound what a better calibration could reach.\n"
           f"- Within a segment, every merge keeps the model's ranking; only how the segments interleave changes.")
    write_report(path, 'Two-Segment Merge', 'src/jobs/22_segment_merge_main.py',
                 [('Average precision of the merged ranking', md_table(t, floatfmt='{:.4f}', numparse=False)),
                  ('AP within each segment', md_table(seg, floatfmt='{:.4f}', numparse=False)),
                  ('Calibration: mean predicted probability vs bad rate', md_table(cal, numparse=False)),
                  ('Offset added to the clean margin', md_table(sw, floatfmt='{:.4f}'))], pre)


def main() -> None:
    p = argparse.ArgumentParser(description='segment_merge_main')
    p.add_argument('--stage', choices=('all', 'score', 'merge'), default='all')
    p.add_argument('--training-set', default=curated_path('training_set'))
    p.add_argument('--regional', default=curated_path('regional_features'), help='written by job 20')
    p.add_argument('--split-json', default=str(settings.model_dir / 'gbm_split.json'))
    p.add_argument('--elimination-json', default=str(settings.model_dir / 'gbm_elimination.json'))
    p.add_argument('--selection-json', default=str(settings.model_dir / 'gbm_corr_selection.json'))
    p.add_argument('--univariate-json', default=str(settings.model_dir / 'gbm_univariate.json'))
    p.add_argument('--gbm-models-dir', default=str(settings.model_dir / 'gbm_tuning_models'))
    p.add_argument('--models-dir', default=str(settings.model_dir / 'regional_models'))
    p.add_argument('--tables', default=str(settings.model_dir / 'segment_merge_tables.parquet'),
                   help='(version, split, rounded margin) -> rows, bads')
    p.add_argument('--result-json', default=str(settings.model_dir / 'segment_merge.json'))
    p.add_argument('--doc', default=str(Path(__file__).resolve().parents[2] / 'docs' / 'segment_merge.md'))
    a = p.parse_args()

    cfg = load_split(Path(a.split_json))['config']
    feats = json.loads(Path(a.elimination_json).read_text())['suggested_features']
    reg = g20.REGIONAL
    kinds = json.loads(Path(a.selection_json).read_text())['kinds'] | {c: 'numerical' for c in reg}
    cats = json.loads(Path(a.univariate_json).read_text())['categories']

    if a.stage in ('all', 'score'):
        spark = spark_session('segment_merge', conf={'spark.sql.execution.arrow.pyspark.enabled': 'true',
                                                     'spark.driver.maxResultSize': '4g'})
        try:
            score_tables(spark, a, cfg, feats, reg, kinds, cats).to_parquet(a.tables)
        finally:
            spark.stop()
    if a.stage in ('all', 'merge'):
        m = merge(pd.read_parquet(a.tables))
        full = json.loads((Path(a.gbm_models_dir) / 'full_ap.json').read_text())[g20.VERSION]
        result = {'meta': {'clean_model': CLEAN_MODEL, 'delinquent_model': DELINQUENT_MODEL, 'split': cfg,
                           'source': 'src/jobs/22_segment_merge_main.py',
                           'generated': datetime.now().isoformat(timespec='seconds')},
                  'reference': {sp: full[sp]['average_precision'] for sp in ('valid', 'test')}, 'merge': m}
        Path(a.result_json).write_text(json.dumps(result, indent=2, default=float))
        write_doc(Path(a.doc), result)
        for sp in ('valid', 'test'):
            r = m[sp]
            log.info('[%s] current %.4f | merge raw %.4f, Platt (valid) %.4f, Platt oracle %.4f, best offset %.4f '
                     'at %+.1f', sp, result['reference'][sp], r['raw'], r['platt_valid'], r['platt_oracle'],
                     r['offset_best'], r['offset_best_at'])
        log.info('Result in %s, report in %s', a.result_json, a.doc)


if __name__ == '__main__':
    main()
