import argparse
import importlib
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from pyspark.sql import functions as F
from sklearn.metrics import average_precision_score
from helpers.config import settings
from helpers import hdfs
from helpers.gbm import SPLIT, TARGET, load_split, sample_meta_path, to_frame, uniform, with_split, ym
from helpers.report import md_table, write_report
from helpers.spark_io import curated_path, resolve_path, spark_session

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s | %(message)s')
log = logging.getLogger('regional_all_rows_main')

g15 = importlib.import_module('jobs.15_gbm_tuning_main')
g20 = importlib.import_module('jobs.20_regional_features_main')

VERSION = g20.VERSION
HALF_LIFE = g20.HALF_LIFE
WEIGHT = g20.WEIGHT
BASE = f'all_rows_{VERSION}'
REGIONAL_MODEL = 'all_rows_regional'
SEGMENTS = ('clean', 'delinquent')

os.environ.setdefault('PYSPARK_PYTHON', sys.executable)


def build_sample(spark, a, cfg: dict, feats: List[str], kinds: Dict[str, str], gf: float) -> None:
    """Train / valid rows of job 15's tuning sample (same draw), with the regional features."""
    loan, period = F.col('loan_identifier'), F.col('period')
    df = with_split(spark.read.parquet(resolve_path(a.training_set)).where(ym() <= F.lit(int(cfg['end_ym']))), cfg)
    keep = (F.col(TARGET) == 1) | (uniform(loan, period, salt='gbm_goods') < F.lit(gf))
    df = g20.with_regional(spark, df.where(F.col(SPLIT).isin('train', 'valid') & keep), a)
    (df.select(SPLIT, F.col(TARGET).cast('tinyint').alias(TARGET),
               F.when(F.col(TARGET) == 1, F.lit(1.0)).otherwise(F.lit(1 / gf)).alias(WEIGHT),
               ym().cast('int').alias('ym'),
               *[F.col(c).cast('float').alias(c) if kinds[c] == 'numerical' else F.col(c) for c in feats])
     .coalesce(64).write.mode('overwrite').parquet(resolve_path(a.sample)))
    s = spark.read.parquet(resolve_path(a.sample))
    c = s.groupBy(SPLIT).agg(F.count(F.lit(1)).alias('rows'), F.sum(TARGET).alias('bads')).collect()
    log.info('Sample: %s', {r[SPLIT]: (f"{r['rows']:,}", f"{r['bads']:,}") for r in c})
    matched = s.agg(*[F.avg(F.col(f'{lv}_dpd30_rate').isNotNull().cast('double')).alias(lv)
                      for lv in g20.LEVELS]).first()
    log.info('Sample rows with regional features, by level: %s', {k: round(v, 4) for k, v in matched.asDict().items()})

def fit(a, cfg: dict, feats: List[str], kinds: Dict[str, str], cats: Dict[str, list], params: dict) -> dict:
    """The regional model: job 15's recency_12 refitted with the regional features added, hyperparameters
    unchanged, on the same rows (every train bad and the 2% draw of goods) with the same weights and early
    stopping on the same valid sample."""
    strings = [SPLIT] + [c for c in feats if kinds[c] == 'categorical']
    with hdfs.local_copy(a.sample) as local:
        pdf = pq.read_table(local, read_dictionary=strings).to_pandas()
    train, valid = pdf[pdf[SPLIT] == 'train'], pdf[pdf[SPLIT] == 'valid']
    del pdf
    w = train[WEIGHT].to_numpy(np.float64) * 0.5 ** (g15.months_before(train['ym'].to_numpy(), int(cfg['end_ym']))
                                                      / HALF_LIFE)
    y, vy, vw = train[TARGET].to_numpy(np.float32), valid[TARGET].to_numpy(np.float32), valid[WEIGHT].to_numpy(np.float32)
    log.info('Sample: %s train rows (%s bads), %s valid rows', f'{len(y):,}', f'{int(y.sum()):,}', f'{len(vy):,}')
    t = time.time()
    dtrain = xgb.QuantileDMatrix(to_frame(train, feats, kinds, cats), y, weight=w.astype(np.float32),
                                 enable_categorical=True, max_bin=256)
    dvalid = xgb.QuantileDMatrix(to_frame(valid, feats, kinds, cats), vy, weight=vw, enable_categorical=True,
                                 ref=dtrain)
    b = xgb.train(params, dtrain, a.num_rounds, evals=[(dvalid, 'valid')], early_stopping_rounds=a.early_stopping,
                  verbose_eval=False)
    m = b.predict(dvalid, output_margin=True, iteration_range=(0, b.best_iteration + 1))
    ap = float(average_precision_score(vy, m, sample_weight=vw))
    path = Path(a.models_dir) / f'{REGIONAL_MODEL}.ubj'
    b.save_model(str(path))
    gain = b.get_score(importance_type='total_gain')
    total = sum(gain.values())
    log.info('[%s] %s features, %s rounds, valid AP (sample) %.4f, %.0f min', REGIONAL_MODEL, len(feats),
             b.best_iteration + 1, ap, (time.time() - t) / 60)
    return {'features': feats, 'best_iteration': int(b.best_iteration), 'valid_ap_sample': ap,
            'seconds': round(time.time() - t), 'path': str(path),
            'gain_share': dict(sorted(((k, v / total) for k, v in gain.items()), key=lambda kv: -kv[1]))}

def evaluate(spark, a, cfg: dict, feats: List[str], reg: List[str], kinds: Dict[str, str],
             cats: Dict[str, list], fitted: dict) -> Dict[str, dict]:
    """AP on every valid and test row, overall and by segment (clean / delinquent). The segment rides in the
    split column, so job 15's one-scan scoring counts by (split, segment)."""
    df = with_split(spark.read.parquet(resolve_path(a.training_set)), cfg).where(F.col(SPLIT).isin('valid', 'test'))
    seg = F.when(g20.clean(), F.lit('clean')).otherwise(F.lit('delinquent'))
    df = g20.with_regional(spark, df, a).withColumn(SPLIT, F.concat_ws('/', F.col(SPLIT), seg))
    info = json.loads((Path(a.gbm_models_dir) / f'{VERSION}.json').read_text())
    t = time.time()
    s = pd.concat([
        g15.score_all(df, {BASE: ((Path(a.gbm_models_dir) / f'{VERSION}.ubj').read_bytes(),
                                  int(info['best_iteration']))}, feats, kinds, cats),
        g15.score_all(df, {REGIONAL_MODEL: (Path(fitted['path']).read_bytes(), fitted['best_iteration'])},
                      feats + reg, kinds, cats)])
    log.info('Scored every valid / test row in %.0f min', (time.time() - t) / 60)
    whole = (s.assign(**{SPLIT: s[SPLIT].str.split('/').str[0]})
             .groupby(['version', SPLIT, 'm'], as_index=False)[['n', 'bad']].sum())
    out = g15.ap_table(whole)
    for v, d in g15.ap_table(s).items():
        out[v] |= d
    return out

def write_doc(path: Path, r: dict) -> None:
    full, fitted = r['full'], r['fitted']
    rows = []
    for key, what in [('valid', 'valid, all rows'), ('test', 'test, all rows'), ('valid/clean', 'valid, clean'),
                      ('test/clean', 'test, clean'), ('valid/delinquent', 'valid, delinquent'),
                      ('test/delinquent', 'test, delinquent')]:
        b, g = full[BASE][key], full[REGIONAL_MODEL][key]
        d = g['average_precision'] - b['average_precision']
        rows.append({'rows': what, 'count': f"{b['rows']:,}", 'bad rate': f"{b['bad_rate']:.4%}",
                     f'{BASE} AP': b['average_precision'], f'{REGIONAL_MODEL} AP': g['average_precision'],
                     'change': f'{d:+.4f}', 'relative': f"{d / b['average_precision']:+.1%}"})
    t = pd.DataFrame(rows)
    cl = r.get('clean_models', {})
    if cl:
        t2 = pd.DataFrame([{'model': n, 'trained on': 'clean rows (job 20)' if n.startswith('clean') else 'every row',
                            'clean valid AP': cl[n]['valid'], 'clean test AP': cl[n]['test']}
                           for n in cl])
    gain = pd.DataFrame([{'feature': k, 'gain share': v} for k, v in fitted['gain_share'].items()])
    gain['regional'] = gain['feature'].isin(r['regional'])
    reg_share = float(gain.loc[gain['regional'], 'gain share'].sum())
    pre = (f"Job 20's {len(r['regional'])} regional delinquency features (3-digit zip, MSA and state of month t, "
           f"from the raw performance tape; see `docs/regional_features.md`) added to the all-rows XGBoost model.\n\n"
           f"- **{BASE}** is job 15's model as it is. **{REGIONAL_MODEL}** is fitted on the same rows (every train bad "
           f"and job 15's 2% draw of goods, weighted 50), with the same hyperparameters, recency weights (half-life "
           f"{HALF_LIFE} months) and early stopping on the same valid sample; only the features differ "
           f"({len(r['features'])} vs {len(fitted['features'])}). {REGIONAL_MODEL} stopped at "
           f"{fitted['best_iteration'] + 1} rounds (valid AP on the sample {fitted['valid_ap_sample']:.4f}).\n"
           f"- The hyperparameters were tuned for {BASE} (best of 30 trials on the valid sample), so valid AP favours "
           f"it; test is untouched by either.\n"
           f"- **Segments** as in `19_error_analysis_main`: clean = no 30+ DPD in the last 3 months and no 60+ in the "
           f"last 12; delinquent = the rest. No-skill AP is the bad rate.\n"
           f"- Regional features take **{reg_share:.1%}** of {REGIONAL_MODEL}'s total gain.")
    sections = [('Average precision on every valid and test row', md_table(t, floatfmt='{:.4f}', numparse=False))]
    if cl:
        sections.append(('Clean rows: against the clean-only models of job 20', md_table(t2, floatfmt='{:.4f}')))
    sections.append((f'{REGIONAL_MODEL}: gain share, top 25', md_table(gain.head(25), floatfmt='{:.4f}')))
    write_report(path, 'Regional Features in the All-Rows Model', 'src/jobs/21_regional_all_rows_main.py',
                 sections, pre)


def main() -> None:
    p = argparse.ArgumentParser(description='regional_all_rows_main')
    p.add_argument('--stage', choices=('all', 'sample', 'fit', 'evaluate'), default='all')
    p.add_argument('--training-set', default=curated_path('training_set'))
    p.add_argument('--regional', default=curated_path('regional_features'), help='written by job 20')
    p.add_argument('--sample', default=curated_path('regional_all_rows_sample'))
    p.add_argument('--split-json', default=str(settings.model_dir / 'gbm_split.json'))
    p.add_argument('--elimination-json', default=str(settings.model_dir / 'gbm_elimination.json'))
    p.add_argument('--selection-json', default=str(settings.model_dir / 'gbm_corr_selection.json'))
    p.add_argument('--univariate-json', default=str(settings.model_dir / 'gbm_univariate.json'))
    p.add_argument('--elimination-sample', default=curated_path('gbm_elimination_sample'),
                   help='its meta file gives the goods fraction')
    p.add_argument('--gbm-models-dir', default=str(settings.model_dir / 'gbm_tuning_models'))
    p.add_argument('--models-dir', default=str(settings.model_dir / 'regional_models'))
    p.add_argument('--clean-json', default=str(settings.model_dir / 'regional_features.json'),
                   help="job 20's result, for the clean-only models' AP")
    p.add_argument('--num-rounds', type=int, default=g15.NUM_ROUNDS)
    p.add_argument('--early-stopping', type=int, default=g15.EARLY_STOPPING)
    p.add_argument('--result-json', default=str(settings.model_dir / 'regional_all_rows.json'))
    p.add_argument('--doc', default=str(Path(__file__).resolve().parents[2] / 'docs' / 'regional_all_rows.md'))
    a = p.parse_args()
    run = {s: a.stage in ('all', s) for s in ('sample', 'fit', 'evaluate')}

    cfg = load_split(Path(a.split_json))['config']
    feats = json.loads(Path(a.elimination_json).read_text())['suggested_features']
    reg = g20.REGIONAL
    kinds = json.loads(Path(a.selection_json).read_text())['kinds'] | {c: 'numerical' for c in reg}
    cats = json.loads(Path(a.univariate_json).read_text())['categories']
    gf = float(json.loads(sample_meta_path(a.elimination_sample).read_text())['goods_fraction'])
    params = json.loads((Path(a.gbm_models_dir) / f'{VERSION}.json').read_text())['params']
    Path(a.models_dir).mkdir(parents=True, exist_ok=True)
    fitted_path = Path(a.models_dir) / f'{REGIONAL_MODEL}.json'

    spark = None
    if run['sample'] or run['evaluate']:
        spark = spark_session('regional_all_rows', conf={'spark.sql.execution.arrow.pyspark.enabled': 'true',
                                                         'spark.driver.maxResultSize': '4g'})
    try:
        if run['sample']:
            t = time.time()
            build_sample(spark, a, cfg, feats + reg, kinds, gf)
            log.info('Sample in %.0f min -> %s', (time.time() - t) / 60, a.sample)
        if run['fit']:
            fitted_path.write_text(json.dumps(fit(a, cfg, feats + reg, kinds, cats, params), indent=2))
        if run['evaluate']:
            fitted = json.loads(fitted_path.read_text())
            full = evaluate(spark, a, cfg, feats, reg, kinds, cats, fitted)
            clean_models = {}
            if Path(a.clean_json).exists():
                c = json.loads(Path(a.clean_json).read_text())['full']
                clean_models = {n: {'valid': c[n]['valid']['average_precision'],
                                    'test': c[n]['test']['average_precision']} for n in c if n.startswith('clean')}
            for n in (BASE, REGIONAL_MODEL):
                clean_models[n] = {'valid': full[n]['valid/clean']['average_precision'],
                                   'test': full[n]['test/clean']['average_precision']}
            result = {'meta': {'version': VERSION, 'half_life': HALF_LIFE, 'split': cfg, 'regional': a.regional,
                               'source': 'src/jobs/21_regional_all_rows_main.py',
                               'generated': datetime.now().isoformat(timespec='seconds')},
                      'features': feats, 'regional': reg, 'fitted': fitted, 'full': full,
                      'clean_models': clean_models}
            Path(a.result_json).write_text(json.dumps(result, indent=2, default=float))
            write_doc(Path(a.doc), result)
            for n, f in full.items():
                log.info('[%s] %s', n, ', '.join(f"{k} {v['average_precision']:.4f}" for k, v in sorted(f.items())))
            log.info('Result in %s, report in %s', a.result_json, a.doc)
    finally:
        if spark is not None:
            spark.stop()


if __name__ == '__main__':
    main()
