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
from pyspark.sql import Column, DataFrame, Window, functions as F
from sklearn.metrics import average_precision_score
from helpers.config import settings
from helpers import hdfs
from helpers.gbm import SPLIT, TARGET, load_split, sample_meta_path, to_frame, uniform, with_split, ym
from helpers.report import md_table, write_report
from helpers.spark_io import curated_path, resolve_path, spark_session

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s | %(message)s')
log = logging.getLogger('regional_features_main')

g15 = importlib.import_module('jobs.15_gbm_tuning_main')

LEVELS = {'zip3': 'postal_code', 'msa': 'metropolitan_statistical_area_or_metropolitan_division',
          'state': 'property_state'}
REGIONAL = [f'{lv}_{f}' for lv in LEVELS for f in ('loans', 'dpd30_rate', 'dpd90_rate', 'new_dpd30_3m',
                                                     'dpd30_chg_6m')]
FIRST_MONTH = '2015-01-01'

VERSION = 'recency_12'
HALF_LIFE = 12
WEIGHT = 'sample_weight'

os.environ.setdefault('PYSPARK_PYTHON', sys.executable)


def clean() -> Column:
    """The clean segment of 19_error_analysis_main: no 30+ DPD in the last 3 months and no 60+ DPD in the last 12
    (missing = never delinquent)."""
    return ((F.coalesce(F.col('dpd_cnt30_l3m'), F.lit(0)) == 0)
            & ~F.coalesce(F.col('months_since_last_dpd60') <= 11, F.lit(False)))

def build_regional(spark, a) -> None:
    """(level, region, month) -> features, one parquet per level. Built from the raw monthly performance tape
    (every loan active in t), not from training_set, which drops rows on what happens after t (prepaid or
    unobserved inside the label window). Month t itself is fine: a kept row is not bad in t, so its label is
    about t+1 .. t+25, and month t's statuses are on the same tape as the row's own features. FIRST_MONTH is
    12 months before the first training month, for the lags."""
    perf = (spark.read.parquet(resolve_path(a.performance))
            .where((F.col('period') >= F.lit(FIRST_MONTH)) & F.col('zero_balance_code').isNull()
                   & F.col('dpd').isNotNull())
            .select('loan_identifier', 'period', 'dpd'))
    orig = spark.read.parquet(resolve_path(a.origination)).select('loan_identifier', *LEVELS.values())
    df = perf.join(orig, 'loan_identifier').cache()
    for lv, col in LEVELS.items():
        g = (df.where(F.col(col).isNotNull()).groupBy(F.col(col).alias('region'), 'period')
             .agg(F.count(F.lit(1)).alias('loans'),
                  F.avg((F.col('dpd') >= 1).cast('double')).alias('dpd30_rate'),
                  F.avg((F.col('dpd') >= 3).cast('double')).alias('dpd90_rate'),
                  F.avg((F.col('dpd') == 1).cast('double')).alias('new_dpd30')))
        mi = F.year('period') * 12 + F.month('period')
        w = Window.partitionBy('region').orderBy(mi)
        g = (g.withColumn('new_dpd30_3m', F.avg('new_dpd30').over(w.rangeBetween(-2, 0)))
             .withColumn('dpd30_chg_6m', F.col('dpd30_rate') - F.first('dpd30_rate').over(w.rangeBetween(-6, -6)))
             .select(F.col('region').alias(col), 'period',
                     *[F.col(f).alias(f'{lv}_{f}') for f in ('loans', 'dpd30_rate', 'dpd90_rate', 'new_dpd30_3m',
                                                             'dpd30_chg_6m')]))
        g.coalesce(8).write.mode('overwrite').parquet(resolve_path(f'{a.regional}/level={lv}'))
        log.info('Regional features by %s written', lv)
    df.unpersist()

def with_regional(spark, df: DataFrame, a) -> DataFrame:
    """df with the regional features of its zip3, MSA and state in its month joined on."""
    for lv, col in LEVELS.items():
        r = spark.read.parquet(resolve_path(f'{a.regional}/level={lv}'))
        df = df.join(F.broadcast(r), [col, 'period'], 'left')
    return df

def build_sample(spark, a, cfg: dict, feats: List[str], kinds: Dict[str, str], gf: float) -> None:
    """Clean train / valid rows, every bad and job 15's draw of goods (weighted 1 / gf)."""
    loan, period = F.col('loan_identifier'), F.col('period')
    df = with_split(spark.read.parquet(resolve_path(a.training_set)).where(ym() <= F.lit(int(cfg['end_ym']))), cfg)
    keep = (F.col(TARGET) == 1) | (uniform(loan, period, salt='gbm_goods') < F.lit(gf))
    df = with_regional(spark, df.where(clean() & keep), a)
    (df.select(SPLIT, F.col(TARGET).cast('tinyint').alias(TARGET),
               F.when(F.col(TARGET) == 1, F.lit(1.0)).otherwise(F.lit(1 / gf)).alias(WEIGHT),
               ym().cast('int').alias('ym'),
               *[F.col(c).cast('float').alias(c) if kinds[c] == 'numerical' else F.col(c) for c in feats])
     .coalesce(64).write.mode('overwrite').parquet(resolve_path(a.sample)))
    s = spark.read.parquet(resolve_path(a.sample))
    matched = s.agg(*[F.avg(F.col(f'{lv}_dpd30_rate').isNotNull().cast('double')).alias(lv) for lv in LEVELS]).first()
    log.info('Sample rows with regional features, by level: %s', {k: round(v, 4) for k, v in matched.asDict().items()})

def fit(a, cfg: dict, feats: List[str], reg: List[str], kinds: Dict[str, str], cats: Dict[str, list],
        params: dict) -> Dict[str, dict]:
    """XGBoost on the clean sample, recency weighted, early stopped on the clean valid sample;
    with and without the regional features."""
    strings = [SPLIT] + [c for c in feats if kinds[c] == 'categorical']
    with hdfs.local_copy(a.sample) as local:
        pdf = pq.read_table(local, read_dictionary=strings).to_pandas()
    train, valid = pdf[pdf[SPLIT] == 'train'], pdf[pdf[SPLIT] == 'valid']
    del pdf
    end_ym = int(cfg['end_ym'])
    w = train[WEIGHT].to_numpy(np.float64) * 0.5 ** (g15.months_before(train['ym'].to_numpy(), end_ym) / HALF_LIFE)
    y, vy, vw = train[TARGET].to_numpy(np.float32), valid[TARGET].to_numpy(np.float32), valid[WEIGHT].to_numpy(np.float32)
    log.info('Clean sample: %s train rows (%s bads), %s valid rows', f'{len(y):,}', f'{int(y.sum()):,}',
             f'{len(vy):,}')
    out = {}
    for name, fs in (('clean_base', feats), ('clean_regional', feats + reg)):
        t = time.time()
        dtrain = xgb.QuantileDMatrix(to_frame(train, fs, kinds, cats), y, weight=w.astype(np.float32),
                                     enable_categorical=True, max_bin=256)
        dvalid = xgb.QuantileDMatrix(to_frame(valid, fs, kinds, cats), vy, weight=vw, enable_categorical=True,
                                     ref=dtrain)
        b = xgb.train(params, dtrain, a.num_rounds, evals=[(dvalid, 'valid')],
                      early_stopping_rounds=a.early_stopping, verbose_eval=False)
        m = b.predict(dvalid, output_margin=True, iteration_range=(0, b.best_iteration + 1))
        ap = float(average_precision_score(vy, m, sample_weight=vw))
        path = Path(a.models_dir) / f'{name}.ubj'
        b.save_model(str(path))
        gain = b.get_score(importance_type='total_gain')
        total = sum(gain.values())
        out[name] = {'features': fs, 'best_iteration': int(b.best_iteration), 'valid_ap_sample': ap,
                     'seconds': round(time.time() - t), 'path': str(path),
                     'gain_share': dict(sorted(((k, v / total) for k, v in gain.items()), key=lambda kv: -kv[1]))}
        log.info('[%s] %s features, %s rounds, valid AP (sample) %.4f, %.0f min', name, len(fs),
                 b.best_iteration + 1, ap, (time.time() - t) / 60)
        del dtrain, dvalid
    return out

def evaluate(spark, a, cfg: dict, feats: List[str], reg: List[str], kinds: Dict[str, str],
             cats: Dict[str, list], fitted: Dict[str, dict]) -> Dict[str, dict]:
    """AP on every clean valid and test row: the two clean models and the all-rows model."""
    df = with_split(spark.read.parquet(resolve_path(a.training_set)), cfg)
    df = with_regional(spark, df.where(clean() & F.col(SPLIT).isin('valid', 'test')), a)
    info = json.loads((Path(a.gbm_models_dir) / f'{VERSION}.json').read_text())
    base = {f'all_rows_{VERSION}': ((Path(a.gbm_models_dir) / f'{VERSION}.ubj').read_bytes(),
                                    int(info['best_iteration'])),
            'clean_base': (Path(fitted['clean_base']['path']).read_bytes(), fitted['clean_base']['best_iteration'])}
    regional = {'clean_regional': (Path(fitted['clean_regional']['path']).read_bytes(),
                                   fitted['clean_regional']['best_iteration'])}
    t = time.time()
    full = g15.ap_table(g15.score_all(df, base, feats, kinds, cats))
    full |= g15.ap_table(g15.score_all(df, regional, feats + reg, kinds, cats))
    log.info('Scored every clean valid / test row in %.0f min', (time.time() - t) / 60)
    return full

def write_doc(path: Path, r: dict) -> None:
    full, fitted = r['full'], r['fitted']
    ref = f'all_rows_{VERSION}'
    rows = []
    for name in (ref, 'clean_base', 'clean_regional'):
        f = full[name]
        rows.append({'model': name, 'trained on': 'every row (job 15)' if name == ref else 'clean rows',
                     'features': len(fitted.get(name, {}).get('features', r['features'])),
                     'clean valid AP': f['valid']['average_precision'], 'clean test AP': f['test']['average_precision'],
                     'test vs all-rows model': f"{f['test']['average_precision'] - full[ref]['test']['average_precision']:+.4f}"})
    t = pd.DataFrame(rows)
    gain = pd.DataFrame([{'feature': k, 'gain share': v} for k, v in fitted['clean_regional']['gain_share'].items()])
    gain['regional'] = gain['feature'].isin(r['regional'])
    reg_share = float(gain.loc[gain['regional'], 'gain share'].sum())
    te, va = full[ref]['test'], full[ref]['valid']
    pre = (f"Do regional delinquency features help rank the **clean** loans (no 30+ DPD in the last 3 months, no 60+ "
           f"in the last 12), where `19_error_analysis_main` found the models miss 97-99% of bads? Clean rows: "
           f"valid {va['rows']:,} (bad rate {va['bad_rate']:.4%}), test {te['rows']:,} (bad rate "
           f"{te['bad_rate']:.4%}); no-skill AP is the bad rate.\n\n"
           f"- **Regional features** of month t, for the 3-digit zip, the MSA and the state, over every loan active "
           f"in t on the raw performance tape: active loans, share 30+ DPD, share 90+ DPD, share newly 30 DPD "
           f"(averaged over t-2..t) and the 6-month change of the 30+ share. Month t is not leakage: a kept row is "
           f"not bad in t, so its label covers t+1..t+25. The raw tape, not training_set, because training_set "
           f"drops rows on what happens after t.\n"
           f"- **Clean models**: XGBoost with `{VERSION}`'s hyperparameters and recency weights (half-life "
           f"{HALF_LIFE} months), on every clean train bad and job 15's 2% draw of clean goods (weighted 50), early "
           f"stopped on the clean valid sample; the same with the {len(r['regional'])} regional features added. "
           f"Hyperparameters are not re-tuned for either, so the comparison is like for like.\n"
           f"- **all_rows_{VERSION}** is the current XGBoost model (fitted on every row), scored on the same clean "
           f"rows.\n"
           f"- Regional features take **{reg_share:.1%}** of the clean_regional model's total gain.")
    write_report(path, 'Regional Features on the Clean Segment', 'src/jobs/20_regional_features_main.py',
                 [('Average precision on every clean row', md_table(t, floatfmt='{:.4f}', numparse=False)),
                  ('clean_regional: gain share, top 25', md_table(gain.head(25), floatfmt='{:.4f}'))], pre)


def main() -> None:
    p = argparse.ArgumentParser(description='regional_features_main')
    p.add_argument('--stage', choices=('all', 'build', 'sample', 'fit', 'evaluate'), default='all')
    p.add_argument('--performance', default=curated_path('performance'))
    p.add_argument('--origination', default=curated_path('origination'))
    p.add_argument('--training-set', default=curated_path('training_set'))
    p.add_argument('--regional', default=curated_path('regional_features'))
    p.add_argument('--sample', default=curated_path('clean_regional_sample'))
    p.add_argument('--split-json', default=str(settings.model_dir / 'gbm_split.json'))
    p.add_argument('--elimination-json', default=str(settings.model_dir / 'gbm_elimination.json'))
    p.add_argument('--selection-json', default=str(settings.model_dir / 'gbm_corr_selection.json'))
    p.add_argument('--univariate-json', default=str(settings.model_dir / 'gbm_univariate.json'))
    p.add_argument('--elimination-sample', default=curated_path('gbm_elimination_sample'),
                   help='its meta file gives the goods fraction')
    p.add_argument('--gbm-models-dir', default=str(settings.model_dir / 'gbm_tuning_models'))
    p.add_argument('--models-dir', default=str(settings.model_dir / 'regional_models'))
    p.add_argument('--num-rounds', type=int, default=g15.NUM_ROUNDS)
    p.add_argument('--early-stopping', type=int, default=g15.EARLY_STOPPING)
    p.add_argument('--result-json', default=str(settings.model_dir / 'regional_features.json'))
    p.add_argument('--doc', default=str(Path(__file__).resolve().parents[2] / 'docs' / 'regional_features.md'))
    a = p.parse_args()
    run = {s: a.stage in ('all', s) for s in ('build', 'sample', 'fit', 'evaluate')}

    cfg = load_split(Path(a.split_json))['config']
    feats = json.loads(Path(a.elimination_json).read_text())['suggested_features']
    kinds = json.loads(Path(a.selection_json).read_text())['kinds'] | {c: 'numerical' for c in REGIONAL}
    cats = json.loads(Path(a.univariate_json).read_text())['categories']
    gf = float(json.loads(sample_meta_path(a.elimination_sample).read_text())['goods_fraction'])
    params = json.loads((Path(a.gbm_models_dir) / f'{VERSION}.json').read_text())['params']
    Path(a.models_dir).mkdir(parents=True, exist_ok=True)
    fitted_path = Path(a.models_dir) / 'fitted.json'

    spark = None
    if run['build'] or run['sample'] or run['evaluate']:
        spark = spark_session('regional_features', conf={'spark.sql.execution.arrow.pyspark.enabled': 'true',
                                                         'spark.driver.maxResultSize': '4g'})
    try:
        if run['build']:
            t = time.time()
            build_regional(spark, a)
            log.info('Regional features in %.0f min -> %s', (time.time() - t) / 60, a.regional)
        if run['sample']:
            t = time.time()
            build_sample(spark, a, cfg, feats + REGIONAL, kinds, gf)
            log.info('Clean sample in %.0f min -> %s', (time.time() - t) / 60, a.sample)
        if run['fit']:
            fitted_path.write_text(json.dumps(fit(a, cfg, feats, REGIONAL, kinds, cats, params), indent=2))
        if run['evaluate']:
            fitted = json.loads(fitted_path.read_text())
            full = evaluate(spark, a, cfg, feats, REGIONAL, kinds, cats, fitted)
            result = {'meta': {'version': VERSION, 'half_life': HALF_LIFE, 'split': cfg, 'regional': a.regional,
                               'source': 'src/jobs/20_regional_features_main.py',
                               'generated': datetime.now().isoformat(timespec='seconds')},
                      'features': feats, 'regional': REGIONAL, 'fitted': fitted, 'full': full}
            Path(a.result_json).write_text(json.dumps(result, indent=2, default=float))
            write_doc(Path(a.doc), result)
            for name, f in full.items():
                log.info('[%s] clean valid AP %.4f, clean test AP %.4f', name, f['valid']['average_precision'],
                         f['test']['average_precision'])
            log.info('Result in %s, report in %s', a.result_json, a.doc)
    finally:
        if spark is not None:
            spark.stop()


if __name__ == '__main__':
    main()
