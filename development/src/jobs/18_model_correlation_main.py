import argparse
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
import xgboost as xgb
from pyspark.sql import DataFrame, functions as F
from pyspark.sql.types import ByteType, DoubleType, StructField, StructType
from helpers.config import settings
from helpers import hdfs
from helpers.gbm import TARGET, average_precision, load_split, to_frame, ym
from helpers.report import md_table, write_report
from helpers.rf import bad_proba, cached_model, to_matrix
from helpers.spark_io import curated_path, resolve_path, spark_session

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s | %(message)s')
log = logging.getLogger('model_correlation_main')

BEST = {'lr': 'recency_6', 'xgb': 'recency_12', 'rf': 'undersample'}
NAMES = {'lr': 'Logistic regression', 'xgb': 'XGBoost', 'rf': 'Random forest'}
MODELS = list(BEST)
LOGIT_DECIMALS = 4
PROB_DECIMALS = 6
RF_CLIP = 1e-6
KEYS = ['loan_identifier', 'period']

os.environ.setdefault('PYSPARK_PYTHON', sys.executable)


def lr_logit(m: dict):
    """The logistic regression's logit from its WoE columns, rounded to LOGIT_DECIMALS."""
    expr = F.lit(m['intercept'])
    for c in m['coefficients']:
        expr = expr + F.col(c['feature']).cast('double') * F.lit(c['coef'])
    return F.round(expr, LOGIT_DECIMALS)

def model_features(a) -> Dict[str, List[str]]:
    """The raw columns each model reads (LR: the columns behind its WoE features)."""
    lr = json.loads(Path(a.lr_json).read_text())['versions'][BEST['lr']]
    return {'lr': [c['feature'].removesuffix('_woe') for c in lr['coefficients']],
            'xgb': json.loads(Path(a.gbm_elimination_json).read_text())['suggested_features'],
            'rf': json.loads(Path(a.rf_elimination_json).read_text())['suggested_features']}

def scored_test(spark, a, cfg: dict, bads_only: bool = False, keep: List[str] = ()) -> DataFrame:
    """Test rows with their label, LR logit, XGBoost margin and RF P(bad) (rounded
    as in the AP tables), plus the raw columns in keep."""
    end_ym = int(cfg['end_ym'])
    lr = json.loads(Path(a.lr_json).read_text())['versions'][BEST['lr']]
    kinds = json.loads(Path(a.selection_json).read_text())['kinds']
    xf = json.loads(Path(a.gbm_elimination_json).read_text())['suggested_features']
    cats = json.loads(Path(a.univariate_json).read_text())['categories']
    elim = json.loads(Path(a.rf_elimination_json).read_text())
    rf_feats, enc = elim['suggested_features'], elim['encoding']
    enc = {c: enc[c] for c in rf_feats if kinds[c] == 'categorical'}
    xinfo = json.loads((Path(a.gbm_models_dir) / f"{BEST['xgb']}.json").read_text())
    xraw, xbest = (Path(a.gbm_models_dir) / f"{BEST['xgb']}.ubj").read_bytes(), int(xinfo['best_iteration'])
    rf_path = str((Path(a.rf_models_dir) / f"{BEST['rf']}.joblib").resolve())
    raw_feats = sorted(set(xf) | set(rf_feats) | set(keep))
    keep = list(keep)

    raw = spark.read.parquet(resolve_path(a.training_set)).where(ym() > F.lit(end_ym))
    woe = spark.read.parquet(resolve_path(a.woe_selected)).where(ym() > F.lit(end_ym))
    if bads_only:
        raw, woe = raw.where(F.col(TARGET) == 1), woe.where(F.col(TARGET) == 1)
    df = raw.select(*KEYS, TARGET, *raw_feats).join(woe.select(*KEYS, lr_logit(lr).alias('lr')), KEYS, 'left')
    bc = spark.sparkContext.broadcast((xraw, xbest, xf, {c: cats[c] for c in xf if kinds[c] == 'categorical'}))

    def score(batches):
        raw_b, best, fs, ct = bc.value
        booster = xgb.Booster()
        booster.load_model(bytearray(raw_b))
        booster.set_param({'nthread': 1})
        forest = cached_model(rf_path)
        for pdf in batches:
            d = xgb.DMatrix(to_frame(pdf, fs, kinds, ct), enable_categorical=True, nthread=1)
            m = booster.predict(d, output_margin=True, iteration_range=(0, best + 1)).astype(np.float64)
            p = bad_proba(forest, to_matrix(pdf, rf_feats, kinds, enc)).astype(np.float64)
            out = pd.DataFrame({TARGET: pdf[TARGET].to_numpy(np.int8), 'lr': pdf['lr'].to_numpy(np.float64),
                                'xgb': np.round(m, LOGIT_DECIMALS), 'rf': np.round(p, PROB_DECIMALS)})
            for c in keep:
                out[c] = pdf[c].to_numpy()
            yield out

    schema = StructType([StructField(TARGET, ByteType())] + [StructField(c, DoubleType()) for c in MODELS]
                        + [df.schema[c] for c in keep])
    return df.mapInPandas(score, schema)

def score_test(spark, a, cfg: dict) -> None:
    """Every test row's scores, with its label, to a.scores."""
    scored_test(spark, a, cfg).write.mode('overwrite').parquet(resolve_path(a.scores))

def mid_ranks(df: DataFrame, col: str) -> pd.DataFrame:
    """Distinct value -> its mid-rank (ties share the average of their ranks)."""
    c = df.groupBy(col).count().toPandas().sort_values(col)
    n = c['count'].to_numpy(np.float64)
    c['r_' + col] = np.cumsum(n) - (n - 1) / 2
    return c[[col, 'r_' + col]]

def pair_table(df: DataFrame, cols: Dict[str, str]) -> pd.DataFrame:
    """Pearson correlation of every model pair on the given per-model columns."""
    pairs = [(x, y) for i, x in enumerate(MODELS) for y in MODELS[i + 1:]]
    r = df.agg(*[F.corr(cols[x], cols[y]).alias(f'{x}|{y}') for x, y in pairs]).first().asDict()
    m = pd.DataFrame(np.eye(len(MODELS)), index=MODELS, columns=MODELS)
    for k, v in r.items():
        x, y = k.split('|')
        m.loc[x, y] = m.loc[y, x] = v
    return m

def correlations(spark, a) -> dict:
    """From the stored test scores of the best version of each model (by test AP on every test row): the test AP
    recomputed from them, which must match each model's own job, and the Pearson correlation on probabilities
    and on logits, the Spearman correlation on mid-ranks, and the logit correlation within bads and within goods
    (do the models agree on which bads, or goods, look riskiest). Scores are rounded as in the AP tables (logit
    and margin to 1e-4, RF P(bad) to 1e-6), so ties share a mid-rank; RF P(bad) is clipped to RF_CLIP before its
    logit is taken, as it can be exactly 0 or 1."""
    df = spark.read.parquet(resolve_path(a.scores)).cache()
    counts = df.agg(F.count(F.lit(1)).alias('rows'), F.sum(TARGET).alias('bads'),
                    *[F.sum(F.col(c).isNull().cast('long')).alias(f'null_{c}') for c in MODELS]).first().asDict()
    log.info('Scores: %s', counts)

    ap = {}
    for c in MODELS:
        g = df.groupBy(c).agg(F.count(F.lit(1)).alias('n'), F.sum(TARGET).alias('bad')).toPandas().dropna()
        ap[c] = average_precision(g[c].to_numpy(), g['bad'].to_numpy(float), g['n'].to_numpy(float))
    log.info('Test AP from the stored scores: %s', {k: round(v, 4) for k, v in ap.items()})

    clip = F.least(F.greatest(F.col('rf'), F.lit(RF_CLIP)), F.lit(1 - RF_CLIP))
    sc = (df.withColumn('p_lr', 1 / (1 + F.exp(-F.col('lr')))).withColumn('p_xgb', 1 / (1 + F.exp(-F.col('xgb'))))
          .withColumn('z_rf', F.log(clip / (1 - clip))))
    for c in MODELS:
        sc = sc.join(F.broadcast(spark.createDataFrame(mid_ranks(df, c))), c, 'left')
    sc = sc.cache()
    out = {'counts': counts, 'test_ap': ap, 'by_label': {}}
    out['pearson_prob'] = pair_table(sc, {'lr': 'p_lr', 'xgb': 'p_xgb', 'rf': 'rf'})
    out['pearson_logit'] = pair_table(sc, {'lr': 'lr', 'xgb': 'xgb', 'rf': 'z_rf'})
    out['spearman'] = pair_table(sc, {c: 'r_' + c for c in MODELS})
    for y, name in ((1, 'bads'), (0, 'goods')):
        out['by_label'][name] = pair_table(sc.where(F.col(TARGET) == y), {'lr': 'lr', 'xgb': 'xgb', 'rf': 'z_rf'})
    sc.unpersist()
    df.unpersist()
    return out

def write_doc(path: Path, r: dict, cfg: dict) -> None:
    label = {c: f"{NAMES[c]} ({BEST[c]})" for c in MODELS}

    def named(m: pd.DataFrame) -> pd.DataFrame:
        return m.rename(index=label, columns=label).reset_index().rename(columns={'index': ''})

    n = r['counts']
    pre = (f"Correlation of the predicted P(bad) of the best version of each model (by test AP on every test row) "
           f"over **all {n['rows']:,} test rows** (after {cfg['dev_end']}; {n['bads']:,} bads). Each row is "
           f"scored by the three models in one pass: the logistic regression on its WoE columns "
           f"(`training_set_woe_selected`), XGBoost and the random forest on the raw columns (`training_set`), "
           f"joined on (loan, period).\n\n"
           f"- **Pearson (probability)**: on P(bad) itself; dominated by the few high-risk rows.\n"
           f"- **Pearson (logit)**: on log-odds (LR logit, XGBoost margin, RF P(bad) clipped to "
           f"[{RF_CLIP:g}, 1 - {RF_CLIP:g}]); spreads the low-risk mass out.\n"
           f"- **Spearman**: Pearson of the ranks, ties sharing their mid-rank; scores rounded as in the AP "
           f"tables (logit and margin to 1e-{LOGIT_DECIMALS}, RF P(bad) to 1e-{PROB_DECIMALS}). Rank agreement "
           f"is what matters for AP.\n"
           f"- Test AP recomputed from these scores: "
           + ', '.join(f"{label[c]} {r['test_ap'][c]:.4f}" for c in MODELS)
           + " (matches each model's own job).")
    fmt = '{:.4f}'
    write_report(path, 'Model Score Correlation on Test', 'src/jobs/18_model_correlation_main.py',
                 [('Spearman (rank)', md_table(named(r['spearman']), floatfmt=fmt)),
                  ('Pearson (probability)', md_table(named(r['pearson_prob']), floatfmt=fmt)),
                  ('Pearson (logit)', md_table(named(r['pearson_logit']), floatfmt=fmt)),
                  ('Pearson (logit), bads only', md_table(named(r['by_label']['bads']), floatfmt=fmt)),
                  ('Pearson (logit), goods only', md_table(named(r['by_label']['goods']), floatfmt=fmt))], pre)


def add_model_args(p: argparse.ArgumentParser) -> None:
    """Inputs of the three best models and their test rows (shared with 19_error_analysis_main)."""
    p.add_argument('--training-set', default=curated_path('training_set'))
    p.add_argument('--woe-selected', default=curated_path('training_set_woe_selected'))
    p.add_argument('--scores', default=curated_path('model_scores_test'),
                   help='per-row test scores of the three models (HDFS), written by 18_model_correlation_main')
    p.add_argument('--split-json', default=str(settings.model_dir / 'gbm_split.json'))
    p.add_argument('--lr-json', default=str(settings.model_dir / 'lr_models.json'))
    p.add_argument('--selection-json', default=str(settings.model_dir / 'gbm_corr_selection.json'))
    p.add_argument('--gbm-elimination-json', default=str(settings.model_dir / 'gbm_elimination.json'))
    p.add_argument('--univariate-json', default=str(settings.model_dir / 'gbm_univariate.json'))
    p.add_argument('--rf-elimination-json', default=str(settings.model_dir / 'rf_elimination.json'))
    p.add_argument('--gbm-models-dir', default=str(settings.model_dir / 'gbm_tuning_models'))
    p.add_argument('--rf-models-dir', default=str(settings.model_dir / 'rf_tuning_models'))


def main() -> None:
    p = argparse.ArgumentParser(description='model_correlation_main')
    p.add_argument('--stage', choices=('all', 'score', 'correlate'), default='all')
    p.add_argument('--rescore', action='store_true', help='score the test rows again even if --scores exists')
    add_model_args(p)
    p.add_argument('--result-json', default=str(settings.model_dir / 'model_correlation.json'))
    p.add_argument('--doc', default=str(Path(__file__).resolve().parents[2] / 'docs' / 'model_correlation.md'))
    a = p.parse_args()

    cfg = load_split(Path(a.split_json))['config']
    spark = spark_session('model_correlation', conf={'spark.sql.execution.arrow.pyspark.enabled': 'true',
                                                     'spark.sql.execution.arrow.maxRecordsPerBatch': '50000',
                                                     'spark.driver.maxResultSize': '4g'})
    try:
        if a.stage in ('all', 'score'):
            if hdfs.exists(f'{a.scores}/_SUCCESS') and not a.rescore:
                log.info('Scores already at %s; --rescore to score again', a.scores)
            else:
                t = time.time()
                score_test(spark, a, cfg)
                log.info('Scored every test row in %.0f min -> %s', (time.time() - t) / 60, a.scores)
        if a.stage in ('all', 'correlate'):
            t = time.time()
            r = correlations(spark, a)
            log.info('Correlations in %.0f min', (time.time() - t) / 60)
    finally:
        spark.stop()

    if a.stage in ('all', 'correlate'):
        for k in ('spearman', 'pearson_prob', 'pearson_logit'):
            log.info('%s:\n%s', k, r[k].round(4))
        js = {k: (v.to_dict() if isinstance(v, pd.DataFrame) else v) for k, v in r.items()}
        js['by_label'] = {k: v.to_dict() for k, v in r['by_label'].items()}
        js['meta'] = {'best': BEST, 'split': cfg, 'scores': a.scores, 'source': 'src/jobs/18_model_correlation_main.py',
                      'generated': datetime.now().isoformat(timespec='seconds')}
        Path(a.result_json).write_text(json.dumps(js, indent=2, default=float))
        write_doc(Path(a.doc), r, cfg)
        log.info('Result in %s, report in %s', a.result_json, a.doc)


if __name__ == '__main__':
    main()
