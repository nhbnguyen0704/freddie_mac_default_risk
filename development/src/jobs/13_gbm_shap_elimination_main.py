import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from pyspark.sql import DataFrame, functions as F
from sklearn.metrics import average_precision_score
from helpers.config import settings
from helpers.gbm import SPLIT, TARGET, average_precision, dev_rows, load_split, sample_meta_path, to_frame, uniform
from helpers.hdfs import local_copy
from helpers.report import md_table, write_report
from helpers.spark_io import curated_path, spark_session

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s | %(message)s')
log = logging.getLogger('gbm_shap_elimination_main')

WEIGHT = 'sample_weight'
SHAP_ROW = 'shap_row'

GOODS_FRACTION = 0.02

SHAP_FRACTION = 0.1

PARAMS: Dict[str, Any] = {
    'objective': 'binary:logistic', 'eval_metric': 'aucpr', 'tree_method': 'hist', 'learning_rate': 0.1,
    'max_depth': 6, 'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 0.8, 'max_bin': 256, 'seed': 42,
}
NUM_ROUNDS = 2000
EARLY_STOPPING = 50

MARGIN_DECIMALS = 4

SHAP_TIE = 1e-9

ELBOW_TOLERANCE = 0.005

os.environ.setdefault('PYSPARK_PYTHON', sys.executable)

def build_sample(df: DataFrame, feats: List[str], kinds: Dict[str, str], goods_fraction: float,
                 shap_fraction: float, out: str) -> Dict[str, Any]:
    """Write the elimination sample: every bad of train and valid and a seeded goods_fraction of the goods, goods
    weighted 1/goods_fraction, with a seeded shap_fraction of the valid rows flagged for SHAP. Returns its
    record."""
    keep = (F.col(TARGET) == 1) | (uniform(F.col('loan_identifier'), F.col('period'), salt='gbm_goods')
                                  < F.lit(goods_fraction))
    shap = (F.col(SPLIT) == 'valid') & (uniform(F.col('loan_identifier'), F.col('period'), salt='gbm_shap')
                                        < F.lit(shap_fraction))
    sample = (df.where(F.col(SPLIT).isin('train', 'valid') & keep)
              .select(SPLIT, F.col(TARGET).cast('tinyint').alias(TARGET),
                      F.when(F.col(TARGET) == 1, F.lit(1)).otherwise(F.lit(1 / goods_fraction)).alias(WEIGHT),
                      shap.alias(SHAP_ROW),
                      *[F.col(c).cast('float').alias(c) if kinds[c] == 'numerical' else F.col(c) for c in feats]))
    sample.coalesce(64).write.mode('overwrite').parquet(out)
    c = (df.sparkSession.read.parquet(out).groupBy(SPLIT)
         .agg(F.count(F.lit(1)).alias('rows'), F.sum(TARGET).alias('bads'), F.sum(WEIGHT).alias('weighted_rows'),
              F.sum(F.col(SHAP_ROW).cast('long')).alias('shap_rows'))
         .toPandas().set_index(SPLIT))
    for s, r in c.iterrows():
        log.info('Sample %s: %s rows, %s bads, weighted %s rows, %s SHAP rows', s, f"{int(r['rows']):,}",
                 f"{int(r['bads']):,}", f"{r['weighted_rows']:,.0f}", f"{int(r['shap_rows']):,}")
    return {'path': out, 'goods_fraction': goods_fraction, 'shap_fraction': shap_fraction,
            'counts': {s: {k: float(v) for k, v in r.items()} for s, r in c.iterrows()}}

def load_sample(path: Path, feats: List[str], kinds: Dict[str, str], cats: Dict[str, List[str]]):
    """The sample in memory as XGBoost inputs, labels, weights and SHAP flags per split."""
    t = time.time()
    strings = [SPLIT] + [c for c in feats if kinds[c] == 'categorical']
    pdf = pq.read_table(path, columns=[SPLIT, TARGET, WEIGHT, SHAP_ROW, *feats], read_dictionary=strings).to_pandas()
    parts = {}
    for s in ('train', 'valid'):
        d = pdf[pdf[SPLIT] == s]
        parts[s] = {'X': to_frame(d, feats, kinds, cats).reset_index(drop=True),
                    'y': d[TARGET].to_numpy(np.float32), 'w': d[WEIGHT].to_numpy(np.float32),
                    'shap': d[SHAP_ROW].to_numpy(bool)}
    del pdf
    log.info('Sample loaded in %.0fs: train %s rows, valid %s rows', time.time() - t,
             f"{len(parts['train']['y']):,}", f"{len(parts['valid']['y']):,}")
    return parts

def fit(parts, keep: List[str], params: Dict[str, Any], rounds: int, early: int) -> xgb.Booster:
    """XGBoost on the train sample with the kept features, early stopped on the weighted valid sample."""
    tr, va = parts['train'], parts['valid']
    dtrain = xgb.QuantileDMatrix(tr['X'][keep], tr['y'], weight=tr['w'], enable_categorical=True,
                                 max_bin=params['max_bin'])
    dvalid = xgb.QuantileDMatrix(va['X'][keep], va['y'], weight=va['w'], enable_categorical=True, ref=dtrain)
    return xgb.train(params, dtrain, rounds, evals=[(dvalid, 'valid')], early_stopping_rounds=early, verbose_eval=False)

def sample_ap(b: xgb.Booster, part, keep: List[str]) -> float:
    """Weighted AP of a booster on a sample split."""
    d = xgb.DMatrix(part['X'][keep], enable_categorical=True)
    p = b.predict(d, iteration_range=(0, b.best_iteration + 1))
    return float(average_precision_score(part['y'], p, sample_weight=part['w']))

def mean_abs_shap(b: xgb.Booster, part, keep: List[str]) -> pd.Series:
    """Each feature's weighted mean |SHAP| on the valid SHAP rows."""
    m = part['shap']
    d = xgb.DMatrix(part['X'].loc[m, keep], enable_categorical=True)
    c = b.predict(d, pred_contribs=True, iteration_range=(0, b.best_iteration + 1))[:, :-1]
    w = part['w'][m].astype(np.float64)
    return pd.Series(np.abs(c).T @ w / w.sum(), index=keep)

def full_ap(df: DataFrame, b: xgb.Booster, keep: List[str], kinds: Dict[str, str],
            cats: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
    """AP on every train and valid row of the full data, scored in Spark with the booster broadcast, on margins
    rounded to MARGIN_DECIMALS."""
    sc = df.sparkSession.sparkContext
    bc = sc.broadcast((bytes(b.save_raw('ubj')), b.best_iteration, keep, kinds,
                       {c: cats[c] for c in keep if kinds[c] == 'categorical'}))

    def score(batches):
        raw, best, feats, kd, ct = bc.value
        booster = xgb.Booster()
        booster.load_model(bytearray(raw))
        booster.set_param({'nthread': 1})
        frames, acc = [], None
        for pdf in batches:
            m = booster.predict(xgb.DMatrix(to_frame(pdf, feats, kd, ct), enable_categorical=True, nthread=1),
                                output_margin=True, iteration_range=(0, best + 1))
            frames.append(pd.DataFrame({SPLIT: pdf[SPLIT].to_numpy(), 'm': np.round(m.astype(np.float64),
                                                                                    MARGIN_DECIMALS),
                                        'y': pdf[TARGET].to_numpy(np.int64)})
                          .groupby([SPLIT, 'm'], sort=False)['y'].agg(n='size', bad='sum').reset_index())
            if len(frames) >= 50:
                acc = pd.concat(([acc] if acc is not None else []) + frames).groupby([SPLIT, 'm'], sort=False)[
                    ['n', 'bad']
                ].sum().reset_index()
                frames = []
        if frames or acc is not None:
            yield pd.concat(([acc] if acc is not None else []) + frames).groupby([SPLIT, 'm'], sort=False)[
                ['n', 'bad']
            ].sum().reset_index()

    s = (df.where(F.col(SPLIT).isin('train', 'valid')).select(SPLIT, TARGET, *keep)
         .mapInPandas(score, 'split string, m double, n long, bad long')
         .groupBy(SPLIT, 'm').agg(F.sum('n').alias('n'), F.sum('bad').alias('bad'))
         .toPandas())
    bc.unpersist()
    out = {}
    for split, g in s.groupby(SPLIT):
        n, bad = g['n'].to_numpy(float), g['bad'].to_numpy(float)
        out[split] = {'rows': int(n.sum()), 'bads': int(bad.sum()), 'bad_rate': float(bad.sum() / n.sum()),
                      'average_precision': average_precision(g['m'].to_numpy(), bad, n)}
    return out

def elbow(steps: List[dict], tolerance: float) -> Dict[str, Any]:
    """The suggested step: the fewest features whose full valid AP is within tolerance of the best."""
    curve = pd.DataFrame([{'n': s['n_features'], 'ap': s['ap_full_valid']} for s in steps])
    best = curve.loc[curve['ap'].idxmax()]
    ok = curve[curve['ap'] >= best['ap'] * (1 - tolerance)]
    pick = ok.loc[ok['n'].idxmin()]
    return {'best_n': int(best['n']), 'best_ap_valid': float(best['ap']), 'tolerance': tolerance,
            'suggested_n': int(pick['n']), 'suggested_ap_valid': float(pick['ap'])}

def write_doc(path: Path, result: dict) -> None:
    m, e, steps = result['meta'], result['elbow'], result['steps']
    s = m['sample']['counts']
    preamble = (
        f"Backward elimination with XGBoost on the **raw** features, from the {m['candidates']} candidates "
        f"left by `12_gbm_corr_selection_main`. Each step fits one model, scores it, and drops the feature with "
        f"the lowest **mean |SHAP|** (lowest univariate AP on a tie), until {m['min_features']} remain.\n\n"
        f"- **Fit** on every train bad and {m['sample']['goods_fraction']:.0%} of train goods (goods weighted "
        f"{1 / m['sample']['goods_fraction']:g}): {s['train']['rows']:,.0f} rows standing for "
        f"{s['train']['weighted_rows']:,.0f}. Early stopping on the valid sample, drawn the same way "
        f"({s['valid']['rows']:,.0f} rows), weighted AUC-PR, patience {m['early_stopping']}, at most "
        f"{m['num_rounds']} rounds. Fixed parameters: "
        + ', '.join(f'{k}={v}' for k, v in m['params'].items() if k not in ('objective', 'eval_metric'))
        + ".\n"
        f"- **Mean |SHAP|** (TreeSHAP, weighted) on {s['valid']['shap_rows']:,.0f} rows of the valid sample.\n"
        f"- **AP full**: the model scored on **every** train and valid row (train {m['full_rows']['train']:,}, "
        f"valid {m['full_rows']['valid']:,}), unweighted, margin rounded to 1e-{MARGIN_DECIMALS}. "
        f"AP sample: on the weighted samples.\n\n"
        f"**Elbow**: best full valid AP {e['best_ap_valid']:.4f} with {e['best_n']} features; the fewest features "
        f"within {e['tolerance']:.1%} of it: **{e['suggested_n']}** (valid AP {e['suggested_ap_valid']:.4f}). "
        f"This is a suggestion; the curve is below.")
    curve = pd.DataFrame([{'step': x['step'], 'features': x['n_features'], 'rounds': x['best_iteration'] + 1,
                           'AP full train': x['ap_full_train'], 'AP full valid': x['ap_full_valid'],
                           'train - valid': x['ap_full_train'] - x['ap_full_valid'],
                           'AP sample train': x['ap_sample_train'], 'AP sample valid': x['ap_sample_valid'],
                           'dropped next': x.get('dropped') or '--',
                           'its mean |SHAP|': x.get('dropped_shap', float('nan'))} for x in steps])
    chosen = next(x for x in steps if x['n_features'] == e['suggested_n'])
    imp = pd.Series(chosen['shap']).sort_values(ascending=False).rename('mean |SHAP|').rename_axis('feature').reset_index()
    write_report(path, 'XGBoost Feature Selection: SHAP Backward Elimination',
                 'src/jobs/13_gbm_shap_elimination_main.py',
                 [('Elimination curve', md_table(curve, floatfmt='{:.4f}')),
                  (f"Suggested features ({e['suggested_n']}), by mean |SHAP|", md_table(imp, floatfmt='{:.5f}'))],
                 preamble)

def _save(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state['saved'] = datetime.now().isoformat(timespec='seconds')
    path.write_text(json.dumps(state, indent=2, default=float))

def main() -> None:
    p = argparse.ArgumentParser(description='gbm_shap_elimination_main')
    p.add_argument('--stage', choices=('all', 'sample', 'eliminate', 'report'), default='all',
                   help='sample: draw the elimination sample; eliminate: the backward elimination; '
                        'report: rewrite the report from --progress-json')
    p.add_argument('--training-set', default=curated_path('training_set'))
    p.add_argument('--split-json', default=str(settings.model_dir / 'gbm_split.json'))
    p.add_argument('--selection-json', default=str(settings.model_dir / 'gbm_corr_selection.json'),
                   help='candidates from 12_gbm_corr_selection_main')
    p.add_argument('--univariate-json', default=str(settings.model_dir / 'gbm_univariate.json'),
                   help='category lists and univariate AP (the tie breaker)')
    p.add_argument('--sample-out', default=curated_path('gbm_elimination_sample'),
                   help='on HDFS; the elimination copies it to a local temp dir to load it into memory')
    p.add_argument('--goods-fraction', type=float, default=GOODS_FRACTION)
    p.add_argument('--shap-fraction', type=float, default=SHAP_FRACTION)
    p.add_argument('--learning-rate', type=float, default=PARAMS['learning_rate'])
    p.add_argument('--max-depth', type=int, default=PARAMS['max_depth'])
    p.add_argument('--num-rounds', type=int, default=NUM_ROUNDS)
    p.add_argument('--early-stopping', type=int, default=EARLY_STOPPING)
    p.add_argument('--min-features', type=int, default=1, help='stop when this many features are left')
    p.add_argument('--elbow-tolerance', type=float, default=ELBOW_TOLERANCE)
    p.add_argument('--progress-json', default=str(settings.model_dir / 'gbm_elimination_progress.json'))
    p.add_argument('--resume', action='store_true', help='continue from --progress-json')
    p.add_argument('--models-dir', default=str(settings.model_dir / 'gbm_elimination_models'),
                   help="every step's model, step_NN.ubj")
    p.add_argument('--result-json', default=str(settings.model_dir / 'gbm_elimination.json'))
    p.add_argument('--doc', default=str(Path(__file__).resolve().parents[2] / 'docs' / 'gbm_elimination.md'))
    a = p.parse_args()
    if not 0 < a.goods_fraction <= 1 or not 0 < a.shap_fraction <= 1:
        raise SystemExit('--goods-fraction and --shap-fraction must be in (0, 1].')

    cfg = load_split(Path(a.split_json))['config']
    sel = json.loads(Path(a.selection_json).read_text())
    uj = json.loads(Path(a.univariate_json).read_text())
    kinds, cats = sel['kinds'], uj['categories']
    candidates = sel['candidates']
    uni_ap = pd.Series({r['feature']: r['univariate_ap_valid'] for r in uj['features']})
    params = {**PARAMS, 'learning_rate': a.learning_rate, 'max_depth': a.max_depth}
    progress, sample_dir = Path(a.progress_json), a.sample_out
    meta_file = sample_meta_path(sample_dir)

    state: Dict[str, Any] = {}
    if a.stage in ('all', 'eliminate', 'report'):
        if a.resume or a.stage == 'report':
            if not progress.exists():
                raise SystemExit(f'Nothing to resume: no {progress}.')
            state = json.loads(progress.read_text())
            if not set(state['keep']) <= set(candidates):
                raise SystemExit(f'{progress} is from another candidate set: rerun without --resume.')
            log.info('Resuming from %s: %s steps done, %s features left', progress, len(state['steps']),
                     len(state['keep']))

    if a.stage != 'report':
        spark = spark_session('gbm_shap_elimination',
                              conf={'spark.sql.execution.arrow.pyspark.enabled': 'true',
                                    'spark.driver.maxResultSize': '4g'})
        try:
            dev = dev_rows(spark, a.training_set, cfg)
            if a.stage in ('all', 'sample'):
                t = time.time()
                smeta = build_sample(dev, candidates, kinds, a.goods_fraction, a.shap_fraction, sample_dir)
                smeta['features'] = candidates
                meta_file.write_text(json.dumps(smeta, indent=2))
                log.info('Sample written to %s in %.0fs', sample_dir, time.time() - t)

            if a.stage in ('all', 'eliminate'):
                if not meta_file.exists():
                    raise SystemExit(f'No sample at {sample_dir}: run --stage sample first.')
                smeta = json.loads(meta_file.read_text())
                if not set(candidates) <= set(smeta['features']):
                    raise SystemExit(f'{sample_dir} lacks candidates: rerun --stage sample.')
                if not state:
                    state = {'keep': list(candidates), 'steps': [], 'done': False}
                with local_copy(sample_dir) as local:
                    parts = load_sample(local, candidates, kinds, cats)
                models = Path(a.models_dir)
                models.mkdir(parents=True, exist_ok=True)
                while not state['done']:
                    keep = state['keep']
                    step = len(state['steps']) + 1
                    t0 = time.time()
                    b = fit(parts, keep, params, a.num_rounds, a.early_stopping)
                    t_fit = time.time() - t0
                    b.save_model(str(models / f'step_{step:02d}.ubj'))
                    ap_tr, ap_va = sample_ap(b, parts['train'], keep), sample_ap(b, parts['valid'], keep)
                    shap = mean_abs_shap(b, parts['valid'], keep)
                    t = time.time()
                    full = full_ap(dev, b, keep, kinds, cats)
                    t_full = time.time() - t
                    rec = {'step': step, 'n_features': len(keep), 'features': list(keep),
                           'best_iteration': int(b.best_iteration),
                           'ap_sample_train': ap_tr, 'ap_sample_valid': ap_va,
                           'ap_full_train': full['train']['average_precision'],
                           'ap_full_valid': full['valid']['average_precision'],
                           'full': full, 'shap': shap.to_dict(),
                           'seconds_fit': round(t_fit), 'seconds_full_ap': round(t_full)}
                    if len(keep) > max(a.min_features, 1):
                        low = shap[shap <= shap.min() + SHAP_TIE].index
                        worst = uni_ap[low].sort_values(kind='stable').index[0] if len(low) > 1 else low[0]
                        rec.update({'dropped': worst, 'dropped_shap': float(shap[worst]),
                                    'tied_at_min': int(len(low))})
                        keep.remove(worst)
                    else:
                        state['done'] = True
                    state['steps'].append(rec)
                    _save(progress, state)
                    log.info('Step %s: %s features, %s rounds, AP full train %.4f / valid %.4f, sample %.4f / '
                             '%.4f; drop %s (mean |SHAP| %.5f%s); fit %.0fs, full AP %.0fs', step, rec['n_features'],
                             rec['best_iteration'] + 1, rec['ap_full_train'], rec['ap_full_valid'], ap_tr, ap_va,
                             rec.get('dropped', '--'), rec.get('dropped_shap', float('nan')),
                             f", tie of {rec['tied_at_min']} broken by univariate AP"
                             if rec.get('tied_at_min', 1) > 1 else '', t_fit, t_full)
        finally:
            spark.stop()

    if a.stage in ('all', 'eliminate', 'report') and state.get('steps'):
        steps = state['steps']
        e = elbow(steps, a.elbow_tolerance)
        chosen = next(x for x in steps if x['n_features'] == e['suggested_n'])
        smeta = json.loads(meta_file.read_text())
        result = {
            'meta': {'split': cfg, 'candidates': len(candidates), 'min_features': a.min_features,
                     'params': params, 'num_rounds': a.num_rounds, 'early_stopping': a.early_stopping,
                     'sample': smeta, 'full_rows': {s: steps[0]['full'][s]['rows'] for s in ('train', 'valid')},
                     'rule': 'drop the lowest mean |SHAP| (valid sample, weighted), lowest univariate AP on a tie',
                     'metric': 'average precision on every train / valid row, margin rounded to 1e-4',
                     'complete': bool(state['done']),
                     'generated': datetime.now().isoformat(timespec='seconds'),
                     'source': 'src/jobs/13_gbm_shap_elimination_main.py'},
            'elbow': e, 'suggested_features': chosen['features'],
            'steps': [{k: v for k, v in x.items() if k != 'features'} for x in steps],
        }
        out = Path(a.result_json)
        out.write_text(json.dumps(result, indent=2, default=float))
        write_doc(Path(a.doc), {**result, 'steps': steps})
        log.info('Elbow: best valid AP %.4f at %s features; suggested %s features (AP %.4f). Result in %s, '
                 'report in %s', e['best_ap_valid'], e['best_n'], e['suggested_n'], e['suggested_ap_valid'],
                 out, a.doc)

if __name__ == '__main__':
    main()