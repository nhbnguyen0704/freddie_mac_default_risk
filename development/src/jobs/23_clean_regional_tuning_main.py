import argparse
import importlib
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import optuna
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from pyspark.sql import functions as F
from helpers.config import settings
from helpers import hdfs
from helpers.gbm import SPLIT, TARGET, load_split, to_frame, with_split
from helpers.report import md_table, write_report
from helpers.spark_io import curated_path, resolve_path, spark_session

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s | %(message)s')
log = logging.getLogger('clean_regional_tuning_main')
optuna.logging.set_verbosity(optuna.logging.WARNING)

g15 = importlib.import_module('jobs.15_gbm_tuning_main')
g20 = importlib.import_module('jobs.20_regional_features_main')
g22 = importlib.import_module('jobs.22_segment_merge_main')

NAME = 'clean_regional_tuned'
VERSION = g15.Version(NAME, 'recency', g20.HALF_LIFE)
SEED_OFFSET = 4

os.environ.setdefault('PYSPARK_PYTHON', sys.executable)


def tune(a, cfg: dict, feats, kinds, cats, start: dict) -> None:
    """clean_regional re-tuned as job 15 tunes a version: same search space, pruning, trial count and objective
    (weighted AP on the clean valid sample), on job 20's clean sample with recency weights (half-life 12). Trial
    0 is recency_12's hyperparameters, the ones job 20 reused, so the search starts from the current model; the
    sampler is seeded as recency_12's was in job 15."""
    strings = [SPLIT] + [c for c in feats if kinds[c] == 'categorical']
    with hdfs.local_copy(a.sample) as local:
        pdf = pq.read_table(local, read_dictionary=strings).to_pandas()
    train, valid = pdf[pdf[SPLIT] == 'train'], pdf[pdf[SPLIT] == 'valid']
    del pdf
    w = train[g20.WEIGHT].to_numpy(np.float64) * 0.5 ** (
        g15.months_before(train['ym'].to_numpy(), int(cfg['end_ym'])) / g20.HALF_LIFE)
    y = train[TARGET].to_numpy(np.float32)
    vy, vw = valid[TARGET].to_numpy(np.float32), valid[g20.WEIGHT].to_numpy(np.float32)
    dtrain = xgb.QuantileDMatrix(to_frame(train, feats, kinds, cats), y, weight=w.astype(np.float32),
                                 enable_categorical=True, max_bin=256)
    dvalid = xgb.QuantileDMatrix(to_frame(valid, feats, kinds, cats), vy, weight=vw, enable_categorical=True,
                                 ref=dtrain)
    del train, valid
    log.info('Clean sample: %s train rows, %s valid rows, %s features', f'{len(y):,}', f'{len(vy):,}', len(feats))
    study = optuna.create_study(
        study_name=f'gbm_{NAME}', storage=a.storage, load_if_exists=True, direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=g15.SEED + SEED_OFFSET, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=g15.PRUNE_STARTUP, n_warmup_steps=g15.PRUNE_WARMUP,
                                           interval_steps=g15.PRUNE_EVERY))
    if not study.trials:
        study.enqueue_trial(start)
    t = time.time()
    g15.tune(VERSION, dtrain, dvalid, vy, vw, study, a.n_trials, Path(a.models_dir) / f'{NAME}.ubj',
             a.num_rounds, a.early_stopping)
    log.info('[%s] best valid AP (sample) %.4f (trial %s); %.0f min', NAME, study.best_value,
             study.best_trial.number, (time.time() - t) / 60)

def score(spark, a, cfg: dict, feats, kinds, cats) -> pd.DataFrame:
    """(version, split, m) -> n, bad on every clean valid / test row."""
    df = with_split(spark.read.parquet(resolve_path(a.training_set)), cfg)
    df = g20.with_regional(spark, df.where(g20.clean() & F.col(SPLIT).isin('valid', 'test')), a)
    mp = Path(a.models_dir) / f'{NAME}.ubj'
    info = json.loads(mp.with_suffix('.json').read_text())
    t = time.time()
    s = g15.score_all(df, {NAME: (mp.read_bytes(), int(info['best_iteration']))}, feats, kinds, cats)
    log.info('Clean rows scored in %.0f min', (time.time() - t) / 60)
    return s.assign(segment='clean')

def write_doc(path: Path, r: dict) -> None:
    before, after = r['before'], r['after']
    rows = []
    for what, key in [('clean valid AP', 'valid/clean'), ('clean test AP', 'test/clean')]:
        b, n = before['segment_ap'][key], after['segment_ap'][key]
        rows.append({'measure': what, g22.CLEAN_MODEL: b, NAME: n, 'change': f'{n - b:+.4f}'})
    for sp in ('valid', 'test'):
        for k, lab in (('raw', 'raw'), ('platt_valid', 'Platt (valid)')):
            b, n = before[sp][k], after[sp][k]
            rows.append({'measure': f'merged {sp} AP, {lab}', g22.CLEAN_MODEL: b, NAME: n, 'change': f'{n - b:+.4f}'})
    t = pd.DataFrame(rows)
    keys = [k for k in r['tuned_params'] if k not in ('eval_metric', 'tree_method', 'max_bin', 'seed', 'objective')]
    params = pd.DataFrame([{'hyperparameter': k, f'{g20.VERSION} (reused by job 20)': r['start_params'].get(k),
                            NAME: r['tuned_params'][k]} for k in keys])
    st = r['study']
    pre = (f"Job 20's **{g22.CLEAN_MODEL}** (clean rows, {len(r['features'])} features with the regional ones) "
           f"reused `{g20.VERSION}`'s hyperparameters. **{NAME}** re-tunes them as job 15 tunes a version: the same "
           f"search space, Optuna TPE, median pruning, {st['trials_complete'] + st['trials_pruned']} trials "
           f"({st['trials_complete']} complete, {st['trials_pruned']} pruned), objective weighted AP on the clean "
           f"valid sample. Trial 0 is `{g20.VERSION}`'s hyperparameters; the best is trial {st['best_trial']} "
           f"(valid AP on the sample {st['best_value']:.4f}, {r['best_iteration'] + 1} rounds).\n\n"
           f"- Clean AP is on every clean valid / test row. The merged ranking is job 22's: the clean model on clean "
           f"rows, `{g22.DELINQUENT_MODEL}` on delinquent rows (its scores reused from job 22). Current single model: "
           f"test AP {r['reference']['test']:.4f}.\n"
           f"- The valid sample picks the trial and stops each fit, so valid AP is optimistic; test is untouched.")
    write_report(path, 'Clean Regional Model: Re-Tuned', 'src/jobs/23_clean_regional_tuning_main.py',
                 [('Before and after re-tuning', md_table(t, floatfmt='{:.4f}', numparse=False)),
                  ('Hyperparameters', md_table(params, floatfmt='{:.4g}'))], pre)


def main() -> None:
    p = argparse.ArgumentParser(description='clean_regional_tuning_main')
    p.add_argument('--stage', choices=('all', 'tune', 'score', 'merge'), default='all')
    p.add_argument('--training-set', default=curated_path('training_set'))
    p.add_argument('--regional', default=curated_path('regional_features'), help='written by job 20')
    p.add_argument('--sample', default=curated_path('clean_regional_sample'), help='written by job 20')
    p.add_argument('--split-json', default=str(settings.model_dir / 'gbm_split.json'))
    p.add_argument('--elimination-json', default=str(settings.model_dir / 'gbm_elimination.json'))
    p.add_argument('--selection-json', default=str(settings.model_dir / 'gbm_corr_selection.json'))
    p.add_argument('--univariate-json', default=str(settings.model_dir / 'gbm_univariate.json'))
    p.add_argument('--gbm-models-dir', default=str(settings.model_dir / 'gbm_tuning_models'))
    p.add_argument('--models-dir', default=str(settings.model_dir / 'regional_models'))
    p.add_argument('--n-trials', type=int, default=g15.N_TRIALS)
    p.add_argument('--num-rounds', type=int, default=g15.NUM_ROUNDS)
    p.add_argument('--early-stopping', type=int, default=g15.EARLY_STOPPING)
    p.add_argument('--storage', default=f"sqlite:///{settings.model_dir / 'clean_regional_tuning.db'}",
                   help='Optuna storage; a rerun continues the study where it stopped')
    p.add_argument('--merge-tables', default=str(settings.model_dir / 'segment_merge_tables.parquet'),
                   help="job 22's score tables (the delinquent model's are reused)")
    p.add_argument('--tables', default=str(settings.model_dir / 'clean_regional_tuned_tables.parquet'))
    p.add_argument('--result-json', default=str(settings.model_dir / 'clean_regional_tuning.json'))
    p.add_argument('--doc', default=str(Path(__file__).resolve().parents[2] / 'docs' / 'clean_regional_tuning.md'))
    a = p.parse_args()

    cfg = load_split(Path(a.split_json))['config']
    feats = json.loads(Path(a.elimination_json).read_text())['suggested_features'] + g20.REGIONAL
    kinds = json.loads(Path(a.selection_json).read_text())['kinds'] | {c: 'numerical' for c in g20.REGIONAL}
    cats = json.loads(Path(a.univariate_json).read_text())['categories']
    start_params = json.loads((Path(a.gbm_models_dir) / f'{g20.VERSION}.json').read_text())['params']
    start = {k: v for k, v in start_params.items()
             if k in ('learning_rate', 'max_depth', 'min_child_weight', 'subsample', 'colsample_bytree', 'gamma',
                      'reg_lambda', 'reg_alpha')}
    Path(a.models_dir).mkdir(parents=True, exist_ok=True)

    if a.stage in ('all', 'tune'):
        tune(a, cfg, feats, kinds, cats, start)
    if a.stage in ('all', 'score'):
        spark = spark_session('clean_regional_tuning', conf={'spark.sql.execution.arrow.pyspark.enabled': 'true',
                                                             'spark.driver.maxResultSize': '4g'})
        try:
            score(spark, a, cfg, feats, kinds, cats).to_parquet(a.tables)
        finally:
            spark.stop()
    if a.stage in ('all', 'merge'):
        old = pd.read_parquet(a.merge_tables)
        new = pd.concat([old[old['segment'] == 'delinquent'], pd.read_parquet(a.tables)], ignore_index=True)
        before, after = g22.merge(old), g22.merge(new)
        study = optuna.load_study(study_name=f'gbm_{NAME}', storage=a.storage)
        states = [x.state for x in study.trials]
        info = json.loads((Path(a.models_dir) / f'{NAME}.json').read_text())
        full = json.loads((Path(a.gbm_models_dir) / 'full_ap.json').read_text())[g20.VERSION]
        result = {'meta': {'split': cfg, 'source': 'src/jobs/23_clean_regional_tuning_main.py',
                           'generated': datetime.now().isoformat(timespec='seconds')},
                  'features': feats, 'start_params': start_params, 'tuned_params': info['params'],
                  'best_iteration': info['best_iteration'],
                  'study': {'trials_complete': states.count(optuna.trial.TrialState.COMPLETE),
                            'trials_pruned': states.count(optuna.trial.TrialState.PRUNED),
                            'best_trial': study.best_trial.number, 'best_value': study.best_value},
                  'reference': {sp: full[sp]['average_precision'] for sp in ('valid', 'test')},
                  'before': before, 'after': after}
        Path(a.result_json).write_text(json.dumps(result, indent=2, default=float))
        write_doc(Path(a.doc), result)
        for k in ('valid/clean', 'test/clean'):
            log.info('%s AP: %.4f -> %.4f', k, before['segment_ap'][k], after['segment_ap'][k])
        for sp in ('valid', 'test'):
            log.info('merged %s AP (Platt, valid): %.4f -> %.4f (current single model %.4f)', sp,
                     before[sp]['platt_valid'], after[sp]['platt_valid'], result['reference'][sp])
        log.info('Result in %s, report in %s', a.result_json, a.doc)


if __name__ == '__main__':
    main()
