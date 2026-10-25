import argparse
import json
import logging
import os
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import optuna
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from pyspark.sql import DataFrame, functions as F
from scipy.special import expit
from sklearn.metrics import average_precision_score
from helpers.config import settings
from helpers import hdfs
from helpers.gbm import (SPLIT, TARGET, average_precision, load_split, row_id, sample_meta_path, to_frame,
                         uniform, with_split, ym)
from helpers.report import md_table, write_report
from helpers.spark_io import curated_path, resolve_path, spark_session

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s | %(message)s')
log = logging.getLogger('gbm_tuning_main')
optuna.logging.set_verbosity(optuna.logging.WARNING)

WEIGHT = 'sample_weight'

N_TRIALS = 30
NUM_ROUNDS = 2000
EARLY_STOPPING = 50
PRUNE_EVERY = 10
PRUNE_STARTUP = 5
PRUNE_WARMUP = 100
SEED = 42
MARGIN_DECIMALS = 4
FOCAL_EPS = 1e-4


@dataclass(frozen=True)
class Version:
    """One XGBoost version: its name, kind (none, spw, focal, recency, oversample, undersample, smoteenn or
    smotetomek) and, for recency, the half-life in months."""
    name: str
    kind: str
    half_life: float = 0.0

    @property
    def label(self) -> str:
        return {'none': 'logloss, no reweighting',
                'spw': 'logloss, scale_pos_weight tuned',
                'focal': 'focal loss, alpha and gamma tuned',
                'recency': f'recency weight, half-life {self.half_life:g} months',
                'oversample': 'random oversampling of bads in lower-rate months',
                'undersample': 'random undersampling of goods in lower-rate months',
                'smoteenn': 'SMOTE then edited nearest neighbours (train split, per month)',
                'smotetomek': 'SMOTE then Tomek links (train split, per month)'}[self.kind]

VERSIONS = [Version('baseline', 'none'), Version('scale_pos_weight', 'spw'), Version('focal_loss', 'focal'),
            Version('recency_6', 'recency', 6), Version('recency_12', 'recency', 12),
            Version('recency_25', 'recency', 25), Version('oversample', 'oversample'),
            Version('undersample', 'undersample'), Version('smoteenn', 'smoteenn'),
            Version('smotetomek', 'smotetomek')]

os.environ.setdefault('PYSPARK_PYTHON', sys.executable)


def build_sample(df: DataFrame, feats: List[str], kinds: Dict[str, str], goods_fraction: float,
                 out: str) -> Dict[str, Any]:
    """The elimination sample's rows (same draw), with what the versions need:
    the month, a row id (to drop goods removed by ENN / Tomek) and two seeded
    uniforms for the over/undersampling draws."""
    loan, period = F.col('loan_identifier'), F.col('period')
    keep = (F.col(TARGET) == 1) | (uniform(loan, period, salt='gbm_goods') < F.lit(goods_fraction))
    sample = (df.where(F.col(SPLIT).isin('train', 'valid') & keep)
              .select(SPLIT, F.col(TARGET).cast('tinyint').alias(TARGET),
                      F.when(F.col(TARGET) == 1, F.lit(1.0)).otherwise(F.lit(1 / goods_fraction)).alias(WEIGHT),
                      ym().cast('int').alias('ym'), row_id().alias('rid'),
                      uniform(loan, period, salt='gbm_over').cast('float').alias('u_over'),
                      uniform(loan, period, salt='gbm_under').cast('float').alias('u_under'),
                      *[F.col(c).cast('float').alias(c) if kinds[c] == 'numerical' else F.col(c) for c in feats]))
    sample.coalesce(64).write.mode('overwrite').parquet(out)
    months = (df.where(F.col(SPLIT) == 'train').groupBy(ym().alias('ym'))
              .agg(F.count(F.lit(1)).alias('n'), F.sum(TARGET).alias('bad')).toPandas()
              .sort_values('ym').reset_index(drop=True))
    c = (df.sparkSession.read.parquet(out).groupBy(SPLIT)
         .agg(F.count(F.lit(1)).alias('rows'), F.sum(TARGET).alias('bads'), F.sum(WEIGHT).alias('weighted_rows'))
         .toPandas().set_index(SPLIT))
    for s, r in c.iterrows():
        log.info('Sample %s: %s rows, %s bads, weighted %s rows', s, f"{int(r['rows']):,}", f"{int(r['bads']):,}",
                 f"{r['weighted_rows']:,.0f}")
    return {'path': out, 'goods_fraction': goods_fraction, 'features': feats,
            'counts': {s: {k: float(v) for k, v in r.items()} for s, r in c.iterrows()},
            'train_months': {int(r.ym): [int(r.n), int(r.bad)] for r in months.itertuples()}}

def load_sample(path: Path, feats: List[str], kinds: Dict[str, str]) -> pd.DataFrame:
    """The tuning sample in memory, string columns as dictionaries."""
    strings = [SPLIT] + [c for c in feats if kinds[c] == 'categorical']
    t = time.time()
    pdf = pq.read_table(path, read_dictionary=strings).to_pandas()
    log.info('Sample loaded in %.0fs: %s rows', time.time() - t, f'{len(pdf):,}')
    return pdf


def month_plan(train_months: Dict[int, List[int]]) -> Tuple[float, Dict[int, float], Dict[int, float]]:
    """As 10_lr_model_main.weight_plan: bring every train month up to the
    highest monthly bad rate r*. Oversampling repeats each bad of month m
    k_m = r* G_m / ((1 - r*) B_m) times on average; undersampling keeps each
    good with probability f_m = B_m (1 - r*) / (r* G_m)."""
    m = pd.DataFrame([(k, n, b) for k, (n, b) in train_months.items()], columns=['ym', 'n', 'bad'])
    target = float((m['bad'] / m['n']).max())
    good = m['n'] - m['bad']
    k = (target * good / ((1 - target) * m['bad'])).clip(lower=1.0)
    f = (m['bad'] * (1 - target) / (target * good)).clip(upper=1.0)
    return target, dict(zip(m['ym'], k)), dict(zip(m['ym'], f))

def months_before(ym_: np.ndarray, end_ym: int) -> np.ndarray:
    """Months from each YYYYMM in ym_ to end_ym."""
    return (end_ym // 100 - ym_ // 100) * 12 + (end_ym % 100 - ym_ % 100)

def version_rows(v: Version, train: pd.DataFrame, plan, end_ym: int, resampled: Path,
                 feats: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
    """(rows, weights) a version is fitted on: the sample's weights times the
    version's. Over/undersampling are seeded random draws applied as
    frequency weights, as the LR does."""
    y = train[TARGET].to_numpy()
    w0 = train[WEIGHT].to_numpy(np.float64)
    if v.kind in ('none', 'spw', 'focal'):
        return train, w0
    if v.kind == 'recency':
        return train, w0 * 0.5 ** (months_before(train['ym'].to_numpy(), end_ym) / v.half_life)
    _, k, f = plan
    if v.kind == 'oversample':
        km = train['ym'].map(k).fillna(1.0).to_numpy()
        copies = np.floor(km) + (train['u_over'].to_numpy() < km - np.floor(km))
        return train, np.where(y == 1, copies, w0)
    if v.kind == 'undersample':
        fm = train['ym'].map(f).fillna(1.0).to_numpy()
        w = np.where(y == 1, w0, w0 * (train['u_under'].to_numpy() < fm))
        keep = w > 0
        return train[keep], w[keep]
    removed = pd.concat([pd.read_parquet(p) for p in sorted(resampled.glob('month=*/removed.parquet'))])
    gone = train['rid'].isin(removed.loc[removed['version'] == v.kind, 'rid'])
    syn = pd.concat([pd.read_parquet(p) for p in sorted(resampled.glob('month=*/synthetic.parquet'))],
                    ignore_index=True)
    rows = pd.concat([train.loc[~gone, [TARGET, WEIGHT, *feats]], syn[[TARGET, WEIGHT, *feats]]],
                     ignore_index=True)
    log.info('[%s] %s sampled goods removed, %s synthetic bads added (weight %g)', v.name, f'{int(gone.sum()):,}',
             f'{len(syn):,}', float(syn[WEIGHT].iloc[0]) if len(syn) else float('nan'))
    return rows, rows[WEIGHT].to_numpy(np.float64)


def focal_objective(alpha: float, gamma: float) -> Callable:
    """Focal loss L = -a y (1-p)^g log p - (1-a)(1-y) p^g log(1-p), p = sigmoid(margin).
    Gradient in closed form; the Hessian by a central difference of it, floored
    at a small positive value (the loss is not convex for g > 0)."""
    def grad(z: np.ndarray, y: np.ndarray) -> np.ndarray:
        p = np.clip(expit(z), 1e-12, 1 - 1e-12)
        g1 = alpha * (gamma * p * (1 - p) ** gamma * np.log(p) - (1 - p) ** (gamma + 1))
        g0 = (1 - alpha) * (p ** (gamma + 1) - gamma * p ** gamma * (1 - p) * np.log(1 - p))
        return np.where(y == 1, g1, g0)

    def obj(z: np.ndarray, d: xgb.DMatrix):
        y, w = d.get_label(), d.get_weight()
        z = z.astype(np.float64)
        g = grad(z, y)
        h = np.maximum((grad(z + FOCAL_EPS, y) - grad(z - FOCAL_EPS, y)) / (2 * FOCAL_EPS), 1e-9)
        if len(w):
            g, h = g * w, h * w
        return g, h
    return obj

def suggest(trial: optuna.Trial, v: Version) -> Tuple[Dict[str, Any], Optional[Callable]]:
    """One trial's XGBoost parameters, plus the focal objective for the focal version (logistic otherwise) and
    scale_pos_weight for the spw version."""
    p = {'eval_metric': 'aucpr', 'tree_method': 'hist', 'max_bin': 256, 'seed': SEED,
         'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.3, log=True),
         'max_depth': trial.suggest_int('max_depth', 3, 10),
         'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 1000.0, log=True),
         'subsample': trial.suggest_float('subsample', 0.5, 1.0),
         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
         'gamma': trial.suggest_float('gamma', 0.0, 5.0),
         'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 100.0, log=True),
         'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True)}
    obj = None
    if v.kind == 'focal':
        obj = focal_objective(trial.suggest_float('focal_alpha', 0.05, 0.95),
                              trial.suggest_float('focal_gamma', 0.0, 5.0))
    else:
        p['objective'] = 'binary:logistic'
    if v.kind == 'spw':
        p['scale_pos_weight'] = trial.suggest_float('scale_pos_weight', 1.0, 150.0, log=True)
    return p, obj

class PruneCallback(xgb.callback.TrainingCallback):
    """Reports the valid AUC-PR to Optuna every `every` rounds and stops the fit when the trial should be pruned."""
    def __init__(self, trial: optuna.Trial, every: int):
        super().__init__()
        self.trial, self.every = trial, every

    def after_iteration(self, model, epoch: int, evals_log) -> bool:
        if (epoch + 1) % self.every == 0:
            self.trial.report(float(evals_log['valid']['aucpr'][-1]), epoch + 1)
            if self.trial.should_prune():
                raise optuna.TrialPruned()
        return False

def tune(v: Version, dtrain, dvalid, valid_y: np.ndarray, valid_w: np.ndarray, study: optuna.Study,
         n_trials: int, model_path: Path, rounds: int, early: int) -> None:
    """Run the version's Optuna study up to n_trials finished trials, saving the best trial's model and its
    parameters to model_path. Tuned on the elimination sample's design: every bad of train / valid and
    GOODS_FRACTION of the goods, goods weighted back up, each version's weights multiplying these, so the
    objective (weighted AP on the valid sample) estimates AP on the whole valid split. Pruning: the valid AUC-PR
    is reported every PRUNE_EVERY rounds and a trial stops when it is below the median of earlier trials at that
    round, after PRUNE_STARTUP complete trials and PRUNE_WARMUP rounds."""
    def objective(trial: optuna.Trial) -> float:
        t = time.time()
        params, obj = suggest(trial, v)
        b = xgb.train(params, dtrain, rounds, evals=[(dvalid, 'valid')], obj=obj, early_stopping_rounds=early,
                      callbacks=[PruneCallback(trial, PRUNE_EVERY)], verbose_eval=False)
        m = b.predict(dvalid, output_margin=True, iteration_range=(0, b.best_iteration + 1))
        ap = float(average_precision_score(valid_y, m, sample_weight=valid_w))
        trial.set_user_attr('best_iteration', int(b.best_iteration))
        trial.set_user_attr('seconds', round(time.time() - t))
        done = [x for x in study.trials if x.state == optuna.trial.TrialState.COMPLETE]
        if not done or ap > max(x.value for x in done):
            b.save_model(str(model_path))
            json.dump({'params': params, 'trial': trial.number, 'valid_ap_sample': ap,
                       'best_iteration': int(b.best_iteration)},
                      model_path.with_suffix('.json').open('w'), indent=2, default=float)
        log.info('[%s] trial %s: valid AP %.4f, %s rounds, %.0fs', v.name, trial.number, ap, b.best_iteration + 1,
                 time.time() - t)
        return ap

    def pruned(study, trial):
        if trial.state == optuna.trial.TrialState.PRUNED:
            log.info('[%s] trial %s pruned at round %s', v.name, trial.number, trial.last_step)

    finished = sum(x.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
                   for x in study.trials)
    if finished < n_trials:
        study.optimize(objective, n_trials=n_trials - finished, callbacks=[pruned], gc_after_trial=True)


def score_all(df: DataFrame, models: Dict[str, Tuple[bytes, int]], feats: List[str], kinds: Dict[str, str],
              cats: Dict[str, List[str]]) -> pd.DataFrame:
    """(version, split, rounded margin) -> rows, bads in ONE scan for all versions."""
    sc = df.sparkSession.sparkContext
    bc = sc.broadcast((models, feats, kinds, {c: cats[c] for c in feats if kinds[c] == 'categorical'}))

    def score(batches):
        mods, fs, kd, ct = bc.value
        boosters = {}
        for name, (raw, best) in mods.items():
            b = xgb.Booster()
            b.load_model(bytearray(raw))
            b.set_param({'nthread': 1})
            boosters[name] = (b, best)
        acc, frames = None, []
        for pdf in batches:
            d = xgb.DMatrix(to_frame(pdf, fs, kd, ct), enable_categorical=True, nthread=1)
            s, y = pdf[SPLIT].to_numpy(), pdf[TARGET].to_numpy(np.int64)
            for name, (b, best) in boosters.items():
                m = np.round(b.predict(d, output_margin=True, iteration_range=(0, best + 1)).astype(np.float64),
                             MARGIN_DECIMALS)
                frames.append(pd.DataFrame({'version': name, SPLIT: s, 'm': m, 'y': y})
                              .groupby(['version', SPLIT, 'm'], sort=False)['y'].agg(n='size', bad='sum')
                              .reset_index())
            if len(frames) >= 200:
                acc = pd.concat(([acc] if acc is not None else []) + frames).groupby(
                    ['version', SPLIT, 'm'], sort=False)[['n', 'bad']].sum().reset_index()
                frames = []
        if frames or acc is not None:
            yield pd.concat(([acc] if acc is not None else []) + frames).groupby(
                ['version', SPLIT, 'm'], sort=False)[['n', 'bad']].sum().reset_index()

    out = (df.select(SPLIT, TARGET, *feats)
           .mapInPandas(score, 'version string, split string, m double, n long, bad long')
           .groupBy('version', SPLIT, 'm').agg(F.sum('n').alias('n'), F.sum('bad').alias('bad'))
           .toPandas())
    bc.unpersist()
    return out

def ap_table(s: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Rows, bads, bad rate and AP per version and split, from score_all's grouped table."""
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for (v, split), g in s.groupby(['version', SPLIT]):
        n, bad = g['n'].to_numpy(float), g['bad'].to_numpy(float)
        out.setdefault(v, {})[split] = {'rows': int(n.sum()), 'bads': int(bad.sum()),
                                        'bad_rate': float(bad.sum() / n.sum()),
                                        'average_precision': average_precision(g['m'].to_numpy(), bad, n)}
    return out


def write_doc(path: Path, result: dict) -> None:
    m, vs = result['meta'], result['versions']
    base = vs.get('baseline', {}).get('full', {}).get('test', {}).get('average_precision', float('nan'))
    rows = []
    for name, r in vs.items():
        f = r.get('full', {})
        te = f.get('test', {}).get('average_precision', float('nan'))
        tr = f.get('train', {}).get('average_precision', float('nan'))
        rows.append({'version': name, 'what': r['label'], 'trials': r['trials_complete'],
                     'pruned': r['trials_pruned'], 'rounds': r['best_iteration'] + 1,
                     'valid AP (sample, tuning)': r['valid_ap_sample'],
                     'train AP': tr, 'valid AP': f.get('valid', {}).get('average_precision', float('nan')),
                     'test AP': te, 'train - test': tr - te,
                     'test vs baseline': f'{te - base:+.4f}' if base == base and te == te else '--'})
    t = pd.DataFrame(rows).sort_values('test AP', ascending=False)
    params = pd.DataFrame([{'version': n, **{k: v for k, v in r['best_params'].items()}} for n, r in vs.items()])
    rate = next(iter(vs.values())).get('full', {})
    noskill = ', '.join(f"{s} {rate[s]['bad_rate']:.4f}" for s in ('train', 'valid', 'test') if s in rate)
    s = m['sample']['counts']
    pre = (f"XGBoost on the {len(m['features'])} features suggested by `13_gbm_shap_elimination_main`, tuned with "
           f"Optuna (TPE, seed {SEED}) in {len(vs)} versions, **ranked by average precision on every test row** "
           f"(after {m['split']['dev_end']}).\n\n"
           f"- **Tuning data**: every train bad and {m['sample']['goods_fraction']:.0%} of train goods "
           f"({s['train']['rows']:,.0f} rows), goods weighted {1 / m['sample']['goods_fraction']:g} so the rows stand "
           f"for the whole train split; each version's weights multiply these. Objective: weighted AP on the valid "
           f"sample drawn the same way ({s['valid']['rows']:,.0f} rows), unchanged across versions.\n"
           f"- **Per trial**: at most {m['num_rounds']} rounds, early stopping after {m['early_stopping']} rounds "
           f"without a better valid AUC-PR, and median pruning (valid AUC-PR every {PRUNE_EVERY} rounds, after "
           f"{PRUNE_STARTUP} complete trials and {PRUNE_WARMUP} rounds). {m['n_trials']} trials per version "
           f"(complete + pruned).\n"
           f"- **Scored** with the best trial's model on every train ({rate.get('train', {}).get('rows', 0):,}), "
           f"valid ({rate.get('valid', {}).get('rows', 0):,}) and test ({rate.get('test', {}).get('rows', 0):,}) row, "
           f"unweighted; margin rounded to 1e-{MARGIN_DECIMALS}. No-skill AP is the bad rate: {noskill}.\n"
           f"- Over/undersampling lift every train month to the highest monthly train bad rate "
           f"({m['target_rate']:.4%}) as seeded frequency weights; recency weights are 0.5^(months before "
           f"{m['split']['dev_end']} / half-life); SMOTEENN / SMOTETomek come from `14_gbm_resample_main` "
           f"(see `docs/gbm_resample.md`).\n"
           f"- The valid sample both stops each fit early and scores the trials, so valid AP is optimistic; "
           f"test is untouched until this comparison.")
    write_report(path, 'XGBoost: Tuned Versions', 'src/jobs/15_gbm_tuning_main.py',
                 [('Versions, best test AP first', md_table(t, floatfmt='{:.4f}', numparse=False)),
                  ('Best parameters', md_table(params, floatfmt='{:.4g}'))], pre)


def main() -> None:
    """Stages: sample (the tuning sample), tune (each version's study), evaluate (every version scored on every
    row), report. A focal-loss fit starts from the weighted bad rate's log-odds as base margin, since a custom
    objective otherwise starts from margin 0 (p = 0.5)."""
    p = argparse.ArgumentParser(description='gbm_tuning_main')
    p.add_argument('--stage', choices=('all', 'sample', 'tune', 'evaluate', 'report'), default='all')
    p.add_argument('--versions', default=','.join(v.name for v in VERSIONS),
                   help=f"comma-separated subset of: {', '.join(v.name for v in VERSIONS)}")
    p.add_argument('--training-set', default=curated_path('training_set'))
    p.add_argument('--split-json', default=str(settings.model_dir / 'gbm_split.json'))
    p.add_argument('--elimination-json', default=str(settings.model_dir / 'gbm_elimination.json'))
    p.add_argument('--selection-json', default=str(settings.model_dir / 'gbm_corr_selection.json'))
    p.add_argument('--univariate-json', default=str(settings.model_dir / 'gbm_univariate.json'))
    p.add_argument('--elimination-sample', default=curated_path('gbm_elimination_sample'),
                   help='its meta file gives the goods fraction, so the tuning rows are the same draw')
    p.add_argument('--sample-out', default=curated_path('gbm_tuning_sample'), help='on HDFS')
    p.add_argument('--resampled', default=curated_path('gbm_resampled'),
                   help='output of 14_gbm_resample_main, on HDFS')
    p.add_argument('--n-trials', type=int, default=N_TRIALS)
    p.add_argument('--num-rounds', type=int, default=NUM_ROUNDS)
    p.add_argument('--early-stopping', type=int, default=EARLY_STOPPING)
    p.add_argument('--storage', default=f"sqlite:///{settings.model_dir / 'gbm_tuning.db'}",
                   help='Optuna storage; a rerun continues each study where it stopped')
    p.add_argument('--models-dir', default=str(settings.model_dir / 'gbm_tuning_models'))
    p.add_argument('--result-json', default=str(settings.model_dir / 'gbm_tuning.json'))
    p.add_argument('--doc', default=str(Path(__file__).resolve().parents[2] / 'docs' / 'gbm_tuning.md'))
    a = p.parse_args()

    by_name = {v.name: v for v in VERSIONS}
    wanted = [x.strip() for x in a.versions.split(',') if x.strip()]
    unknown = [x for x in wanted if x not in by_name]
    if unknown:
        raise SystemExit(f"Unknown version(s): {', '.join(unknown)}")
    versions = [by_name[x] for x in wanted]
    run = {s: a.stage in ('all', s) for s in ('sample', 'tune', 'evaluate')}

    cfg = load_split(Path(a.split_json))['config']
    end_ym = int(cfg['end_ym'])
    feats = json.loads(Path(a.elimination_json).read_text())['suggested_features']
    kinds = json.loads(Path(a.selection_json).read_text())['kinds']
    cats = json.loads(Path(a.univariate_json).read_text())['categories']
    gf = float(json.loads(sample_meta_path(a.elimination_sample).read_text())['goods_fraction'])
    sample_dir = a.sample_out
    meta_file = sample_meta_path(sample_dir)
    models_dir = Path(a.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    resampled = a.resampled.rstrip('/')

    spark = None
    if run['sample'] or run['evaluate']:
        spark = spark_session('gbm_tuning', conf={'spark.sql.execution.arrow.pyspark.enabled': 'true',
                                                  'spark.driver.maxResultSize': '4g'})
    copies = ExitStack()
    try:
        if run['sample']:
            dev = with_split(spark.read.parquet(resolve_path(a.training_set))
                             .where(ym() <= F.lit(end_ym)), cfg)
            t = time.time()
            smeta = build_sample(dev, feats, kinds, gf, sample_dir)
            meta_file.write_text(json.dumps(smeta, indent=2))
            log.info('Tuning sample in %s (%.0fs)', sample_dir, time.time() - t)

        if not meta_file.exists():
            raise SystemExit(f'No tuning sample at {sample_dir}: run --stage sample first.')
        smeta = json.loads(meta_file.read_text())
        plan = month_plan({int(k): v for k, v in smeta['train_months'].items()})

        if run['tune']:
            local_resampled = None
            if any(v.kind in ('smoteenn', 'smotetomek') for v in versions):
                if not hdfs.exists(f'{resampled}/meta.json'):
                    raise SystemExit(f'No resampled rows at {resampled}: run 14_gbm_resample_main first.')
                local_resampled = copies.enter_context(hdfs.local_copy(resampled))
            with hdfs.local_copy(sample_dir) as local:
                pdf = load_sample(local, feats, kinds)
            train = pdf[pdf[SPLIT] == 'train'].reset_index(drop=True)
            valid = pdf[pdf[SPLIT] == 'valid'].reset_index(drop=True)
            del pdf
            vy, vw = valid[TARGET].to_numpy(np.float32), valid[WEIGHT].to_numpy(np.float32)
            vx = to_frame(valid, feats, kinds, cats)
            for i, v in enumerate(versions):
                t0 = time.time()
                rows, w = version_rows(v, train, plan, end_ym, local_resampled, feats)
                y = rows[TARGET].to_numpy(np.float32)
                margin = {}
                if v.kind == 'focal':
                    r = float(np.sum(w * y) / np.sum(w))
                    margin = {'base_margin': np.full(len(y), np.log(r / (1 - r)), np.float32)}
                dtrain = xgb.QuantileDMatrix(to_frame(rows, feats, kinds, cats), y, weight=w.astype(np.float32),
                                             enable_categorical=True, max_bin=256, **margin)
                vm = {'base_margin': np.full(len(vy), margin['base_margin'][0], np.float32)} if margin else {}
                dvalid = xgb.QuantileDMatrix(vx, vy, weight=vw, enable_categorical=True, ref=dtrain, **vm)
                log.info('[%s] %s: %s train rows, weighted bad rate %.3f%%', v.name, v.label, f'{len(y):,}',
                         100 * float(np.sum(w * y) / np.sum(w)))
                study = optuna.create_study(
                    study_name=f'gbm_{v.name}', storage=a.storage, load_if_exists=True, direction='maximize',
                    sampler=optuna.samplers.TPESampler(seed=SEED + i, multivariate=True),
                    pruner=optuna.pruners.MedianPruner(n_startup_trials=PRUNE_STARTUP, n_warmup_steps=PRUNE_WARMUP,
                                                       interval_steps=PRUNE_EVERY))
                study.set_user_attr('label', v.label)
                tune(v, dtrain, dvalid, vy, vw, study, a.n_trials, models_dir / f'{v.name}.ubj',
                     a.num_rounds, a.early_stopping)
                log.info('[%s] best valid AP %.4f (trial %s); %.0f min', v.name, study.best_value,
                         study.best_trial.number, (time.time() - t0) / 60)
                del dtrain, dvalid, rows

        if run['evaluate']:
            models = {}
            for v in versions:
                mp = models_dir / f'{v.name}.ubj'
                if not mp.exists():
                    raise SystemExit(f'No model for {v.name} at {mp}: run --stage tune first.')
                info = json.loads(mp.with_suffix('.json').read_text())
                models[v.name] = (mp.read_bytes(), int(info['best_iteration']))
            t = time.time()
            df = with_split(spark.read.parquet(resolve_path(a.training_set)), cfg)
            full = ap_table(score_all(df, models, feats, kinds, cats))
            log.info('Scored %s versions on every row in %.0f min', len(models), (time.time() - t) / 60)
            (models_dir / 'full_ap.json').write_text(json.dumps(full, indent=2))
    finally:
        copies.close()
        if spark is not None:
            spark.stop()

    if a.stage in ('all', 'evaluate', 'report'):
        full_path = models_dir / 'full_ap.json'
        full = json.loads(full_path.read_text()) if full_path.exists() else {}
        out_versions = {}
        for v in versions:
            try:
                study = optuna.load_study(study_name=f'gbm_{v.name}', storage=a.storage)
            except KeyError:
                continue
            states = [x.state for x in study.trials]
            info = json.loads((models_dir / f'{v.name}.json').read_text())
            out_versions[v.name] = {
                'label': v.label, 'kind': v.kind, 'half_life_months': v.half_life or None,
                'trials_complete': states.count(optuna.trial.TrialState.COMPLETE),
                'trials_pruned': states.count(optuna.trial.TrialState.PRUNED),
                'best_trial': study.best_trial.number, 'best_params': study.best_trial.params,
                'valid_ap_sample': study.best_value, 'best_iteration': info['best_iteration'],
                'full': full.get(v.name, {})}
            te = full.get(v.name, {}).get('test', {}).get('average_precision')
            if te is not None:
                log.info('[%s] AP train %.4f, valid %.4f, test %.4f', v.name,
                         full[v.name]['train']['average_precision'], full[v.name]['valid']['average_precision'], te)
        result = {'meta': {'split': cfg, 'features': feats, 'sample': smeta | {'train_months': None},
                           'target_rate': plan[0], 'n_trials': a.n_trials, 'num_rounds': a.num_rounds,
                           'early_stopping': a.early_stopping, 'storage': a.storage,
                           'generated': datetime.now().isoformat(timespec='seconds'),
                           'source': 'src/jobs/15_gbm_tuning_main.py'},
                  'versions': out_versions}
        Path(a.result_json).write_text(json.dumps(result, indent=2, default=float))
        write_doc(Path(a.doc), result)
        log.info('Result in %s, report in %s', a.result_json, a.doc)


if __name__ == '__main__':
    main()
