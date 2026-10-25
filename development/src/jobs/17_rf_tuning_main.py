import argparse
import importlib
import json
import logging
import os
import sys
import time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import optuna
import pandas as pd
from pyspark.sql import functions as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from helpers.config import settings
from helpers import hdfs
from helpers.gbm import SPLIT, TARGET, load_split, sample_meta_path, with_split, ym
from helpers.report import md_table, write_report
from helpers.rf import PROB_DECIMALS, score_counts, split_ap, to_matrix, tree_sum
from helpers.spark_io import curated_path, resolve_path, spark_session

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s | %(message)s')
log = logging.getLogger('rf_tuning_main')
optuna.logging.set_verbosity(optuna.logging.WARNING)

g15 = importlib.import_module('jobs.15_gbm_tuning_main')
Version = g15.Version

WEIGHT = 'sample_weight'

N_TRIALS = 30
SEED = 42
TREE_CHUNK = 50
PRUNE_STARTUP = 5
PRUNE_WARMUP = TREE_CHUNK

VERSIONS = [Version('baseline', 'none'), Version('class_weight', 'spw'),
            Version('recency_6', 'recency', 6), Version('recency_12', 'recency', 12),
            Version('recency_25', 'recency', 25), Version('oversample', 'oversample'),
            Version('undersample', 'undersample'), Version('smoteenn', 'smoteenn'),
            Version('smotetomek', 'smotetomek')]
XGB_TWIN = {'class_weight': 'scale_pos_weight'}

os.environ.setdefault('PYSPARK_PYTHON', sys.executable)


def label(v: Version) -> str:
    """The version's description in the report; class_weight tunes a weight on the bads."""
    return {'none': 'no reweighting', 'spw': 'weight on bads tuned'}.get(v.kind, v.label)


def suggest(trial: optuna.Trial, v: Version) -> Dict[str, Any]:
    """One trial's forest parameters, plus the bad weight for the class_weight version."""
    p = {'criterion': 'gini', 'bootstrap': True, 'random_state': SEED,
         'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=TREE_CHUNK),
         'max_features': trial.suggest_float('max_features', 0.05, 0.6),
         'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 5000, log=True),
         'max_samples': trial.suggest_int('max_samples', 200_000, 2_000_000, log=True),
         'max_depth': trial.suggest_int('max_depth', 8, 48)}
    if v.kind == 'spw':
        p['bad_weight'] = trial.suggest_float('bad_weight', 1.0, 150.0, log=True)
    return p

def tune(v: Version, x: np.ndarray, y: np.ndarray, w: np.ndarray, vx: np.ndarray, vy: np.ndarray,
         vw: np.ndarray, study: optuna.Study, n_trials: int, model_path: Path) -> None:
    """Run the version's Optuna study up to n_trials finished trials, saving the best forest and its parameters to
    model_path. Tuned on the XGBoost tuning design (every train / valid bad and 2% of goods, goods weighted 50)
    with job 15's versions, rows, weights and month plan; focal loss has no forest counterpart and
    scale_pos_weight becomes a tuned weight on the bads. The forest grows TREE_CHUNK trees at a time (warm
    start) and reports the valid AP after each chunk for median pruning, after PRUNE_STARTUP complete trials and
    from the second chunk on."""
    def objective(trial: optuna.Trial) -> float:
        t = time.time()
        p = suggest(trial, v)
        n_trees = p.pop('n_estimators')
        ww = w * np.where(y == 1, p.pop('bad_weight'), 1.0) if 'bad_weight' in p else w
        rf = RandomForestClassifier(**p, n_estimators=TREE_CHUNK, warm_start=True, n_jobs=-1)
        s = np.zeros(len(vy))
        for k in range(TREE_CHUNK, n_trees + 1, TREE_CHUNK):
            rf.set_params(n_estimators=k)
            rf.fit(x, y, sample_weight=ww)
            s += tree_sum(rf.estimators_[k - TREE_CHUNK:], vx)
            ap = float(average_precision_score(vy, s / k, sample_weight=vw))
            trial.report(ap, k)
            if k < n_trees and trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr('seconds', round(time.time() - t))
        trial.set_user_attr('mean_nodes_per_tree', float(np.mean([e.tree_.node_count for e in rf.estimators_])))
        done = [z for z in study.trials if z.state == optuna.trial.TrialState.COMPLETE]
        if not done or ap > max(z.value for z in done):
            rf.set_params(warm_start=False)
            joblib.dump(rf, model_path, compress=3)
            json.dump({'params': {**trial.params}, 'trial': trial.number, 'valid_ap_sample': ap},
                      model_path.with_suffix('.json').open('w'), indent=2, default=float)
        log.info('[%s] trial %s: valid AP %.4f, %s trees, %.0fs', v.name, trial.number, ap, n_trees,
                 time.time() - t)
        return ap

    def pruned(study, trial):
        if trial.state == optuna.trial.TrialState.PRUNED:
            log.info('[%s] trial %s pruned at %s trees', v.name, trial.number, trial.last_step)

    finished = sum(z.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
                   for z in study.trials)
    if finished < n_trials:
        study.optimize(objective, n_trials=n_trials - finished, callbacks=[pruned], gc_after_trial=True)


def resampled_features(uri: str) -> Optional[List[str]]:
    """The features a 14_gbm_resample_main output was built on (None if there is none)."""
    if not hdfs.exists(f'{uri}/meta.json'):
        return None
    return hdfs.read_jsons(f'{uri}/meta.json')[0]['config']['features']


def write_doc(path: Path, result: dict) -> None:
    m, vs, xg = result['meta'], result['versions'], result['xgboost']
    base = vs.get('baseline', {}).get('full', {}).get('test', {}).get('average_precision', float('nan'))
    rows = []
    for name, r in vs.items():
        f = r.get('full', {})
        te = f.get('test', {}).get('average_precision', float('nan'))
        tr = f.get('train', {}).get('average_precision', float('nan'))
        xte = xg.get(XGB_TWIN.get(name, name), float('nan'))
        rows.append({'version': name, 'what': r['label'], 'trials': r['trials_complete'],
                     'pruned': r['trials_pruned'], 'trees': r['best_params'].get('n_estimators'),
                     'valid AP (sample, tuning)': r['valid_ap_sample'],
                     'train AP': tr, 'valid AP': f.get('valid', {}).get('average_precision', float('nan')),
                     'test AP': te, 'train - test': tr - te,
                     'test vs baseline': f'{te - base:+.4f}' if base == base and te == te else '--',
                     'XGBoost test AP': xte,
                     'RF - XGBoost': f'{te - xte:+.4f}' if te == te and xte == xte else '--'})
    t = pd.DataFrame(rows).sort_values('test AP', ascending=False)
    params = pd.DataFrame([{'version': n, **r['best_params']} for n, r in vs.items()])
    rate = next(iter(vs.values())).get('full', {})
    noskill = ', '.join(f"{s} {rate[s]['bad_rate']:.4f}" for s in ('train', 'valid', 'test') if s in rate)
    s = m['sample']['counts']
    pre = (f"Random forest (scikit-learn) on the {len(m['features'])} features suggested by "
           f"`16_rf_perm_elimination_main`, tuned with Optuna (TPE, seed {SEED}) in {len(vs)} versions, "
           f"**ranked by average precision on every test row** (after {m['split']['dev_end']}). The tuning design, "
           f"the versions' weights and the SMOTE rows are those of `15_gbm_tuning_main` / `14_gbm_resample_main`; "
           f"focal loss has no forest counterpart and is left out, and scale_pos_weight becomes a tuned weight on "
           f"the bads (`class_weight`, compared with XGBoost's `scale_pos_weight`).\n\n"
           f"- **Tuning data**: every train bad and {m['sample']['goods_fraction']:.0%} of train goods "
           f"({s['train']['rows']:,.0f} rows, `{m['sample']['path'].rsplit('/', 1)[-1]}`), goods weighted "
           f"{1 / m['sample']['goods_fraction']:g}; each version's weights multiply these. Objective: weighted AP "
           f"on the valid sample drawn the same way ({s['valid']['rows']:,.0f} rows).\n"
           f"- **Per trial**: the forest is grown {TREE_CHUNK} trees at a time up to its n_estimators; the valid AP "
           f"after each chunk feeds median pruning (after {PRUNE_STARTUP} complete trials). Searched: n_estimators "
           f"100-500, max_features 0.05-0.6, min_samples_leaf 5-5000, max_samples (bootstrap rows per tree) "
           f"200k-2M, max_depth 8-48, and the bad weight 1-150 for `class_weight`. {m['n_trials']} trials per "
           f"version (complete + pruned).\n"
           f"- **Strings** as codes ordered by weighted train bad rate (from `16_rf_perm_elimination_main`).\n"
           f"- **Scored** with the best trial's forest on every train ({rate.get('train', {}).get('rows', 0):,}), "
           f"valid ({rate.get('valid', {}).get('rows', 0):,}) and test ({rate.get('test', {}).get('rows', 0):,}) row, "
           f"unweighted; P(bad) rounded to 1e-{PROB_DECIMALS}. No-skill AP is the bad rate: {noskill}.\n"
           f"- The valid sample scores the trials, so valid AP is optimistic; test is untouched until this "
           f"comparison. XGBoost test AP from `artifacts/gbm_tuning.json`.")
    write_report(path, 'Random Forest: Tuned Versions', 'src/jobs/17_rf_tuning_main.py',
                 [('Versions, best test AP first', md_table(t, floatfmt='{:.4f}', numparse=False)),
                  ('Best parameters', md_table(params, floatfmt='{:.4g}'))], pre)


def main() -> None:
    """Stages: sample, tune, evaluate, report. Tuning reuses the XGBoost tuning sample when it has every RF
    feature, else draws an RF one the same way; each version is compared in the report with its XGBoost twin."""
    p = argparse.ArgumentParser(description='rf_tuning_main')
    p.add_argument('--stage', choices=('all', 'sample', 'tune', 'evaluate', 'report'), default='all')
    p.add_argument('--versions', default=','.join(v.name for v in VERSIONS),
                   help=f"comma-separated subset of: {', '.join(v.name for v in VERSIONS)}")
    p.add_argument('--training-set', default=curated_path('training_set'))
    p.add_argument('--split-json', default=str(settings.model_dir / 'gbm_split.json'))
    p.add_argument('--elimination-json', default=str(settings.model_dir / 'rf_elimination.json'),
                   help='features and string codes from 16_rf_perm_elimination_main')
    p.add_argument('--selection-json', default=str(settings.model_dir / 'gbm_corr_selection.json'))
    p.add_argument('--elimination-sample', default=curated_path('gbm_elimination_sample'),
                   help='its meta file gives the goods fraction')
    p.add_argument('--gbm-sample', default=curated_path('gbm_tuning_sample'),
                   help="15_gbm_tuning_main's sample: reused when it holds every RF feature")
    p.add_argument('--sample-out', default=curated_path('rf_tuning_sample'),
                   help='on HDFS; built (same draw as the XGBoost one) only when --gbm-sample lacks RF features')
    p.add_argument('--resampled', default='',
                   help='14_gbm_resample_main output for the RF features; default: gbm_resampled if it was built '
                        'on them, else rf_resampled')
    p.add_argument('--n-trials', type=int, default=N_TRIALS)
    p.add_argument('--storage', default=f"sqlite:///{settings.model_dir / 'rf_tuning.db'}",
                   help='Optuna storage; a rerun continues each study where it stopped')
    p.add_argument('--models-dir', default=str(settings.model_dir / 'rf_tuning_models'))
    p.add_argument('--rescore', action='store_true', help='evaluate versions already in full_ap.json again')
    p.add_argument('--xgboost-json', default=str(settings.model_dir / 'gbm_tuning.json'))
    p.add_argument('--result-json', default=str(settings.model_dir / 'rf_tuning.json'))
    p.add_argument('--doc', default=str(Path(__file__).resolve().parents[2] / 'docs' / 'rf_tuning.md'))
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
    elim = json.loads(Path(a.elimination_json).read_text())
    feats, enc = elim['suggested_features'], elim['encoding']
    kinds = json.loads(Path(a.selection_json).read_text())['kinds']
    gf = float(json.loads(sample_meta_path(a.elimination_sample).read_text())['goods_fraction'])
    models_dir = Path(a.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    gbm_meta = sample_meta_path(a.gbm_sample)
    reuse = gbm_meta.exists() and set(feats) <= set(json.loads(gbm_meta.read_text())['features'])
    sample_dir = a.gbm_sample if reuse else a.sample_out
    meta_file = sample_meta_path(sample_dir)
    log.info('%s features; tuning sample %s (%s)', len(feats), sample_dir,
             'the XGBoost one, reused' if reuse else 'RF-specific')

    spark = None
    if (run['sample'] and not reuse) or run['evaluate']:
        spark = spark_session('rf_tuning', conf={'spark.sql.execution.arrow.pyspark.enabled': 'true',
                                                 'spark.sql.execution.arrow.maxRecordsPerBatch': '50000',
                                                 'spark.driver.maxResultSize': '4g'})
    copies = ExitStack()
    try:
        if run['sample'] and not reuse:
            dev = with_split(spark.read.parquet(resolve_path(a.training_set)).where(ym() <= F.lit(end_ym)), cfg)
            t = time.time()
            smeta = g15.build_sample(dev, feats, kinds, gf, sample_dir)
            meta_file.write_text(json.dumps(smeta, indent=2))
            log.info('Tuning sample in %s (%.0fs)', sample_dir, time.time() - t)

        if not meta_file.exists():
            raise SystemExit(f'No tuning sample at {sample_dir}: run --stage sample first.')
        smeta = json.loads(meta_file.read_text())
        plan = g15.month_plan({int(k): v for k, v in smeta['train_months'].items()})

        if run['tune']:
            local_resampled = None
            if any(v.kind in ('smoteenn', 'smotetomek') for v in versions):
                resampled = a.resampled.rstrip('/')
                if not resampled:
                    gbm_res = curated_path('gbm_resampled')
                    same = set(resampled_features(gbm_res) or []) == set(feats)
                    resampled = gbm_res if same else curated_path('rf_resampled')
                built_on = resampled_features(resampled)
                if built_on is None or set(built_on) != set(feats):
                    raise SystemExit(
                        f'No SMOTE rows for the RF features at {resampled}. Run, from src:\n'
                        f'  python -m jobs.14_gbm_resample_main --elimination-json {a.elimination_json} '
                        f"--out {curated_path('rf_resampled')} --doc "
                        f"{Path(__file__).resolve().parents[2] / 'docs' / 'rf_resample.md'}")
                log.info('SMOTE rows from %s', resampled)
                local_resampled = copies.enter_context(hdfs.local_copy(resampled))
            with hdfs.local_copy(sample_dir) as local:
                pdf = g15.load_sample(local, feats, kinds)
            train = pdf[pdf[SPLIT] == 'train'].reset_index(drop=True)
            valid = pdf[pdf[SPLIT] == 'valid'].reset_index(drop=True)
            del pdf
            vy, vw = valid[TARGET].to_numpy(np.int8), valid[WEIGHT].to_numpy(np.float64)
            vx = to_matrix(valid, feats, kinds, enc)
            del valid
            for i, v in enumerate(versions):
                t0 = time.time()
                rows, w = g15.version_rows(v, train, plan, end_ym, local_resampled, feats)
                y = rows[TARGET].to_numpy(np.int8)
                x = to_matrix(rows, feats, kinds, enc)
                del rows
                log.info('[%s] %s: %s train rows, weighted bad rate %.3f%%', v.name, label(v), f'{len(y):,}',
                         100 * float(np.sum(w * y) / np.sum(w)))
                study = optuna.create_study(
                    study_name=f'rf_{v.name}', storage=a.storage, load_if_exists=True, direction='maximize',
                    sampler=optuna.samplers.TPESampler(seed=SEED + i, multivariate=True),
                    pruner=optuna.pruners.MedianPruner(n_startup_trials=PRUNE_STARTUP, n_warmup_steps=PRUNE_WARMUP,
                                                       interval_steps=1))
                study.set_user_attr('label', label(v))
                tune(v, x, y, w, vx, vy, vw, study, a.n_trials, models_dir / f'{v.name}.joblib')
                log.info('[%s] best valid AP %.4f (trial %s); %.0f min', v.name, study.best_value,
                         study.best_trial.number, (time.time() - t0) / 60)
                del x, y, w

        if run['evaluate']:
            full_path = models_dir / 'full_ap.json'
            full = json.loads(full_path.read_text()) if full_path.exists() else {}
            df = with_split(spark.read.parquet(resolve_path(a.training_set)), cfg)
            for v in versions:
                mp = models_dir / f'{v.name}.joblib'
                if not mp.exists():
                    raise SystemExit(f'No model for {v.name} at {mp}: run --stage tune first.')
                if v.name in full and not a.rescore:
                    log.info('[%s] already scored; --rescore to score again', v.name)
                    continue
                t = time.time()
                full[v.name] = split_ap(score_counts(df, mp, feats, kinds, enc))
                full_path.write_text(json.dumps(full, indent=2))
                log.info('[%s] scored on every row in %.0f min: AP test %.4f', v.name, (time.time() - t) / 60,
                         full[v.name]['test']['average_precision'])
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
                study = optuna.load_study(study_name=f'rf_{v.name}', storage=a.storage)
            except KeyError:
                continue
            states = [z.state for z in study.trials]
            out_versions[v.name] = {
                'label': label(v), 'kind': v.kind, 'half_life_months': v.half_life or None,
                'trials_complete': states.count(optuna.trial.TrialState.COMPLETE),
                'trials_pruned': states.count(optuna.trial.TrialState.PRUNED),
                'best_trial': study.best_trial.number, 'best_params': study.best_trial.params,
                'valid_ap_sample': study.best_value, 'seconds': study.best_trial.user_attrs.get('seconds'),
                'full': full.get(v.name, {})}
        xp = Path(a.xgboost_json)
        xgb = ({n: r.get('full', {}).get('test', {}).get('average_precision', float('nan'))
                for n, r in json.loads(xp.read_text())['versions'].items()} if xp.exists() else {})
        result = {'meta': {'split': cfg, 'features': feats, 'sample': smeta | {'train_months': None},
                           'target_rate': plan[0], 'n_trials': a.n_trials, 'tree_chunk': TREE_CHUNK,
                           'storage': a.storage, 'generated': datetime.now().isoformat(timespec='seconds'),
                           'source': 'src/jobs/17_rf_tuning_main.py'},
                  'versions': out_versions, 'xgboost': xgb}
        Path(a.result_json).write_text(json.dumps(result, indent=2, default=float))
        if out_versions:
            write_doc(Path(a.doc), result)
        log.info('Result in %s, report in %s', a.result_json, a.doc)


if __name__ == '__main__':
    main()
