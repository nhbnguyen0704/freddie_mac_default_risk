import argparse
import importlib
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from helpers.config import settings
from helpers.gbm import SPLIT, TARGET, dev_rows, load_split, sample_meta_path
from helpers.hdfs import local_copy
from helpers.report import md_table, write_report
from helpers.rf import PROB_DECIMALS, bad_proba, bad_rate_codes, score_counts, split_ap, to_matrix
from helpers.spark_io import curated_path, spark_session

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s | %(message)s')
log = logging.getLogger('rf_perm_elimination_main')

elbow = importlib.import_module('jobs.13_gbm_shap_elimination_main').elbow

WEIGHT = 'sample_weight'
SHAP_ROW = 'shap_row'

PERM_REPEATS = 3

PARAMS: Dict[str, Any] = {
    'n_estimators': 300, 'criterion': 'gini', 'max_features': 'sqrt', 'min_samples_leaf': 100,
    'max_samples': 1_000_000, 'bootstrap': True, 'random_state': 42,
}
IMPORTANCE_TIE = 1e-9
ELBOW_TOLERANCE = 0.005

os.environ.setdefault('PYSPARK_PYTHON', sys.executable)


def load_sample(path: Path, feats: List[str], kinds: Dict[str, str]):
    """The sample as RF matrices, and the bad-rate codes of its strings (from its train rows, weighted)."""
    t = time.time()
    cat = [c for c in feats if kinds[c] == 'categorical']
    pdf = pq.read_table(path, columns=[SPLIT, TARGET, WEIGHT, SHAP_ROW, *feats],
                        read_dictionary=[SPLIT, *cat]).to_pandas()
    tr = (pdf[SPLIT] == 'train').to_numpy()
    enc = bad_rate_codes(pdf.loc[tr, cat], cat, pdf.loc[tr, TARGET].to_numpy(np.float64),
                         pdf.loc[tr, WEIGHT].to_numpy(np.float64))
    parts = {}
    for s in ('train', 'valid'):
        d = pdf[pdf[SPLIT] == s]
        parts[s] = {'X': to_matrix(d, feats, kinds, enc), 'y': d[TARGET].to_numpy(np.int8),
                    'w': d[WEIGHT].to_numpy(np.float64), 'shap': d[SHAP_ROW].to_numpy(bool)}
    del pdf
    log.info('Sample loaded in %.0fs: train %s rows, valid %s rows', time.time() - t,
             f"{len(parts['train']['y']):,}", f"{len(parts['valid']['y']):,}")
    return parts, enc


def columns(x: np.ndarray, idx: List[int], rows=None) -> np.ndarray:
    """x[rows][:, idx] in column order, copied once."""
    out = np.empty((len(x) if rows is None else int(rows.sum()), len(idx)), dtype=x.dtype, order='F')
    for j, i in enumerate(idx):
        out[:, j] = x[:, i] if rows is None else x[rows, i]
    return out

def fit(x: np.ndarray, part, params: Dict[str, Any]) -> RandomForestClassifier:
    """A random forest on the sample rows with their weights."""
    return RandomForestClassifier(**params, n_jobs=-1).fit(x, part['y'], sample_weight=part['w'])

def sample_ap(rf: RandomForestClassifier, x: np.ndarray, part) -> float:
    """Weighted AP of a forest on a sample split."""
    return float(average_precision_score(part['y'], bad_proba(rf, x), sample_weight=part['w']))

def permutation_importance(rf: RandomForestClassifier, part, idx: List[int], keep: List[str], repeats: int,
                           seed: int) -> pd.DataFrame:
    """Weighted AP on the SHAP rows minus the AP with one column shuffled, per feature."""
    m = part['shap']
    x = columns(part['X'], idx, m)
    y, w = part['y'][m], part['w'][m]
    base = float(average_precision_score(y, bad_proba(rf, x), sample_weight=w))
    rng = np.random.default_rng(seed)
    drops = np.zeros((len(keep), repeats))
    for j in range(len(keep)):
        col = x[:, j].copy()
        for r in range(repeats):
            x[:, j] = col[rng.permutation(len(col))]
            drops[j, r] = base - average_precision_score(y, bad_proba(rf, x), sample_weight=w)
        x[:, j] = col
    return pd.DataFrame({'importance': drops.mean(axis=1), 'sd': drops.std(axis=1, ddof=1) if repeats > 1 else 0.0},
                        index=keep)

def full_ap(df, split: str, model_path: Path, keep: List[str], kinds: Dict[str, str],
            enc: Dict[str, dict]) -> Dict[str, float]:
    """AP on every row of the split, unweighted, P(bad) rounded."""
    return split_ap(score_counts(df.where(df[SPLIT] == split), model_path, keep, kinds, enc))[split]

def nan_if_none(v) -> float:
    return float('nan') if v is None else v


def write_doc(path: Path, result: dict) -> None:
    m, e, steps = result['meta'], result['elbow'], result['steps']
    s = m['sample']['counts']
    preamble = (
        f"Backward elimination with a **random forest** (scikit-learn) on the **raw** features, from the "
        f"{m['candidates']} candidates left by `12_gbm_corr_selection_main` -- the split, candidates and sample "
        f"are the XGBoost ones. Each step fits one forest, scores it, and drops the feature with the lowest "
        f"**permutation importance** (lowest univariate AP on a tie), until {m['min_features']} remain.\n\n"
        f"- **Fit** on the elimination sample of `13_gbm_shap_elimination_main`: every train bad and "
        f"{m['sample']['goods_fraction']:.0%} of train goods (goods weighted {1 / m['sample']['goods_fraction']:g}), "
        f"{s['train']['rows']:,.0f} rows standing for {s['train']['weighted_rows']:,.0f}. Fixed parameters: "
        + ', '.join(f'{k}={v}' for k, v in m['params'].items()) + ". Each tree is grown on a bootstrap of "
        f"max_samples rows; the weights act inside it.\n"
        f"- **Strings** as codes ordered by their weighted train bad rate (missing a category of its own, unseen "
        f"categories as missing), so a split on the code is a split of the categories into low- and high-rate "
        f"groups.\n"
        f"- **Permutation importance**: weighted AP on {s['valid']['shap_rows']:,.0f} rows of the valid sample less "
        f"the AP with the feature's column shuffled, mean of {m['perm_repeats']} shuffles (sd in the progress "
        f"file). TreeSHAP is not used: sklearn's trees are grown to full depth and its cost is quadratic in it.\n"
        f"- **AP full valid**: the forest scored on **every** valid row ({m['full_rows']['valid']:,}) at every "
        f"step, unweighted, P(bad) rounded to 1e-{m['prob_decimals']}. **AP full train** (every train row, "
        f"{m['full_rows']['train'] or 0:,}) only for the suggested step: scoring sklearn forests is slow enough that "
        f"train at every step would quadruple the run. AP sample: on the weighted samples (train AP is in-bag, so "
        f"optimistic).\n\n"
        f"**Elbow**: best full valid AP {e['best_ap_valid']:.4f} with {e['best_n']} features; the fewest features "
        f"within {e['tolerance']:.1%} of it: **{e['suggested_n']}** (valid AP {e['suggested_ap_valid']:.4f}). "
        f"This is a suggestion; the curve is below.")
    curve = pd.DataFrame([{'step': x['step'], 'features': x['n_features'],
                           'AP full train': nan_if_none(x.get('ap_full_train')), 'AP full valid': x['ap_full_valid'],
                           'train - valid': nan_if_none(x.get('ap_full_train')) - x['ap_full_valid'],
                           'AP sample train': x['ap_sample_train'], 'AP sample valid': x['ap_sample_valid'],
                           'dropped next': x.get('dropped') or '--',
                           'its importance': x.get('dropped_importance', float('nan'))} for x in steps])
    chosen = next(x for x in steps if x['n_features'] == e['suggested_n'])
    imp = (pd.DataFrame({'permutation importance': chosen['importance'], 'sd': chosen['importance_sd']})
           .sort_values('permutation importance', ascending=False).rename_axis('feature').reset_index())
    xgb_feats = set(m.get('xgboost_features') or [])
    if xgb_feats:
        imp['in XGBoost set'] = imp['feature'].map(lambda c: 'yes' if c in xgb_feats else 'no')
    sections = [('Elimination curve', md_table(curve, floatfmt='{:.4f}')),
                (f"Suggested features ({e['suggested_n']}), by permutation importance",
                 md_table(imp, floatfmt='{:.5f}'))]
    if xgb_feats:
        mine = set(result['suggested_features'])
        sections.append(('Against the XGBoost selection',
                         f"In both: {len(mine & xgb_feats)}. RF only: "
                         f"{', '.join(f'`{c}`' for c in sorted(mine - xgb_feats)) or '--'}. XGBoost only: "
                         f"{', '.join(f'`{c}`' for c in sorted(xgb_feats - mine)) or '--'}."))
    write_report(path, 'Random Forest Feature Selection: Permutation Backward Elimination',
                 'src/jobs/16_rf_perm_elimination_main.py', sections, preamble)


def _save(path: Path, state: dict) -> None:
    """Checkpoint the elimination state to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state['saved'] = datetime.now().isoformat(timespec='seconds')
    path.write_text(json.dumps(state, indent=2, default=float))


def main() -> None:
    """Backward elimination with a random forest on the XGBoost elimination sample (every train / valid bad and 2%
    of goods, goods weighted 50) from the same 55 candidates, with job 13's elbow rule. Mean |SHAP| is not
    affordable for sklearn's deep trees (TreeSHAP is quadratic in depth), so a feature's importance is its
    permutation importance: the drop in weighted AP when its column is shuffled on the SHAP rows, averaged over
    PERM_REPEATS shuffles; importances closer than IMPORTANCE_TIE tie and the lower univariate AP goes.
    Parameters are fixed and untuned; max_samples bounds each tree's bootstrap so fit time and forest size do
    not grow with the 13M rows. Full-data AP is computed on every valid row at every step and on every train row
    only for the suggested step, as a 300-tree forest scores about 190k rows/s. The bad-rate codes come from the
    sample, so they are the same on a resume."""
    p = argparse.ArgumentParser(description='rf_perm_elimination_main')
    p.add_argument('--stage', choices=('all', 'eliminate', 'report'), default='all',
                   help='eliminate: the backward elimination; report: rewrite the report from --progress-json')
    p.add_argument('--training-set', default=curated_path('training_set'))
    p.add_argument('--split-json', default=str(settings.model_dir / 'gbm_split.json'))
    p.add_argument('--selection-json', default=str(settings.model_dir / 'gbm_corr_selection.json'),
                   help='candidates from 12_gbm_corr_selection_main')
    p.add_argument('--univariate-json', default=str(settings.model_dir / 'gbm_univariate.json'),
                   help='univariate AP (the tie breaker)')
    p.add_argument('--xgboost-json', default=str(settings.model_dir / 'gbm_elimination.json'),
                   help="XGBoost's selection, for the comparison in the report")
    p.add_argument('--sample', default=curated_path('gbm_elimination_sample'),
                   help='the XGBoost elimination sample (13_gbm_shap_elimination_main --stage sample)')
    p.add_argument('--n-estimators', type=int, default=PARAMS['n_estimators'])
    p.add_argument('--min-samples-leaf', type=int, default=PARAMS['min_samples_leaf'])
    p.add_argument('--max-samples', type=int, default=PARAMS['max_samples'])
    p.add_argument('--perm-repeats', type=int, default=PERM_REPEATS)
    p.add_argument('--min-features', type=int, default=1, help='stop when this many features are left')
    p.add_argument('--elbow-tolerance', type=float, default=ELBOW_TOLERANCE)
    p.add_argument('--progress-json', default=str(settings.model_dir / 'rf_elimination_progress.json'))
    p.add_argument('--resume', action='store_true', help='continue from --progress-json')
    p.add_argument('--models-dir', default=str(settings.model_dir / 'rf_elimination_models'),
                   help="every step's forest, step_NN.joblib")
    p.add_argument('--result-json', default=str(settings.model_dir / 'rf_elimination.json'))
    p.add_argument('--doc', default=str(Path(__file__).resolve().parents[2] / 'docs' / 'rf_elimination.md'))
    a = p.parse_args()

    cfg = load_split(Path(a.split_json))['config']
    sel = json.loads(Path(a.selection_json).read_text())
    uj = json.loads(Path(a.univariate_json).read_text())
    kinds, candidates = sel['kinds'], sel['candidates']
    uni_ap = pd.Series({r['feature']: r['univariate_ap_valid'] for r in uj['features']})
    params = {**PARAMS, 'n_estimators': a.n_estimators, 'min_samples_leaf': a.min_samples_leaf,
              'max_samples': a.max_samples}
    progress = Path(a.progress_json)
    meta_file = sample_meta_path(a.sample)
    if not meta_file.exists():
        raise SystemExit(f'No sample meta at {meta_file}: run 13_gbm_shap_elimination_main --stage sample first.')
    smeta = json.loads(meta_file.read_text())
    if not set(candidates) <= set(smeta['features']):
        raise SystemExit(f'{a.sample} lacks candidates: rerun 13_gbm_shap_elimination_main --stage sample.')

    state: Dict[str, Any] = {}
    if a.resume or a.stage == 'report':
        if not progress.exists():
            raise SystemExit(f'Nothing to resume: no {progress}.')
        state = json.loads(progress.read_text())
        if not set(state['keep']) <= set(candidates):
            raise SystemExit(f'{progress} is from another candidate set: rerun without --resume.')
        log.info('Resuming from %s: %s steps done, %s features left', progress, len(state['steps']),
                 len(state['keep']))

    if a.stage != 'report':
        with local_copy(a.sample) as local:
            parts, enc = load_sample(local, candidates, kinds)
        if not state:
            state = {'keep': list(candidates), 'steps': [], 'done': False, 'encoding': enc}
        enc = state['encoding']
        col = {c: i for i, c in enumerate(candidates)}
        models = Path(a.models_dir)
        models.mkdir(parents=True, exist_ok=True)
        spark = spark_session('rf_perm_elimination',
                              conf={'spark.sql.execution.arrow.pyspark.enabled': 'true',
                                    'spark.sql.execution.arrow.maxRecordsPerBatch': '50000',
                                    'spark.driver.maxResultSize': '4g'})
        try:
            dev = dev_rows(spark, a.training_set, cfg)
            while not state['done']:
                keep = state['keep']
                idx = [col[c] for c in keep]
                step = len(state['steps']) + 1
                t0 = time.time()
                xt = columns(parts['train']['X'], idx)
                rf = fit(xt, parts['train'], params)
                t_fit = time.time() - t0
                model_path = models / f'step_{step:02d}.joblib'
                joblib.dump(rf, model_path, compress=3)
                ap_tr = sample_ap(rf, xt, parts['train'])
                del xt
                ap_va = sample_ap(rf, columns(parts['valid']['X'], idx), parts['valid'])
                t = time.time()
                imp = permutation_importance(rf, parts['valid'], idx, keep, a.perm_repeats, seed=step)
                t_perm = time.time() - t
                t = time.time()
                full = {'valid': full_ap(dev, 'valid', model_path, keep, kinds, enc)}
                t_full = time.time() - t
                rec = {'step': step, 'n_features': len(keep), 'features': list(keep),
                       'ap_sample_train': ap_tr, 'ap_sample_valid': ap_va, 'ap_full_train': None,
                       'ap_full_valid': full['valid']['average_precision'],
                       'full': full, 'importance': imp['importance'].to_dict(), 'importance_sd': imp['sd'].to_dict(),
                       'mean_nodes_per_tree': float(np.mean([e.tree_.node_count for e in rf.estimators_])),
                       'seconds_fit': round(t_fit), 'seconds_perm': round(t_perm), 'seconds_full_ap': round(t_full)}
                del rf
                if len(keep) > max(a.min_features, 1):
                    s = imp['importance']
                    low = s[s <= s.min() + IMPORTANCE_TIE].index
                    worst = uni_ap[low].sort_values(kind='stable').index[0] if len(low) > 1 else low[0]
                    rec.update({'dropped': worst, 'dropped_importance': float(s[worst]),
                                'tied_at_min': int(len(low))})
                    keep.remove(worst)
                else:
                    state['done'] = True
                state['steps'].append(rec)
                _save(progress, state)
                log.info('Step %s: %s features, AP full valid %.4f, sample %.4f / %.4f; drop %s '
                         '(importance %.5f%s); fit %.0fs, permutation %.0fs, full AP %.0fs', step, rec['n_features'],
                         rec['ap_full_valid'], ap_tr, ap_va, rec.get('dropped', '--'),
                         rec.get('dropped_importance', float('nan')),
                         f", tie of {rec['tied_at_min']} broken by univariate AP"
                         if rec.get('tied_at_min', 1) > 1 else '', t_fit, t_perm, t_full)
            chosen = next(x for x in state['steps']
                          if x['n_features'] == elbow(state['steps'], a.elbow_tolerance)['suggested_n'])
            if chosen.get('ap_full_train') is None:
                t = time.time()
                chosen['full']['train'] = full_ap(dev, 'train', models / f"step_{chosen['step']:02d}.joblib",
                                                  chosen['features'], kinds, enc)
                chosen['ap_full_train'] = chosen['full']['train']['average_precision']
                _save(progress, state)
                log.info('Suggested step %s (%s features): AP full train %.4f, valid %.4f (%.0fs)', chosen['step'],
                         chosen['n_features'], chosen['ap_full_train'], chosen['ap_full_valid'], time.time() - t)
        finally:
            spark.stop()

    if state.get('steps'):
        steps = state['steps']
        e = elbow(steps, a.elbow_tolerance)
        chosen = next(x for x in steps if x['n_features'] == e['suggested_n'])
        xgb_path = Path(a.xgboost_json)
        xgb_feats = json.loads(xgb_path.read_text())['suggested_features'] if xgb_path.exists() else None
        result = {
            'meta': {'split': cfg, 'candidates': len(candidates), 'min_features': a.min_features,
                     'params': params, 'perm_repeats': a.perm_repeats, 'prob_decimals': PROB_DECIMALS, 'sample': smeta,
                     'full_rows': {s: next((x['full'][s]['rows'] for x in steps if s in x['full']), None)
                                   for s in ('train', 'valid')},
                     'rule': 'drop the lowest permutation importance (weighted AP drop on the valid SHAP rows), '
                             'lowest univariate AP on a tie',
                     'metric': f'average precision on every valid row (every train row for the suggested step), '
                               f'P(bad) rounded to 1e-{PROB_DECIMALS}',
                     'xgboost_features': xgb_feats, 'complete': bool(state['done']),
                     'generated': datetime.now().isoformat(timespec='seconds'),
                     'source': 'src/jobs/16_rf_perm_elimination_main.py'},
            'elbow': e, 'suggested_features': chosen['features'], 'encoding': state['encoding'],
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
