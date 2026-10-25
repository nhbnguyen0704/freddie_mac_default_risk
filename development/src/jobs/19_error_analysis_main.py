import argparse
import importlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text
from helpers.config import settings
from helpers.gbm import TARGET, load_split
from helpers.report import md_table, write_report
from helpers.spark_io import resolve_path, spark_session

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s | %(message)s')
log = logging.getLogger('error_analysis_main')

g18 = importlib.import_module('jobs.18_model_correlation_main')
MODELS, BEST, NAMES = g18.MODELS, g18.BEST, g18.NAMES

MAX_DEPTH = 3
MIN_LEAF_SHARE = 0.01
MIN_CAT_BADS = 500
OTHER, MISSING = '__other__', '__missing__'
SHOW_CATS = 6


def f1_threshold(g: pd.DataFrame, col: str) -> Dict[str, float]:
    """The score maximizing F1 when every row scoring at or above it is flagged,
    from (score -> rows, bads)."""
    g = g.sort_values(col, ascending=False)
    tp, flagged = np.cumsum(g['bad'].to_numpy(float)), np.cumsum(g['n'].to_numpy(float))
    precision, recall = tp / flagged, tp / tp[-1]
    f1 = np.where(tp > 0, 2 * precision * recall / (precision + recall), 0.0)
    k = int(np.argmax(f1))
    return {'threshold': float(g[col].iloc[k]), 'f1': float(f1[k]), 'precision': float(precision[k]),
            'recall': float(recall[k]), 'flagged': int(flagged[k]), 'flagged_share': float(flagged[k] / flagged[-1]),
            'true_positives': int(tp[k]), 'missed_bads': int(tp[-1] - tp[k])}

def encode(bads: pd.DataFrame, feats: List[str], kinds: Dict[str, str], miss: np.ndarray
           ) -> Tuple[np.ndarray, Dict[str, List[str]]]:
    """Tree input: numerics as they are (NaN routed by the tree), categoricals as
    codes by miss rate; returns the matrix and, per categorical, its categories
    in code order."""
    x = np.empty((len(bads), len(feats)), dtype=np.float32)
    order = {}
    for j, c in enumerate(feats):
        if kinds[c] != 'categorical':
            x[:, j] = bads[c].to_numpy(dtype=np.float32, na_value=np.nan)
            continue
        s = bads[c].astype(object).where(bads[c].notna(), MISSING)
        counts = s.value_counts()
        s = s.where(s.map(counts).to_numpy() >= MIN_CAT_BADS, OTHER)
        rate = pd.Series(miss, index=s.index).groupby(s).mean()
        cats = sorted(rate.index, key=lambda k: (rate[k], k))
        order[c] = cats
        x[:, j] = s.map({k: i for i, k in enumerate(cats)}).to_numpy(np.float32)
    return x, order

def cat_text(cats: List[str]) -> str:
    """A category list for a rule, shortened to SHOW_CATS names."""
    shown = ', '.join(cats[:SHOW_CATS])
    return f"{{{shown}{f', +{len(cats) - SHOW_CATS} more' if len(cats) > SHOW_CATS else ''}}}"

def leaves(tree: DecisionTreeClassifier, feats: List[str], kinds: Dict[str, str], order: Dict[str, List[str]],
           x: np.ndarray, miss: np.ndarray) -> List[dict]:
    """Each leaf: its rule (bounds merged per feature), bads, misses."""
    t = tree.tree_
    leaf_of = tree.apply(x)
    total_miss = miss.sum()
    out = []

    def walk(node: int, cond: Dict[str, dict]) -> None:
        if t.children_left[node] == -1:
            idx = leaf_of == node
            n, m = int(idx.sum()), int(miss[idx].sum())
            rules = []
            for c, b in cond.items():
                if kinds[c] == 'categorical':
                    rules.append(f"{c} in {cat_text(b['cats'])}")
                else:
                    lo, hi = b['lo'], b['hi']
                    r = (f'{lo:.4g} < {c} <= {hi:.4g}' if lo > -np.inf and hi < np.inf
                         else f'{c} <= {hi:.4g}' if hi < np.inf else f'{c} > {lo:.4g}')
                    rules.append(r + (' or missing' if b['missing'] else ''))
            out.append({'rule': ' AND '.join(rules) or '(all bads)', 'bads': n, 'missed': m,
                        'miss_rate': m / n if n else float('nan'), 'share_of_misses': m / total_miss,
                        'conditions': {c: ({'categories': b['cats']} if kinds[c] == 'categorical'
                                           else {'lo': b['lo'], 'hi': b['hi'], 'missing': b['missing']})
                                       for c, b in cond.items()}})
            return
        c, thr = feats[t.feature[node]], float(t.threshold[node])
        miss_left = bool(t.missing_go_to_left[node])
        for child, left in ((t.children_left[node], True), (t.children_right[node], False)):
            nc = {k: dict(v) for k, v in cond.items()}
            if kinds[c] == 'categorical':
                cats = nc.get(c, {'cats': order[c]})['cats']
                nc[c] = {'cats': [k for k in cats if (order[c].index(k) <= thr) == left]}
            else:
                b = nc.get(c, {'lo': -np.inf, 'hi': np.inf, 'missing': True})
                if left:
                    b['hi'] = min(b['hi'], thr)
                else:
                    b['lo'] = max(b['lo'], thr)
                b['missing'] = b['missing'] and (miss_left == left)
                nc[c] = b
            walk(child, nc)

    walk(0, {})
    return sorted(out, key=lambda r: -r['miss_rate'])

def write_doc(path: Path, r: dict, cfg: dict) -> None:
    th = pd.DataFrame([{'model': f"{NAMES[c]} ({BEST[c]})", 'threshold': r['thresholds'][c]['threshold'],
                        'flagged': r['thresholds'][c]['flagged'], 'flagged share': r['thresholds'][c]['flagged_share'],
                        'precision': r['thresholds'][c]['precision'], 'recall': r['thresholds'][c]['recall'],
                        'F1': r['thresholds'][c]['f1'], 'missed bads': r['thresholds'][c]['missed_bads'],
                        'tree AUC': r['trees'][c]['auc']} for c in MODELS])
    ov = pd.DataFrame(r['overlap'])
    n = r['bads']
    pre = (f"Which test bads (after {cfg['dev_end']}; all {n:,} of them) each model's best version misses. A bad is "
           f"**missed** (false negative) when it scores below the model's threshold, the one that maximizes F1 over "
           f"every test row (rows scoring at or above it are flagged). Thresholds are on each model's own scale: "
           f"LR logit, XGBoost margin, RF P(bad).\n\n"
           f"For each model a decision tree (depth {MAX_DEPTH}, each leaf at least {MIN_LEAF_SHARE:.0%} of the bads) "
           f"is fitted on every test bad to predict missed vs caught, on the {len(r['features'])} raw columns any of "
           f"the three models reads, the same for all three. Categoricals are coded by the model's miss rate "
           f"(categories with under {MIN_CAT_BADS} bads pooled as `{OTHER}`, missing as `{MISSING}`), so a split "
           f"is the best two-group partition of the categories; numerics keep missing values, routed by the tree. "
           f"Tree AUC is in-sample over {n:,} rows with {2 ** MAX_DEPTH} leaves at most, so it is close to "
           f"out-of-sample. Leaves are sorted by miss rate; **lift** is the leaf's miss rate over the model's "
           f"overall miss rate (1 - recall).")
    sections = [('Thresholds (max F1 on every test row)', md_table(th, floatfmt='{:.4f}')),
                ('Missed by which models (test bads)', md_table(ov, floatfmt='{:.4f}'))]
    for c in MODELS:
        lv = pd.DataFrame(r['trees'][c]['leaves'])
        base = 1 - r['thresholds'][c]['recall']
        lv['lift'] = lv['miss_rate'] / base
        lv = lv[['rule', 'bads', 'missed', 'miss_rate', 'lift', 'share_of_misses']].rename(
            columns={'miss_rate': 'miss rate', 'share_of_misses': 'share of misses'})
        sections.append((f"{NAMES[c]} ({BEST[c]}): missed bads, overall miss rate {base:.4f}",
                         md_table(lv, floatfmt='{:.4f}', numparse=False) + '\n\n```\n' + r['trees'][c]['text'] + '```'))
    write_report(path, 'Error Analysis: Missed Bads on Test', 'src/jobs/19_error_analysis_main.py', sections, pre)


def main() -> None:
    """Error analysis of the three best models of 18_model_correlation_main. An error is a missed bad: a test bad
    scored below the model's F1-maximising threshold over every test row (flagged when the score is at or above
    it). One shallow tree per model, fitted on every test bad with leaves of at least MIN_LEAF_SHARE of them,
    says which bads it misses. Categoricals become codes ordered by the model's miss rate, so a split on a code
    is the best two-group partition of the categories (Breiman), after pooling categories with fewer than
    MIN_CAT_BADS bads. The bads scored are checked to be the ones the threshold was set on."""
    p = argparse.ArgumentParser(description='error_analysis_main')
    g18.add_model_args(p)
    p.add_argument('--bads-file', default=str(settings.data_curated / 'error_analysis_test_bads.parquet'),
                   help='local copy of the scored test bads with their raw columns; reused unless --rescore')
    p.add_argument('--rescore', action='store_true')
    p.add_argument('--result-json', default=str(settings.model_dir / 'error_analysis.json'))
    p.add_argument('--doc', default=str(Path(__file__).resolve().parents[2] / 'docs' / 'error_analysis.md'))
    a = p.parse_args()

    cfg = load_split(Path(a.split_json))['config']
    kinds = json.loads(Path(a.selection_json).read_text())['kinds']
    feats = sorted(set().union(*g18.model_features(a).values()))
    bads_file = Path(a.bads_file)

    spark = spark_session('error_analysis', conf={'spark.sql.execution.arrow.pyspark.enabled': 'true',
                                                  'spark.sql.execution.arrow.maxRecordsPerBatch': '50000',
                                                  'spark.driver.maxResultSize': '4g'})
    try:
        scores = spark.read.parquet(resolve_path(a.scores))
        thresholds = {}
        for c in MODELS:
            g = scores.groupBy(c).agg(F.count(F.lit(1)).alias('n'), F.sum(TARGET).alias('bad')).toPandas()
            thresholds[c] = f1_threshold(g, c)
            log.info('[%s] max-F1 threshold %s', c, {k: round(v, 4) for k, v in thresholds[c].items()})
        if bads_file.exists() and not a.rescore:
            log.info('Test bads already in %s; --rescore to score them again', bads_file)
        else:
            t = time.time()
            bads_file.parent.mkdir(parents=True, exist_ok=True)
            g18.scored_test(spark, a, cfg, bads_only=True, keep=feats).toPandas().to_parquet(bads_file)
            log.info('Scored every test bad in %.0f min -> %s', (time.time() - t) / 60, bads_file)
    finally:
        spark.stop()

    bads = pd.read_parquet(bads_file)
    log.info('%s test bads, %s features', f'{len(bads):,}', len(feats))
    missed = {}
    for c in MODELS:
        missed[c] = (bads[c] < thresholds[c]['threshold']).to_numpy()
        if int(missed[c].sum()) != thresholds[c]['missed_bads']:
            raise SystemExit(f"[{c}] {missed[c].sum():,} bads below the threshold here, "
                             f"{thresholds[c]['missed_bads']:,} in {a.scores}: the scores disagree")

    trees = {}
    for c in MODELS:
        x, order = encode(bads, feats, kinds, missed[c])
        tree = DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=int(MIN_LEAF_SHARE * len(bads)),
                                      random_state=42).fit(x, missed[c])
        auc = float(roc_auc_score(missed[c], tree.predict_proba(x)[:, 1]))
        trees[c] = {'auc': auc, 'leaves': leaves(tree, feats, kinds, order, x, missed[c]),
                    'text': export_text(tree, feature_names=feats, show_weights=False, decimals=4),
                    'importance': dict(sorted(((f, float(v)) for f, v in zip(feats, tree.feature_importances_) if v > 0),
                                              key=lambda kv: -kv[1]))}
        log.info('[%s] tree AUC %.4f; splits on %s', c, auc, list(trees[c]['importance']))

    combo = pd.DataFrame(missed)
    overlap = (combo.value_counts().rename('bads').reset_index()
               .assign(**{'missed by': lambda d: d[MODELS].apply(
                   lambda r: ', '.join(m for m in MODELS if r[m]) or 'none', axis=1)}))
    overlap['share of bads'] = overlap['bads'] / len(bads)
    overlap = overlap[['missed by', 'bads', 'share of bads']].to_dict('records')

    result = {'meta': {'best': BEST, 'split': cfg, 'max_depth': MAX_DEPTH, 'min_leaf_share': MIN_LEAF_SHARE,
                       'min_cat_bads': MIN_CAT_BADS, 'source': 'src/jobs/19_error_analysis_main.py',
                       'generated': datetime.now().isoformat(timespec='seconds')},
              'bads': len(bads), 'features': feats, 'thresholds': thresholds, 'overlap': overlap, 'trees': trees}
    Path(a.result_json).write_text(json.dumps(result, indent=2, default=float))
    write_doc(Path(a.doc), result, cfg)
    log.info('Result in %s, report in %s', a.result_json, a.doc)


if __name__ == '__main__':
    main()
