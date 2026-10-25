"""Shared by the random forest jobs (16_rf_perm_elimination_main onwards).

They reuse the XGBoost split (gbm_split.json), candidates
(gbm_corr_selection.json) and samples; only what scikit-learn's
RandomForestClassifier needs differently lives here:

- strings as numbers: each category's code is its rank by weighted train bad
  rate, missing a category of its own (anything unseen in train counts as
  missing). For a 0/1 target, splits on these codes reach the best
  two-group partition of the categories (Breiman), without one-hot columns;
- scoring every row in Spark: the forest is a few hundred MB, so it is read
  from a local file by each Python worker (once per worker, cached) rather
  than broadcast. This assumes local mode, as spark_session runs.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Sequence
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, functions as F
from helpers.gbm import SPLIT, TARGET, average_precision

MISSING = '__missing__'
PROB_DECIMALS = 6

_MODELS: Dict[str, object] = {}


def bad_rate_codes(pdf: pd.DataFrame, cat: List[str], y: np.ndarray, w: np.ndarray) -> Dict[str, dict]:
    """Per string feature: category -> code, lowest weighted bad rate first
    (ties by name), missing as a category; 'missing' is its code."""
    out = {}
    for c in cat:
        g = (pd.DataFrame({'g': pdf[c].astype(object).where(pdf[c].notna(), MISSING).to_numpy(),
                           'bad': w * y, 'w': w})
             .groupby('g', sort=False)[['bad', 'w']].sum())
        g['rate'] = g['bad'] / g['w']
        order = g.reset_index().sort_values(['rate', 'g'])['g'].tolist()
        codes = {k: float(i) for i, k in enumerate(order)}
        out[c] = {'codes': {k: v for k, v in codes.items() if k != MISSING},
                  'missing': codes.get(MISSING, float(len(order)))}
    return out

def to_matrix(pdf: pd.DataFrame, feats: Sequence[str], kinds: Dict[str, str], enc: Dict[str, dict]) -> np.ndarray:
    """RF input, float32 in column order (the tree builder reads a feature at a
    time): numerics as they are (NaN is missing, which sklearn's trees route),
    strings as their bad-rate codes."""
    x = np.empty((len(pdf), len(feats)), dtype=np.float32, order='F')
    for j, c in enumerate(feats):
        s = pdf[c]
        if kinds[c] != 'categorical':
            x[:, j] = s.to_numpy(dtype=np.float32, na_value=np.nan)
            continue
        e = enc[c]
        if isinstance(s.dtype, pd.CategoricalDtype):
            lut = np.array([e['codes'].get(k, e['missing']) for k in s.cat.categories] + [e['missing']], np.float32)
            x[:, j] = lut[s.cat.codes.to_numpy()]
        else:
            x[:, j] = s.map(e['codes']).fillna(e['missing']).to_numpy(np.float32)
    return x

def tree_sum(trees: Sequence, x: np.ndarray, threads: int = 0) -> np.ndarray:
    """Sum of the trees' P(bad) over the rows (the forest's P(bad) times the
    number of trees), the trees run in threads (their predict releases the GIL)."""
    def one(t):
        return t.predict_proba(x, check_input=False)[:, 1].astype(np.float64)
    if not trees:
        return np.zeros(len(x))
    with ThreadPoolExecutor(max_workers=threads or None) as ex:
        return np.sum(list(ex.map(one, trees)), axis=0)

def bad_proba(model, x: np.ndarray) -> np.ndarray:
    """P(bad) per row from a fitted classifier."""
    return model.predict_proba(x)[:, list(model.classes_).index(1)]

def cached_model(path: str):
    """The forest at path, loaded once per Python worker."""
    if path not in _MODELS:
        import joblib
        _MODELS.clear()
        m = joblib.load(path)
        m.set_params(n_jobs=1)
        _MODELS[path] = m
    return _MODELS[path]

def score_counts(df: DataFrame, model_path: Path, feats: List[str], kinds: Dict[str, str],
                 enc: Dict[str, dict]) -> pd.DataFrame:
    """(split, rounded P(bad)) -> rows, bads over every row of df, in ONE scan."""
    path = str(Path(model_path).resolve())
    enc = {c: enc[c] for c in feats if kinds[c] == 'categorical'}

    def score(batches):
        model = cached_model(path)
        acc, frames = None, []
        for pdf in batches:
            p = np.round(bad_proba(model, to_matrix(pdf, feats, kinds, enc)).astype(np.float64), PROB_DECIMALS)
            frames.append(pd.DataFrame({SPLIT: pdf[SPLIT].to_numpy(), 'p': p, 'y': pdf[TARGET].to_numpy(np.int64)})
                          .groupby([SPLIT, 'p'], sort=False)['y'].agg(n='size', bad='sum').reset_index())
            if len(frames) >= 50:
                acc = pd.concat(([acc] if acc is not None else []) + frames).groupby([SPLIT, 'p'], sort=False)[
                    ['n', 'bad']].sum().reset_index()
                frames = []
        if frames or acc is not None:
            yield pd.concat(([acc] if acc is not None else []) + frames).groupby([SPLIT, 'p'], sort=False)[
                ['n', 'bad']].sum().reset_index()

    return (df.select(SPLIT, TARGET, *feats)
            .mapInPandas(score, 'split string, p double, n long, bad long')
            .groupBy(SPLIT, 'p').agg(F.sum('n').alias('n'), F.sum('bad').alias('bad'))
            .toPandas())

def split_ap(s: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Rows, bads, bad rate and average precision per split, from score_counts' grouped table."""
    out = {}
    for split, g in s.groupby(SPLIT):
        n, bad = g['n'].to_numpy(float), g['bad'].to_numpy(float)
        out[split] = {'rows': int(n.sum()), 'bads': int(bad.sum()), 'bad_rate': float(bad.sum() / n.sum()),
                      'average_precision': average_precision(g['p'].to_numpy(), bad, n)}
    return out
