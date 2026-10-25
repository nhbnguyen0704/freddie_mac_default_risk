import argparse
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import norm
from pyspark import StorageLevel
from pyspark.sql import Column, DataFrame, functions as F
from helpers.config import settings
from helpers.period import complete_months, month_partitions
from helpers.report import md_table, write_report
from helpers.spark_io import curated_path, resolve_path, spark_session

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s | %(message)s')
log = logging.getLogger('lr_model_main')

TARGET = 'label'
WOE_SUFFIX = '_woe'

P_THRESHOLD = 0.05

MAX_ITER = 30

TOL = 1e-8

LOGIT_DECIMALS = 4

HASH_BITS = 31

RESAMPLED_KINDS = ('smoteenn', 'smotetomek')
CHUNKS_PER_MONTH = 10
SMOTE_RATIO = 0.1
SMOTE_K = 5
ENN_NEIGHBORS = 3
ENN_KIND = 'all'
CLEAN_STRATEGY = 'majority'
RANDOM_STATE = 42

@dataclass(frozen=True)
class Version:
    """One weighting version of the logistic regression: its name, kind and, for recency, the half-life in months."""
    name: str
    kind: str
    half_life: float = 0.0

    @property
    def label(self) -> str:
        return {'none': 'no weighting',
                'oversample': 'random oversampling of bads in lower-rate months',
                'undersample': 'random undersampling of goods in lower-rate months',
                'recency': f'recency weight, half-life {self.half_life:g} months',
                'smoteenn': 'SMOTE then edited nearest neighbours, per month, on every train row',
                'smotetomek': 'SMOTE then Tomek links, per month, on every train row'}[self.kind]

VERSIONS = [Version('baseline', 'none'),
            Version('oversample', 'oversample'),
            Version('undersample', 'undersample'),
            Version('recency_25', 'recency', 25),
            Version('recency_12', 'recency', 12),
            Version('recency_6', 'recency', 6),
            Version('smoteenn', 'smoteenn'),
            Version('smotetomek', 'smotetomek')]

def _ym(year: str = 'year', month: str = 'month') -> Column:
    return F.col(year) * 100 + F.col(month)

def _uniform(salt: Optional[str] = None) -> Column:
    """A seeded U[0, 1) per (loan, month), optionally salted."""
    cols = [F.col('loan_identifier'), F.col('period')] + ([F.lit(salt)] if salt else [])
    return F.pmod(F.xxhash64(*cols), F.lit(1 << HASH_BITS)) / F.lit(1 << HASH_BITS)

def _months_between(ym: np.ndarray, end_ym: int) -> np.ndarray:
    """Months from each YYYYMM in ym to end_ym."""
    return (end_ym // 100 - ym // 100) * 12 + (end_ym % 100 - ym % 100)

def weight_plan(months: pd.DataFrame, versions: List[Version], end_ym: int) -> Dict[str, Dict[str, Any]]:
    """Per version, its weights by month: bad multipliers (oversample) or good keep rates (undersample) that lift
    every month to the highest monthly bad rate, or the recency weight 0.5^(age / half-life)."""
    m = months.copy()
    m['rate'] = m['bad'] / m['n']
    target = float(m['rate'].max())
    good = m['n'] - m['bad']
    plan: Dict[str, Dict[str, Any]] = {}
    for v in versions:
        if v.kind == 'oversample':
            k = (target * good / ((1 - target) * m['bad'])).clip(lower=1)
            plan[v.name] = {'target_rate': target, 'bad_mult': dict(zip(m['ym'], k))}
        elif v.kind == 'undersample':
            f = (m['bad'] * (1 - target) / (target * good)).clip(upper=1)
            plan[v.name] = {'target_rate': target, 'good_keep': dict(zip(m['ym'], f))}
        elif v.kind == 'recency':
            age = _months_between(m['ym'].to_numpy(), end_ym)
            plan[v.name] = {'month_weight': dict(zip(m['ym'], 0.5 ** (age / v.half_life)))}
        else:
            plan[v.name] = {}
    return plan

def row_weights(v: Version, plan: Dict[str, Any], ym: np.ndarray, y: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Per-row weights of a version: oversampled bads get floor(k) or floor(k)+1 copies, undersampled goods 0 or 1,
    both drawn from the seeded u; recency rows their month's weight; otherwise 1."""
    if v.kind == 'oversample':
        k = pd.Series(ym).map(plan['bad_mult']).fillna(1.0).to_numpy()
        copies = np.floor(k) + (u < k - np.floor(k))
        return np.where(y == 1, copies, 1)
    if v.kind == 'undersample':
        f = pd.Series(ym).map(plan['good_keep']).fillna(1.0).to_numpy()
        return np.where(y == 1, 1, (u < f).astype(float))
    if v.kind == 'recency':
        return pd.Series(ym).map(plan['month_weight']).fillna(0.0).to_numpy()
    return np.ones(len(y))

def weight_summary(v: Version, plan: Dict[str, Any], months: pd.DataFrame) -> Dict[str, Any]:
    """The weighted rows, weighted bad rate, Kish effective rows and monthly bad-rate range a version's weights
    imply, from the monthly counts."""
    ym, n, bad = months['ym'].to_numpy(), months['n'].to_numpy(float), months['bad'].to_numpy(float)
    good = n - bad
    if v.kind == 'oversample':
        k = np.array([plan['bad_mult'][x] for x in ym])
        wb, wg, w2 = k * bad, good, k ** 2 * bad + good
    elif v.kind == 'undersample':
        f = np.array([plan['good_keep'][x] for x in ym])
        wb, wg, w2 = bad, good * f, bad + f * good
    elif v.kind == 'recency':
        w = np.array([plan['month_weight'][x] for x in ym])
        wb, wg, w2 = w * bad, w * good, w ** 2 * n
    else:
        wb, wg, w2 = bad, good, n
    rate = wb / (wb + wg)
    total = wb.sum() + wg.sum()
    out = {'weighted_rows': float(total), 'weighted_bad_rate': float(wb.sum() / total),
           'kish_effective_rows': float(total ** 2 / w2.sum()),
           'monthly_rate_min': float(rate.min()), 'monthly_rate_max': float(rate.max())}
    if 'target_rate' in plan:
        out['target_rate'] = plan['target_rate']
    if v.kind == 'recency':
        out['oldest_month_weight'] = float(min(plan['month_weight'].values()))
    return out

def snap_to_values(new: np.ndarray, values: List[np.ndarray]) -> np.ndarray:
    """Each synthetic value moved to the nearest real value of its column."""
    out = new.copy()
    for j, vals in enumerate(values):
        if len(vals) == 1:
            out[:, j] = vals[0]
            continue
        i = np.clip(np.searchsorted(vals, new[:, j]), 1, len(vals) - 1)
        lo, hi = vals[i - 1], vals[i]
        out[:, j] = np.where(new[:, j] - lo <= hi - new[:, j], lo, hi)
    return out

def smote_month(spark, df: DataFrame, ym: int, feats: List[str], n_good: int,
                cfg: Dict[str, Any]) -> Tuple[Optional[DataFrame], Dict[str, int]]:
    """SMOTE synthetic bads for one month, up to smote_ratio bads per good, built from all of the month's bads and
    snapped to real WoE values; None when the month has fewer than two bads or enough already. imblearn insists
    on two classes, so one placeholder good goes in."""
    from imblearn.over_sampling import SMOTE
    bads = (df.where(F.col(TARGET) == 1).select(*[F.col(c).cast('double') for c in feats]).toPandas()
            .sort_values(feats, kind='mergesort').to_numpy())
    n_bad = len(bads)
    target = int(np.ceil(cfg['smote_ratio'] * n_good))
    info = {'real_bads': n_bad, 'goods': n_good, 'synthetic_bads': 0}
    if n_bad < 2 or target <= n_bad:
        return None, info
    x = np.vstack([bads, np.zeros((1, len(feats)))])
    y = np.r_[np.ones(n_bad, dtype=np.int64), 0]
    xs, _ = SMOTE(sampling_strategy={1: target}, k_neighbors=min(cfg['smote_k'], n_bad - 1),
                  random_state=cfg['random_state'] + ym).fit_resample(x, y)
    row = df.agg(*[F.array_sort(F.collect_set(F.col(c).cast('double'))).alias(c) for c in feats]).first()
    new = snap_to_values(xs[len(x):], [np.array(row[c]) for c in feats])
    info['synthetic_bads'] = len(new)
    pdf = pd.DataFrame(new, columns=feats)
    pdf.insert(0, '_chunk', np.random.default_rng(cfg['random_state'] + ym)
               .integers(0, cfg['chunks_per_month'], len(pdf)).astype(np.int32))
    pdf.insert(1, TARGET, np.int32(1))
    pdf.insert(2, 'synthetic', True)
    schema = '_chunk int, label int, synthetic boolean, ' + ', '.join(f'{c} double' for c in feats)
    return spark.createDataFrame(pdf, schema), info

def resample_month(spark, df: DataFrame, ym: int, n_good: int, feats: List[str], kinds: Dict[str, str],
                   cfg: Dict[str, Any]) -> Tuple[DataFrame, Dict[str, int]]:
    """One month's rows, real and synthetic, cleaned by ENN or Tomek links per version, the month cut into seeded
    random chunks with neighbours searched within a chunk."""
    synthetic, info = smote_month(spark, df, ym, feats, n_good, cfg)

    def clean(pdf: pd.DataFrame) -> pd.DataFrame:
        from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks
        x = pdf[feats].to_numpy(dtype=np.float64)
        y = pdf[TARGET].to_numpy(dtype=np.int64)
        rows = pdf[[TARGET, 'synthetic', *feats]]
        out = []
        for name, kind in kinds.items():
            kept = rows
            if y.min() != y.max() and len(y) > cfg['enn_neighbors']:
                if kind == 'smoteenn':
                    cleaner = EditedNearestNeighbours(n_neighbors=cfg['enn_neighbors'], kind_sel=cfg['enn_kind'],
                                                      sampling_strategy=cfg['clean_strategy'])
                else:
                    cleaner = TomekLinks(sampling_strategy=cfg['clean_strategy'])
                cleaner.fit_resample(x, y)
                kept = rows.iloc[cleaner.sample_indices_]
            out.append(kept.assign(version=name))
        return pd.concat(out, ignore_index=True)[['version', TARGET, 'synthetic', *feats]]

    chunk = F.floor(_uniform('smote_chunk') * F.lit(cfg['chunks_per_month'])).cast('int')
    rows = df.select(chunk.alias('_chunk'), F.col(TARGET).cast('int').alias(TARGET), F.lit(False).alias('synthetic'),
                     *[F.col(c).cast('double') for c in feats])
    if synthetic is not None:
        rows = rows.unionByName(synthetic)
    schema = 'version string, label int, synthetic boolean, ' + ', '.join(f'{c} double' for c in feats)
    return rows.groupBy('_chunk').applyInPandas(clean, schema), info

def resample_train(spark, src: str, outs: Dict[str, Tuple[str, str]], end_ym: int,
                   counts: Dict[int, Tuple[int, int]], feats: List[str], cfg: Dict[str, Any],
                   keep_done: bool) -> Dict[str, Any]:
    """Write the SMOTEENN / SMOTETomek rows for every train month (resuming months already written when keep_done
    is set)."""
    parts = [(y, m, uri) for y, m, uri in month_partitions(spark, src) or [] if y * 100 + m <= end_ym]
    if not parts:
        raise SystemExit(f'{src} has no train months (up to {end_ym}).')
    if keep_done:
        done = {n: complete_months(spark, p) for n, (_, p) in outs.items()}
        for n in outs:
            log.info('[%s] %s of %s train months already resampled',
                     n, sum((y, m) in done[n] for y, m, _ in parts), len(parts))
    else:
        conf = spark._jsc.hadoopConfiguration()
        for _, p in outs.values():
            root = spark._jvm.org.apache.hadoop.fs.Path(resolve_path(p))
            root.getFileSystem(conf).delete(root, True)
        done = {n: set() for n in outs}
    todo = [(y, m, uri, [n for n in outs if (y, m) not in done[n]]) for y, m, uri in parts]
    todo = [t for t in todo if t[3]]
    t0 = time.time()
    for i, (y, m, uri, names) in enumerate(todo, 1):
        t = time.time()
        n, bad = counts[y * 100 + m]
        rows, info = resample_month(spark, spark.read.parquet(uri), y * 100 + m, n - bad, feats,
                                    {v: outs[v][0] for v in names}, cfg)
        if len(names) > 1:
            rows = rows.persist(StorageLevel.DISK_ONLY)
        try:
            for v in names:
                (rows.where(F.col('version') == v).drop('version')
                 .write.mode('overwrite').parquet(f'{outs[v][1].rstrip('/')}/year={y}/month={m}'))
        finally:
            rows.unpersist()
        log.info('[%s/%s] %s-%02d resampled for %s in %.0fs: %s real bads + %s synthetic, %s goods before '
                 'cleaning', i, len(todo), y, m, ', '.join(names), time.time() - t, f"{info['real_bads']:,}",
                 f"{info['synthetic_bads']:,}", f"{info['goods']:,}")
    return {'months': len(parts), 'months_resampled': len(todo), 'months_skipped': len(parts) - len(todo),
            'seconds': round(time.time() - t0)}

def resample_summary(df: DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Rows, bad rate, monthly bad-rate range and synthetic share of a resampled train set."""
    m = (df.groupBy('year', 'month')
         .agg(F.count(F.lit(1)).alias('n'), F.sum(TARGET).alias('bad'),
              F.sum(F.col('synthetic').cast('int')).alias('synthetic'))
         .toPandas())
    n, bad, syn = float(m['n'].sum()), float(m['bad'].sum()), float(m['synthetic'].sum())
    rate = m['bad'] / m['n']
    return {'weighted_rows': n, 'weighted_bad_rate': bad / n, 'kish_effective_rows': n,
            'monthly_rate_min': float(rate.min()), 'monthly_rate_max': float(rate.max()),
            'synthetic_rows': syn, 'real_rows': n - syn, **cfg}

def newton_scan(df: DataFrame, states: Dict[str, dict], versions: Dict[str, Version],
                plan: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """One pass over the rows that accumulates, for every unfinished version, the weighted log-likelihood,
    gradient, Hessian and outer product of scores (for sandwich errors) at its current coefficients."""
    active = [n for n, s in states.items() if not s['done']]
    cols = sorted({c for n in active for c in states[n]['keep']})
    at = {c: i for i, c in enumerate(cols)}
    work = {n: ([at[c] for c in states[n]['keep']], np.array(states[n]['beta'], dtype=float)) for n in active}
    vers = {n: versions[n] for n in active}
    plans = {n: plan[n] for n in active}

    def partial(batches):
        acc = {n: {'ll': 0.0, 'sw': 0.0, 'sw2': 0.0, 'g': np.zeros(len(idx) + 1), 'h': np.zeros((len(idx) + 1,) * 2),
                   'm': np.zeros((len(idx) + 1,) * 2)} for n, (idx, _) in work.items()}
        for pdf in batches:
            y = pdf[TARGET].to_numpy(dtype=np.float64)
            ym = pdf['_ym'].to_numpy()
            u = pdf['_u'].to_numpy(dtype=np.float64)
            xa = pdf[cols].to_numpy(dtype=np.float64)
            for n, (idx, b) in work.items():
                w = row_weights(vers[n], plans[n], ym, y, u)
                keep = w > 0
                if not keep.any():
                    continue
                x = np.empty((int(keep.sum()), len(idx) + 1))
                x[:, 0] = 1.0
                x[:, 1:] = xa[np.ix_(keep, idx)]
                yy, ww = y[keep], w[keep]
                eta = x @ b
                p = 1.0 / (1.0 + np.exp(-eta))
                a = acc[n]
                a['ll'] += float(np.sum(ww * (yy * eta - np.logaddexp(0.0, eta))))
                r = ww * (yy - p)
                a['g'] += x.T @ r
                a['h'] += x.T @ (x * (ww * p * (1.0 - p))[:, None])
                a['m'] += x.T @ (x * (r * r)[:, None])
                a['sw'] += float(ww.sum())
                a['sw2'] += float((ww * ww).sum())
        yield pd.DataFrame([{'version': n, 'll': a['ll'], 'sw': a['sw'], 'sw2': a['sw2'], 'g': a['g'].tolist(),
                             'h': a['h'].ravel().tolist(), 'm': a['m'].ravel().tolist()} for n, a in acc.items()])

    u = _uniform() if 'loan_identifier' in df.columns else F.lit(0.0)
    rows = (df.select(F.col(TARGET).cast('double').alias(TARGET), _ym().alias('_ym'), u.alias('_u'),
                      *[F.col(c).cast('double') for c in cols])
            .mapInPandas(partial, 'version string, ll double, sw double, sw2 double, g array<double>, '
                                  'h array<double>, m array<double>')
            .collect())
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        k = len(work[r['version']][0]) + 1
        o = out.setdefault(r['version'], {'ll': 0.0, 'sw': 0.0, 'sw2': 0.0, 'g': np.zeros(k),
                                          'h': np.zeros((k, k)), 'm': np.zeros((k, k))})
        o['ll'] += r['ll']
        o['sw'] += r['sw']
        o['sw2'] += r['sw2']
        o['g'] += np.array(r['g'])
        o['h'] += np.array(r['h']).reshape(k, k)
        o['m'] += np.array(r['m']).reshape(k, k)
    return out

def dependent_features(h: np.ndarray, feats: List[str]) -> List[str]:
    """The features whose Hessian column is a linear combination of the earlier ones."""
    out = []
    for j in range(1, len(h)):
        prev = list(range(j))
        coef, *_ = np.linalg.lstsq(h[np.ix_(prev, prev)], h[prev, j], rcond=None)
        if h[j, j] - h[j, prev] @ coef < 1e-10 * max(h[j, j], 1e-300):
            out.append(feats[j - 1])
    return out

def new_state(feats: List[str], waiting: bool = False) -> dict:
    """A fresh elimination state for a version: all features kept, coefficients at zero."""
    return {'keep': list(feats), 'beta': [0.0] * (len(feats) + 1), 'steps': [], 'iter': 0, 'done': False,
            'final': None, 'waiting': waiting}

def advance(name: str, v: Version, st: dict, s: Dict[str, Any], iv: pd.Series, threshold: float, max_iter: int,
            tol: float) -> None:
    """One Newton step for a version, then backward elimination once it converges: drop the highest p-value (lowest
    IV on a tie) and refit, until every p-value is at most threshold. An exactly collinear Hessian drops the
    lowest-IV dependent feature. Standard errors are model-based for the baseline and resampled versions
    (optimistic on resampled rows, which are refitted without elimination) and robust (sandwich) for weighted
    ones."""
    keep = st['keep']
    beta = np.array(st['beta'], dtype=float)
    try:
        step = np.linalg.solve(s['h'], s['g'])
    except np.linalg.LinAlgError:
        dep = dependent_features(s['h'], keep) or keep[:]
        worst = iv[dep].idxmin()
        st['steps'].append({'round': len(st['steps']) + 1, 'dropped': worst, 'p_value': float('nan'),
                            'iv': float(iv[worst]), 'coef': float('nan'), 'tied_at_max': len(dep),
                            'features_left': len(keep) - 1})
        log.warning('[%s] round %s: drop %s -- exactly collinear, no p-value (lowest IV of %s)', name,
                    len(st['steps']), worst, ', '.join(dep))
        keep.remove(worst)
        st.update(beta=[0.0] * (len(keep) + 1), iter=0)
        return
    beta = beta + step
    st['iter'] += 1
    moved = float(np.max(np.abs(step)))
    log.info('[%s] iter %s: log-lik %.6f, max |step| %.2e, %s features', name, st['iter'], s['ll'], moved,
             len(keep))
    if moved >= tol:
        if st['iter'] >= max_iter:
            raise SystemExit(f'[{name}] Newton did not converge in {max_iter} iterations '
                             f'(last max |step| {moved:.2e}).')
        st['beta'] = beta.tolist()
        return
    h_inv = np.linalg.inv(s['h'])
    if v.kind == 'none':
        cov, se_kind = h_inv, 'model-based'
    elif v.kind in RESAMPLED_KINDS:
        cov, se_kind = h_inv, 'model-based on resampled rows (optimistic)'
    else:
        cov, se_kind = h_inv @ s['m'] @ h_inv, 'robust (sandwich)'
    se = np.sqrt(np.diag(cov))
    z = beta / se
    pv = 2 * norm.sf(np.abs(z))
    p = pd.Series(pv[1:], index=keep).fillna(1)
    if p.max() <= threshold or v.kind in RESAMPLED_KINDS:
        st.update(beta=beta.tolist(), done=True,
                  final={'se': se.tolist(), 'z': z.tolist(), 'p': pv.tolist(), 'll': s['ll'], 'se_kind': se_kind,
                         'weighted_rows': s['sw'], 'kish_effective_rows': s['sw'] ** 2 / s['sw2']})
        if v.kind in RESAMPLED_KINDS:
            log.info("[%s] done: refitted the baseline's %s features on the resampled rows", name,
                     len(keep))
        else:
            log.info('[%s] done: %s features, every p-value <= %g after %s round(s)', name, len(keep),
                     threshold, len(st['steps']))
        return
    rank = pd.DataFrame({'p': p, 'iv': iv[keep]})
    rank['_p'] = rank['p'].round(12)
    worst = rank.sort_values(['_p', 'iv'], ascending=[False, True]).index[0]
    tied = int((rank['_p'] == rank.loc[worst, '_p']).sum())
    i = keep.index(worst)
    st['steps'].append({'round': len(st['steps']) + 1, 'dropped': worst, 'p_value': float(p[worst]),
                        'iv': float(iv[worst]), 'coef': float(beta[1 + i]), 'tied_at_max': tied,
                        'features_left': len(keep) - 1})
    log.info('[%s] round %s: drop %s (p %.4g, IV %.4f%s)', name, len(st['steps']), worst, p[worst], iv[worst],
             f', tie of {tied} broken by IV' if tied > 1 else '')
    keep.remove(worst)
    st.update(beta=np.delete(beta, 1 + i).tolist(), iter=0)

def coef_table(st: dict, iv: pd.Series) -> pd.DataFrame:
    """Coefficients, standard errors, z, p-values and IV of a finished version, and whether each sign is the
    expected negative one."""
    keep, f = st['keep'], st['final']
    t = pd.DataFrame({'coef': st['beta'][1:], 'std_err': f['se'][1:], 'z': f['z'][1:], 'p_value': f['p'][1:]},
                     index=keep)
    t['iv'] = iv[keep]
    t['sign_ok'] = t['coef'] < 0
    return t

def _save(path: Path, states: Dict[str, dict]) -> None:
    """Checkpoint every version's state to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'states': states, 'saved': datetime.now().isoformat(timespec='seconds')},
                               indent=2, default=float))

def score_tables(df: DataFrame, models: Dict[str, dict]) -> Dict[str, pd.DataFrame]:
    """Per version, (rounded logit -> rows, bads) over df, in one scan."""
    pairs = []
    for name, m in models.items():
        expr = F.lit(m['intercept'])
        for c, b in m['coefs'].items():
            expr = expr + F.col(c).cast('double') * F.lit(b)
        pairs.append(F.struct(F.lit(name).alias('version'), F.round(expr, LOGIT_DECIMALS).alias('logit')))
    s = (df.select(F.col(TARGET), F.explode(F.array(*pairs)).alias('s'))
         .groupBy('s.version', 's.logit').agg(F.sum(TARGET).alias('bad'), F.count(F.lit(1)).alias('n'))
         .toPandas())
    return {name: g for name, g in s.groupby('version')}

def average_precision(logit: np.ndarray, bad: np.ndarray, n: np.ndarray) -> float:
    """Average precision from logit groups, highest logit first."""
    order = np.argsort(-logit, kind='stable')
    tp = np.cumsum(bad[order])
    predicted = np.cumsum(n[order])
    if tp[-1] == 0:
        return float('nan')
    precision = tp / predicted
    recall = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))

def metrics(s: pd.DataFrame) -> Dict[str, Any]:
    """Rows, bads, bad rate, mean PD, AP and no-skill AP from a score table."""
    rows, bads = int(s['n'].sum()), int(s['bad'].sum())
    rate = bads / rows if rows else float('nan')
    return {'rows': rows, 'bads': bads, 'bad_rate': rate,
            'mean_pd': float((s['n'] / (1 + np.exp(-s['logit']))).sum() / rows) if rows else float('nan'),
            'average_precision': average_precision(s['logit'].to_numpy(), s['bad'].to_numpy(), s['n'].to_numpy()),
            'no_skill_average_precision': rate}

def write_doc(path: Path, result: dict) -> None:
    m = result['meta']
    base = result['versions']['baseline']['evaluation']['test']['average_precision'] \
        if 'baseline' in result['versions'] else float('nan')
    rows = []
    for name, r in result['versions'].items():
        tr, te, w = r['evaluation']['train'], r['evaluation']['test'], r['weights']
        rows.append({'version': name, 'weighting': r['weighting'], 'features': len(r['features']),
                     'train AP': tr['average_precision'], 'test AP': te['average_precision'],
                     'train - test': tr['average_precision'] - te['average_precision'],
                     'test vs baseline': f"{te['average_precision'] / base - 1:+.2%}" if base == base else '--',
                     'test mean PD': te['mean_pd'], 'weighted bad rate': w['weighted_bad_rate'],
                     'effective rows': w['kish_effective_rows']})
    comp = pd.DataFrame(rows)
    ranked = comp.sort_values('test AP', ascending=False)[['version', 'test AP', 'test vs baseline']]
    tr0 = next(iter(result['versions'].values()))['evaluation']
    preamble = (f"Logistic regression on the WoE features kept by `09_lr_vif_selection_main`, in "
                f"{len(result['versions'])} versions that differ only in how the train rows "
                f"({m['train_period']}) are weighted. Each is fitted by maximum likelihood (Newton/IRLS) on every "
                f"train row with its weights, then backward eliminated: drop the highest p-value (lowest IV on a "
                f"tie) and refit until every p-value is at most {m['p_threshold']:g}; SMOTEENN and SMOTETomek, "
                f"where present, are fitted on resampled rows instead (see the notes under All versions). "
                f"Every version is scored, "
                f"unchanged and unweighted, on all {tr0['train']['rows']:,} train rows and all "
                f"{tr0['test']['rows']:,} test rows (after {m['test_after']}).\n\n"
                f"Scored by average precision (AP), the step-wise area under the precision-recall curve "
                f"(sklearn's average_precision_score). No-skill AP is the bad rate: train "
                f"{tr0['train']['bad_rate']:.4f}, test {tr0['test']['bad_rate']:.4f}.")
    notes = ("- Over/undersampling are exact random resamples, applied as frequency weights (0, 1, 2, ...) drawn "
             "from a seeded hash of (loan, month): the same rows every Newton scan. Both lift every lower month to "
             f"the highest monthly train bad rate ({m['target_rate']:.4%}).\n"
             "- Weighted versions use robust (sandwich) standard errors, so a duplicated bad counts once in the "
             "p-values; the baseline keeps the model-based ones.\n"
             "- Resampling raises the fitted bad rate, so those versions' mean PD overstates the real bad rate. "
             "That moves the intercept, not the ranking: average precision is unaffected by it.\n"
             "- Effective rows is Kish's (sum w)^2 / sum w^2: how many equally weighted rows the weights are "
             "worth.")
    resampled = {n: r for n, r in result['versions'].items() if r['kind'] in RESAMPLED_KINDS}
    if resampled:
        c = next(iter(resampled.values()))['weights']
        notes += ("\n- SMOTEENN / SMOTETomek (imblearn) run on every train row, one month at a time. SMOTE "
                  f"(k={c['smote_k']}) works on all of the month's bads together and adds synthetic bads up to "
                  f"{c['smote_ratio']:g} bads per good, each snapped to the nearest real WoE value per feature. "
                  f"Then ENN (n_neighbors={c['enn_neighbors']}, kind_sel='{c['enn_kind']}') or Tomek links removes "
                  f"{c['clean_strategy']}-class rows, with the month cut into {c['chunks_per_month']} seeded random "
                  "chunks (one per Spark task) and neighbours searched within a chunk. They refit the baseline's "
                  "final features "
                  "with no elimination: p-values on synthetic rows are too small to select on. Their weighted "
                  "rows and effective rows count the resampled rows, synthetic ones included: "
                  + '; '.join(f"{n} {r['weights']['real_rows']:,.0f} real + {r['weights']['synthetic_rows']:,.0f} "
                              f"synthetic" for n, r in resampled.items()) + '.')
    sections = [('Test average precision (AP), best first', md_table(ranked, floatfmt='{:.4f}')),
                ('All versions', md_table(comp, floatfmt='{:.4f}') + '\n\n' + notes)]
    for name, r in result['versions'].items():
        coef = pd.DataFrame(r['coefficients'])
        steps = r['steps']
        body = (f"{r['weighting']}. Intercept {r['intercept']:.6f}; {len(r['features'])} features; "
                f"standard errors {r['se_kind']}.\n\n" + md_table(coef, floatfmt='{:.4g}')
                + '\n\n' + (md_table(pd.DataFrame(steps), floatfmt='{:.4g}') if steps
                            else "_No elimination: the baseline's final features, refitted on the resampled rows._"
                            if r['kind'] in RESAMPLED_KINDS
                            else '_No feature removed: every p-value was already within the threshold._'))
        sections.append((f'Version: {name}', body))
    write_report(path, 'Logistic Regression: Weighting Versions', 'src/jobs/10_lr_model_main.py', sections,
                 preamble)

def main() -> None:
    p = argparse.ArgumentParser(description='lr_model_main')
    p.add_argument('--selected', default=curated_path('training_set_woe_selected'))
    p.add_argument('--selection-json', default=str(settings.model_dir / 'vif_selection.json'))
    p.add_argument('--binning-json', default=str(settings.model_dir / 'woe_binning.json'))
    p.add_argument('--versions', default=','.join(v.name for v in VERSIONS),
                   help=f"comma-separated subset of: {', '.join(v.name for v in VERSIONS)}")
    p.add_argument('--p-threshold', type=float, default=P_THRESHOLD)
    p.add_argument('--max-iter', type=int, default=MAX_ITER, help='Newton iterations per fit')
    p.add_argument('--progress-json', default=str(settings.model_dir / 'lr_model_progress.json'))
    p.add_argument('--resume', action='store_true', help='continue from --progress-json')
    p.add_argument('--model-json', default=str(settings.model_dir / 'lr_models.json'))
    p.add_argument('--doc', default=str(Path(__file__).resolve().parents[2] / 'docs' / 'lr_model.md'))
    g = p.add_argument_group('SMOTEENN / SMOTETomek')
    g.add_argument('--chunks-per-month', type=int, default=CHUNKS_PER_MONTH,
                   help='ENN / Tomek: each month is cut into this many random chunks, one per Python worker, and '
                        'their neighbours are searched within a chunk (SMOTE always uses the whole month\'s bads)')
    g.add_argument('--smote-ratio', type=float, default=SMOTE_RATIO,
                   help="SMOTE's sampling_strategy: bads per good after oversampling, in every month")
    g.add_argument('--smote-k', type=int, default=SMOTE_K, help="SMOTE's k_neighbors")
    g.add_argument('--enn-neighbors', type=int, default=ENN_NEIGHBORS, help="ENN's n_neighbors")
    g.add_argument('--enn-kind', choices=('all', 'mode'), default=ENN_KIND, help="ENN's kind_sel")
    g.add_argument('--clean-strategy', choices=('majority', 'all'), default=CLEAN_STRATEGY,
                   help="sampling_strategy of ENN and Tomek links")
    g.add_argument('--random-state', type=int, default=RANDOM_STATE)
    g.add_argument('--resampled-root', default=curated_path('training_set_resampled'))
    g.add_argument('--reuse-resampled', action='store_true',
                   help='keep the months already resampled under --resampled-root and resample only the '
                        'missing ones (--resume does the same)')
    a = p.parse_args()

    by_name = {v.name: v for v in VERSIONS}
    wanted = [x.strip() for x in a.versions.split(',') if x.strip()]
    unknown = [x for x in wanted if x not in by_name]
    if unknown:
        raise SystemExit(f"Unknown version(s): {', '.join(unknown)}")
    versions = {n: by_name[n] for n in wanted}
    resampled = [n for n, v in versions.items() if v.kind in RESAMPLED_KINDS]
    if resampled and 'baseline' not in versions:
        raise SystemExit(f"{', '.join(resampled)} refit the baseline's final features: add baseline to --versions.")
    if not 0 < a.smote_ratio <= 1:
        raise SystemExit('--smote-ratio must be in (0, 1].')
    if a.chunks_per_month < 1:
        raise SystemExit('--chunks-per-month must be at least 1.')

    selection = json.loads(Path(a.selection_json).read_text())
    period = selection['meta']['train_period']
    end = pd.Timestamp(period['analysis_end'])
    end_ym = end.year * 100 + end.month
    feats = list(selection['kept'])
    specs = json.loads(Path(a.binning_json).read_text())['features']
    iv = pd.Series({f'{c}{WOE_SUFFIX}': s.get('iv') or 0.0 for c, s in specs.items() if 'bins' in s})
    missing_iv = [f for f in feats if f not in iv.index]
    if missing_iv:
        raise SystemExit(f"No IV in {a.binning_json} for: {', '.join(missing_iv)}")
    log.info('%s candidate features, %s versions (%s); train months %s', len(feats), len(versions),
             ', '.join(versions), period['analysis_period'])

    progress = Path(a.progress_json)
    if a.resume:
        if not progress.exists():
            raise SystemExit(f'Nothing to resume: no {progress}.')
        states = json.loads(progress.read_text())['states']
        if set(states) != set(versions) or any(not set(s['keep']) <= set(feats) for s in states.values()):
            raise SystemExit(f'{progress} is from another set of versions or features: rerun without --resume.')
        log.info('Resuming from %s: %s', progress,
                 ', '.join(f"{n} {'done' if s['done'] else f'{len(s['keep'])} feats, iter {s['iter']}'}"
                           for n, s in states.items()))
    else:
        states = {n: new_state(feats, waiting=n in resampled) for n in versions}

    spark = spark_session('lr_model', conf={'spark.sql.execution.arrow.pyspark.enabled': 'true'})
    try:
        data = spark.read.parquet(resolve_path(a.selected))
        absent = sorted(set(feats) - set(data.columns))
        if absent:
            raise SystemExit(f"Kept features not in {a.selected}: {', '.join(absent)}")

        train, test = data.where(_ym() <= end_ym), data.where(_ym() > end_ym)

        months = (train.groupBy(_ym().alias('ym'))
                  .agg(F.count(F.lit(1)).alias('n'), F.sum(TARGET).alias('bad'))
                  .toPandas().sort_values('ym').reset_index(drop=True))
        months[['n', 'bad']] = months[['n', 'bad']].astype(float)
        plan = weight_plan(months, list(versions.values()), end_ym)
        summaries = {n: weight_summary(v, plan[n], months) for n, v in versions.items() if n not in resampled}
        for n, s in summaries.items():
            log.info('[%s] %s: weighted bad rate %.3f%%, monthly %.3f%%-%.3f%%, effective rows %s', n,
                     versions[n].label, 100 * s['weighted_bad_rate'], 100 * s['monthly_rate_min'],
                     100 * s['monthly_rate_max'], f"{s['kish_effective_rows']:,.0f}")
        for n in resampled:
            if states[n].get('waiting'):
                log.info("[%s] %s: waits for the baseline's final features, then resamples every train month",
                         n, versions[n].label)

        cfg = {'chunks_per_month': a.chunks_per_month, 'smote_ratio': a.smote_ratio,
               'smote_k': a.smote_k, 'enn_neighbors': a.enn_neighbors, 'enn_kind': a.enn_kind,
               'clean_strategy': a.clean_strategy, 'random_state': a.random_state}
        counts = {int(r.ym): (int(r.n), int(r.bad)) for r in months.itertuples()}
        sources: Dict[str, DataFrame] = {}

        def path_of(n: str) -> str:
            return f"{a.resampled_root.rstrip('/')}/{n}"

        def start_resampled() -> None:
            waiting = [n for n in resampled if n not in sources and states[n].get('waiting')]
            if waiting and states['baseline']['done']:
                log.info('[%s] resampling every train row, one month at a time, %s chunks per month',
                         ', '.join(waiting), a.chunks_per_month)
                run = resample_train(spark, a.selected, {n: (versions[n].kind, path_of(n)) for n in waiting}, end_ym,
                                     counts, states['baseline']['keep'], cfg, keep_done=a.reuse_resampled or a.resume)
                log.info('[%s] resampled rows in %s: %s', ', '.join(waiting), ', '.join(map(path_of, waiting)), run)
                for n in waiting:
                    states[n].update(new_state(states['baseline']['keep']))
            for n in resampled:
                if n in sources or states[n].get('waiting'):
                    continue
                if not complete_months(spark, path_of(n)):
                    raise SystemExit(f'[{n}] was started on {path_of(n)}, which is gone: rerun without --resume.')
                sources[n] = spark.read.parquet(path_of(n))

        t0, scans = time.time(), 0
        start_resampled()
        while not all(s['done'] for s in states.values()):
            t = time.time()

            groups = [(train, [n for n in states if n not in resampled])] + [(df, [n]) for n, df in sources.items()]
            for df, names in groups:
                live = {n: states[n] for n in names if not states[n]['done'] and not states[n].get('waiting')}
                if not live:
                    continue
                stats = newton_scan(df, live, versions, plan)
                scans += 1
                for n, s in stats.items():
                    advance(n, versions[n], states[n], s, iv, a.p_threshold, a.max_iter, TOL)
            _save(progress, states)
            start_resampled()
            log.info('Round done in %.0fs; %s of %s versions finished', time.time() - t,
                     sum(s['done'] for s in states.values()), len(states))
        log.info('All versions fitted in %s scans, %.0f min', scans, (time.time() - t0) / 60)
        for n in resampled:
            summaries[n] = resample_summary(sources[n], cfg)
            s = summaries[n]
            log.info('[%s] %s: %s real + %s synthetic rows, bad rate %.3f%%', n, versions[n].label,
                     f"{s['real_rows']:,.0f}", f"{s['synthetic_rows']:,.0f}", 100 * s['weighted_bad_rate'])

        tables = {n: coef_table(states[n], iv) for n in versions}
        models = {n: {'intercept': states[n]['beta'][0], 'coefs': tables[n]['coef'].to_dict()} for n in versions}
        t = time.time()
        ev_train = score_tables(train, models)
        ev_test = score_tables(test, models)
        log.info('Scored train and test for every version in %.0fs', time.time() - t)
    finally:
        spark.stop()

    out_versions = {}
    for n, v in versions.items():
        tr, te = metrics(ev_train[n]), metrics(ev_test[n])
        log.info('[%s] Average precision train %.4f, test %.4f (no-skill %.4f / %.4f)', n,
                 tr['average_precision'], te['average_precision'], tr['bad_rate'], te['bad_rate'])
        st, tb = states[n], tables[n]
        out_versions[n] = {
            'weighting': v.label, 'kind': v.kind, 'half_life_months': v.half_life or None,
            'weights': summaries[n], 'se_kind': st['final']['se_kind'],
            'intercept': st['beta'][0], 'features': st['keep'],
            'coefficients': [{k: (bool(x) if k == 'sign_ok' else x) for k, x in r.items()}
                             for r in tb.reset_index(names='feature').to_dict('records')],
            'steps': st['steps'], 'log_likelihood': st['final']['ll'],
            'evaluation': {'train': tr, 'test': te}}
    result = {
        'meta': {'train_period': period['analysis_period'], 'test_after': f'{end:%Y-%m}',
                 'p_threshold': a.p_threshold, 'candidates': len(feats),
                 'target_rate': float(months['bad'].div(months['n']).max()),
                 'rule': 'drop highest p-value, lowest IV on a tie, refit',
                 'fit': 'weighted maximum likelihood, Newton/IRLS on every train row '
                        '(SMOTEENN / SMOTETomek: on their resampled rows)',
                 'metric': 'average precision (AP) over score groups (logit rounded to 1e-4), unweighted',
                 'selected': a.selected, 'selection_json': a.selection_json, 'binning_json': a.binning_json,
                 'generated': datetime.now().isoformat(timespec='seconds'),
                 'source': 'src/jobs/10_lr_model_main.py'},
        'versions': out_versions,
    }
    out = Path(a.model_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=float))
    write_doc(Path(a.doc), result)
    log.info('Models in %s, report in %s', out, a.doc)

if __name__ == '__main__':
    main()
