import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from functools import reduce
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from pyspark.ml.feature import Bucketizer
from pyspark.sql import DataFrame, functions as F
from helpers.config import settings
from helpers.gbm import SPLIT, TARGET, average_precision, dev_rows, feature_kinds, load_split
from helpers.report import md_table, write_report
from helpers.spark_io import curated_path, spark_session

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s | %(message)s')
log = logging.getLogger('gbm_corr_selection_main')

CORR_THRESHOLD = 0.8

UNIVARIATE_BINS = 100

DISCRETE_MAX = 1000
MISSING = '__missing__'

MAX_EXACT_DISTINCT = 1_000_000
RANK_BUCKETS = 50_000

EDGE_SAMPLE = 0.005
EDGE_SEED = 11

os.environ.setdefault('PYSPARK_PYTHON', sys.executable)

def edges_of(x: np.ndarray, buckets: int, discrete_max: int = 0) -> np.ndarray:
    """Bin edges for x: its distinct values when there are at most discrete_max, else buckets equal-frequency
    quantiles, with any value holding more than 1/buckets of the rows given a bin of its own."""
    if not len(x):
        return np.array([], dtype=np.float64)
    vals, cnt = np.unique(x, return_counts=True)
    if len(vals) <= discrete_max:
        return vals
    heavy = vals[cnt > len(x) / buckets]
    rest = x[~np.isin(x, heavy)]
    q = np.quantile(rest, np.arange(1, buckets) / buckets) if len(rest) else np.array([])
    return np.unique(np.r_[q, heavy, np.nextafter(heavy, np.inf)])

def sample_edges(train: DataFrame, cols: List[str], buckets: int, discrete_max: int = 0) -> Dict[str, np.ndarray]:
    """edges_of per column, on a seeded EDGE_SAMPLE sample of train."""
    pdf = train.sample(fraction=EDGE_SAMPLE, seed=EDGE_SEED).select(
        *[F.col(c).cast('double').alias(c) for c in cols]
    ).toPandas()
    return {c: edges_of(pdf[c].dropna().to_numpy(np.float64), buckets, discrete_max) for c in cols}

def numeric_edges(train: DataFrame, cols: List[str], bins: int) -> Dict[str, List[float]]:
    """The univariate binning edges of each numeric feature."""
    return {c: e.tolist() for c, e in sample_edges(train, cols, bins, DISCRETE_MAX).items()}

def group_counts(df: DataFrame, kinds: Dict[str, str], edges: Dict[str, List[float]]) -> pd.DataFrame:
    """Rows and bads per (feature, group, split) in one scan: a numeric feature's group is its bin, a string's its
    category, missing a group of its own."""
    cols = list(kinds)
    cuts = {c: np.asarray(edges[c]) for c in cols if kinds[c] == 'numerical'}

    def partial(batches):
        acc: Dict[Tuple[str, str, str], List[int]] = defaultdict(lambda: [0, 0])
        for pdf in batches:
            y = pdf[TARGET].to_numpy(dtype=np.int64)
            s = pdf[SPLIT].to_numpy()
            for c in cols:
                v = pdf[c]
                if kinds[c] == 'numerical':
                    x = v.to_numpy(dtype=np.float64, na_value=np.nan)
                    g = np.where(np.isnan(x), -1, np.searchsorted(cuts[c], x, side='right'))
                else:
                    g = v.to_numpy(dtype=object, na_value=MISSING)
                t = (pd.DataFrame({'g': g, 's': s, 'y': y})
                     .groupby(['g', 's'], sort=False)['y'].agg(['size', 'sum']))
                for (gg, ss), (n, b) in zip(t.index, t.to_numpy()):
                    if kinds[c] == 'numerical':
                        gg = MISSING if gg < 0 else str(gg)
                    a = acc[(c, gg, ss)]
                    a[0] += int(n)
                    a[1] += int(b)
        yield pd.DataFrame([(c, g, s, n, b) for (c, g, s), (n, b) in acc.items()],
                           columns=['feature', 'grp', SPLIT, 'n', 'bad'])

    return (df.select(TARGET, SPLIT, *cols)
            .mapInPandas(partial, 'feature string, grp string, split string, n long, bad long')
            .groupBy('feature', 'grp', SPLIT).agg(F.sum('n').alias('n'), F.sum('bad').alias('bad'))
            .toPandas())

def univariate_ap(counts: pd.DataFrame) -> pd.DataFrame:
    """Each feature alone as a lookup scorer: the train bad rate of its group, scored by AP on valid (a group
    unseen in train gets the overall rate) and on train."""
    rows = []
    for c, g in counts.groupby('feature'):
        tr = g[g[SPLIT] == 'train'].set_index('grp')
        va = g[g[SPLIT] == 'valid'].set_index('grp')
        overall = tr['bad'].sum() / tr['n'].sum()
        rate = tr['bad'] / tr['n']
        score_va = va.index.map(rate).to_numpy(dtype=float, na_value=overall)
        rows.append({'feature': c,
                     'univariate_ap_valid': average_precision(score_va, va['bad'].to_numpy(float),
                                                              va['n'].to_numpy(float)),
                     'univariate_ap_train': average_precision(rate.to_numpy(float), tr['bad'].to_numpy(float),
                                                              tr['n'].to_numpy(float)),
                     'groups': len(tr), 'missing_share_train': float(tr['n'].get(MISSING, 0) / tr['n'].sum()),
                     'unseen_in_train_valid_rows': int(va.loc[~va.index.isin(tr.index), 'n'].sum())})
    return pd.DataFrame(rows).set_index('feature')

def categories(counts: pd.DataFrame, kinds: Dict[str, str]) -> Dict[str, List[str]]:
    """Per string feature, its train categories, most frequent first."""
    out = {}
    for c in [c for c, k in kinds.items() if k == 'categorical']:
        g = counts[(counts['feature'] == c) & (counts[SPLIT] == 'train') & (counts['grp'] != MISSING)]
        out[c] = g.sort_values(['n', 'grp'], ascending=[False, True])['grp'].tolist()
    return out

def null_masks(df: DataFrame, cols: List[str]) -> Dict[str, str]:
    """For every nullable column, a representative column with exactly the same null rows, so pairs can share one
    non-null mask."""
    nulls = df.agg(*[F.sum(F.col(c).isNull().cast('long')).alias(c) for c in cols]).collect()[0].asDict()
    nullable = [c for c in cols if nulls[c]]
    if not nullable:
        return {}
    pairs = [(a, b) for i, a in enumerate(nullable) for b in nullable[i + 1:] if nulls[a] == nulls[b]]
    diff = {}
    if pairs:
        r = df.agg(*[F.sum((F.col(a).isNull() != F.col(b).isNull()).cast('long')).alias(f'{a}||{b}')
                     for a, b in pairs]).collect()[0].asDict()
        diff = {tuple(k.split('||')): v for k, v in r.items()}
    rep: Dict[str, str] = {}
    for c in nullable:
        rep[c] = next((r for r in dict.fromkeys(rep.values()) if diff.get((r, c), diff.get((c, r))) == 0), c)
    return rep

def rank_tables(df: DataFrame, cols: List[str], masks: Dict[str, str], distinct: Dict[str, int]) -> Dict[str, dict]:
    """Per numeric column, its value groups and their midranks, both over all its non-null rows and within each
    null mask. Columns with more than MAX_EXACT_DISTINCT values are ranked on a RANK_BUCKETS-quantile grid."""
    reps = list(dict.fromkeys(masks.values()))
    wide = [c for c in cols if distinct[c] > MAX_EXACT_DISTINCT]
    grids: Dict[str, np.ndarray] = {}
    if wide:
        t = time.time()
        grids = {c: np.r_[-np.inf, q] for c, q in sample_edges(df, wide, RANK_BUCKETS).items()}
        log.info('%s-quantile grids for %s in %.0fs', RANK_BUCKETS, ', '.join(wide), time.time() - t)
    out = {}
    for i, c in enumerate(cols, 1):
        t = time.time()
        x = F.col(c).cast('double')
        exact = c not in grids
        if exact:
            key = x
        else:
            key = Bucketizer(splits=grids[c].tolist() + [np.inf], inputCol=c, outputCol='_b')
        base = df.where(F.col(c).isNotNull())
        if not exact:
            base = key.transform(base.withColumn(c, x))
        g = (base.groupBy((x if exact else F.col('_b').cast('long')).alias('k'))
             .agg(F.count(F.lit(1)).alias('all'),
                  *[F.sum(F.col(r).isNotNull().cast('long')).alias(r) for r in reps])
             .toPandas().sort_values('k').reset_index(drop=True))
        g['lo'] = g['k'].to_numpy(np.float64) if exact else grids[c][g['k'].to_numpy(np.int64)]
        ranks, sizes, sumsq = {}, {}, {}
        for cond in ['all'] + reps:
            n = g[cond].to_numpy(np.float64)
            size = n.sum()
            r = ((np.cumsum(n) - (n - 1) / 2) - (size + 1) / 2) / max(size, 1)
            ranks[cond], sizes[cond], sumsq[cond] = r, float(size), float(np.sum(n * r * r))
        out[c] = {'lo': g['lo'].to_numpy(np.float64), 'ranks': ranks, 'sizes': sizes, 'sumsq': sumsq,
                  'groups': len(g), 'exact': exact}
        log.info('[%s/%s] %s: %s value groups (%s) in %.0fs', i, len(cols), c, f'{len(g):,}',
                 'exact' if exact else 'quantile buckets', time.time() - t)
    return out

def cond_of(c: str, masks: Dict[str, str]) -> str:
    """The null mask a column is ranked within when paired: its representative, or 'all'."""
    return masks.get(c, 'all')

def cross_sums(df: DataFrame, cols: List[str], masks: Dict[str, str],
               tables: Dict[str, dict]) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """For every column pair, the sum over rows of the product of their centred ranks, each ranked on the rows
    where both are non-null, in one pass."""
    p = len(cols)
    pairs = [(i, j) for i in range(p) for j in range(i + 1, p)]
    vecs = sorted({(i, cond_of(cols[j], masks)) for i, j in pairs} | {(j, cond_of(cols[i], masks)) for i, j in pairs})
    slot = {v: k for k, v in enumerate(vecs)}
    left = np.array([slot[(i, cond_of(cols[j], masks))] for i, j in pairs])
    right = np.array([slot[(j, cond_of(cols[i], masks))] for i, j in pairs])
    reps = list(dict.fromkeys(masks.values()))
    bc = df.sparkSession.sparkContext.broadcast(
        [(tables[c]['lo'], tables[c]['ranks']) for c in cols]
    )
    chunk = 4096

    def partial(batches):
        los = [lo for lo, _ in bc.value]
        rk = [r for _, r in bc.value]
        acc = np.zeros(len(pairs))
        for pdf in batches:
            x = pdf[cols].to_numpy(dtype=np.float64, na_value=np.nan)
            present = ~np.isnan(x)
            flag = {r: present[:, cols.index(r)] for r in reps}
            idx = [np.maximum(np.searchsorted(los[i], x[:, i], side='right') - 1, 0) for i in range(p)]
            a = np.zeros((len(x), len(vecs)))
            for k, (i, cond) in enumerate(vecs):
                ok = present[:, i] if cond == 'all' else present[:, i] & flag[cond]
                a[:, k] = np.where(ok, rk[i][cond][idx[i]], 0)
            for s in range(0, len(a), chunk):
                b = a[s:s + chunk]
                acc += np.einsum('ij,ij->j', b[:, left], b[:, right])
        yield pd.DataFrame({'s': [acc.tolist()]})

    parts = (df.select(*[F.col(c).cast('double').alias(c) for c in cols])
             .mapInPandas(partial, 's array<double>').collect())
    bc.unpersist()
    return reduce(np.add, (np.array(r['s']) for r in parts)), pairs

def spearman(cols: List[str], masks: Dict[str, str], tables: Dict[str, dict], sums: np.ndarray,
             pairs: List[Tuple[int, int]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """The Spearman correlation and the pairwise row count of every pair, from the rank sums."""
    p = len(cols)
    corr, n = np.eye(p), np.zeros((p, p))
    for (i, j), s in zip(pairs, sums):
        ci, cj = cond_of(cols[j], masks), cond_of(cols[i], masks)
        ti, tj = tables[cols[i]], tables[cols[j]]
        if ti['sizes'][ci] != tj['sizes'][cj]:
            log.warning('%s / %s: %s vs %s rows in the pair, expected equal', cols[i], cols[j],
                        ti['sizes'][ci], tj['sizes'][cj])
        den = np.sqrt(ti['sumsq'][ci] * tj['sumsq'][cj])
        corr[i, j] = corr[j, i] = s / den if den > 0 else np.nan
        n[i, j] = n[j, i] = ti['sizes'][ci]
    for i, c in enumerate(cols):
        n[i, i] = tables[c]['sizes']['all']
    return pd.DataFrame(corr, index=cols, columns=cols), pd.DataFrame(n, index=cols, columns=cols)

def corr_filter(corr: pd.DataFrame, ap: pd.Series, threshold: float) -> Tuple[List[str], pd.DataFrame]:
    """While any pair has |Spearman| above threshold, drop the feature with the highest max |corr|, then the
    highest median |corr|, then the lowest univariate AP. Returns the kept features and the rounds."""
    a = corr.abs().fillna(0).mask(np.eye(len(corr), dtype=bool))
    keep = list(corr.columns)
    steps: List[dict] = []
    while len(keep) > 1:
        sub = a.loc[keep, keep]
        mx = sub.max()
        if mx.max() <= threshold:
            break
        cand = mx.index[mx > threshold]
        rank = pd.DataFrame({'max_corr': mx[cand], 'median_corr': sub[cand].median(), 'ap': ap[cand]})
        worst = rank.sort_values(['max_corr', 'median_corr', 'ap'], ascending=[False, False, True]).index[0]
        partner = sub[worst].idxmax()
        steps.append({'round': len(steps) + 1, 'dropped': worst, 'max_corr': float(rank.loc[worst, 'max_corr']),
                      'with': partner, 'corr': float(corr.loc[worst, partner]),
                      'median_corr': float(rank.loc[worst, 'median_corr']),
                      'univariate_ap': float(ap[worst]), 'partner_univariate_ap': float(ap[partner]),
                      'features_left': len(keep) - 1})
        log.info('Corr round %s: drop %s (|corr| %.3f with %s, median %.3f, AP %.4f vs %.4f)', len(steps),
                 worst, rank.loc[worst, 'max_corr'], partner, rank.loc[worst, 'median_corr'], ap[worst], ap[partner])
        keep.remove(worst)
    return keep, pd.DataFrame(steps)

def write_doc(path: Path, result: dict, uni: pd.DataFrame) -> None:
    m = result['meta']
    steps = result['corr_steps']
    approx = [c for c, r in result['rank_groups'].items() if not r['exact']]
    preamble = (
        f"Correlation filter before the XGBoost elimination, on the **raw** features of the **train** split "
        f"({m['split']['dev_end']} and before, loans not in valid; {m['train_rows']:,} rows). Valid and test are "
        f"not used, except to score the univariate AP.\n\n"
        f"1. **Univariate AP** (tie breaker): each feature alone as a lookup scorer -- the train bad rate of "
        f"its category; of its value for a numeric feature with at most {DISCRETE_MAX:,} distinct values; "
        f"otherwise of its {m['univariate_bins']} equal-frequency train bins, with any value holding more than "
        f"1/{m['univariate_bins']} of the rows as a bin of its own; missing as a group of its own -- and AP over "
        f"every valid row.\n"
        f"2. **Spearman** between every pair of the {len(result['numeric'])} numeric features, on the train rows "
        f"where **both** are non-null, each ranked again within those rows (midranks for ties), as pandas or R "
        f"with pairwise-complete observations. "
        + (f"Columns with more than {m['max_exact_distinct']:,} distinct values are ranked on a "
           f"{m['rank_buckets']:,}-quantile grid of train (a value's rank is its bucket's midrank, so it moves by "
           f"at most ~1/{m['rank_buckets']:,} of the rows): {', '.join(f'`{c}`' for c in approx)}; every other "
           f"column on its exact values. " if approx else "Every column ranked on its exact values. ")
        + f"\n3. **Filter**: while any pair has |Spearman| > {m['corr_threshold']:g}, drop the feature with the "
        f"highest max |corr|, then the highest median |corr|, then the lowest univariate AP ({len(steps)} dropped).\n\n"
        f"The {len(result['categorical'])} string features have no Spearman correlation and all go forward; "
        f"so do numeric features with no correlation above the threshold.\n\n"
        f"**{len(result['candidates'])} candidates for the elimination**: {len(result['kept_numeric'])} numeric + "
        f"{len(result['categorical'])} categorical.")
    rounds = md_table(pd.DataFrame(steps), floatfmt='{:.4f}') if steps else '_(none: no pair was above the threshold)_'
    u = uni.reset_index().sort_values('univariate_ap_valid', ascending=False)
    u['kind'] = u['feature'].map(result['kinds'])
    u['status'] = u['feature'].map(lambda c: 'dropped (correlation)' if c in {s['dropped'] for s in steps}
                                   else 'candidate')
    u = u[['feature', 'kind', 'status', 'univariate_ap_valid', 'univariate_ap_train', 'groups',
           'missing_share_train', 'unseen_in_train_valid_rows']]
    write_report(path, 'XGBoost Feature Selection: Spearman Correlation Filter',
                 'src/jobs/12_gbm_corr_selection_main.py',
                 [('Correlation filter', rounds),
                  ('Candidates for the elimination', ', '.join(f'`{c}`' for c in result['candidates'])),
                  ('Univariate AP (valid), best first', md_table(u, floatfmt='{:.4f}'))], preamble)

def main() -> None:
    p = argparse.ArgumentParser(description='gbm_corr_selection_main')
    p.add_argument('--stage', choices=('all', 'univariate', 'corr', 'filter'), default='all',
                   help='univariate: AP per feature; corr: Spearman matrix on train; filter: the correlation filter')
    p.add_argument('--training-set', default=curated_path('training_set'))
    p.add_argument('--split-json', default=str(settings.model_dir / 'gbm_split.json'))
    p.add_argument('--exclude', default='', help='comma-separated columns to leave out')
    p.add_argument('--univariate-bins', type=int, default=UNIVARIATE_BINS)
    p.add_argument('--univariate-json', default=str(settings.model_dir / 'gbm_univariate.json'))
    p.add_argument('--corr-threshold', type=float, default=CORR_THRESHOLD)
    p.add_argument('--corr-csv', default=str(settings.model_dir / 'gbm_spearman_train.csv'))
    p.add_argument('--pairs-csv', default=str(settings.model_dir / 'gbm_spearman_pair_rows.csv'),
                   help='rows behind each pairwise correlation')
    p.add_argument('--selection-json', default=str(settings.model_dir / 'gbm_corr_selection.json'))
    p.add_argument('--doc', default=str(Path(__file__).resolve().parents[2] / 'docs' / 'gbm_corr_selection.md'))
    a = p.parse_args()

    cfg = load_split(Path(a.split_json))['config']
    exclude = [c.strip() for c in a.exclude.split(',') if c.strip()]
    uni_path, corr_path, pairs_path = Path(a.univariate_json), Path(a.corr_csv), Path(a.pairs_csv)
    run = {s: a.stage in ('all', s) for s in ('univariate', 'corr', 'filter')}

    spark = None
    if run['univariate'] or run['corr']:
        spark = spark_session('gbm_corr_selection', conf={'spark.sql.execution.arrow.pyspark.enabled': 'true',
                                                          'spark.driver.maxResultSize': '4g'})
    try:
        if run['univariate']:
            df = dev_rows(spark, a.training_set, cfg)
            kinds = feature_kinds(df, exclude)
            num = [c for c, k in kinds.items() if k == 'numerical']
            log.info('%s features (%s numeric, %s categorical)', len(kinds), len(num), len(kinds) - len(num))
            t = time.time()
            edges = numeric_edges(df.where(F.col(SPLIT) == 'train'), num, a.univariate_bins)
            log.info('Train quantiles in %.0fs', time.time() - t)
            t = time.time()
            counts = group_counts(df.where(F.col(SPLIT).isin('train', 'valid')), kinds, edges)
            log.info('Group counts in %.0fs', time.time() - t)
            uni = univariate_ap(counts)
            for c, r in uni.sort_values('univariate_ap_valid', ascending=False).iterrows():
                log.info('  %-55s AP valid %.4f, train %.4f, %s groups', c, r['univariate_ap_valid'],
                         r['univariate_ap_train'], r['groups'])
            uni_path.parent.mkdir(parents=True, exist_ok=True)
            uni_path.write_text(json.dumps({
                'meta': {'split': cfg, 'bins': a.univariate_bins, 'edge_sample': EDGE_SAMPLE,
                         'scorer': 'train bad rate of the feature group; AP over valid rows',
                         'generated': datetime.now().isoformat(timespec='seconds'),
                         'source': 'src/jobs/12_gbm_corr_selection_main.py'},
                'kinds': kinds, 'categories': categories(counts, kinds), 'edges': edges,
                'features': uni.reset_index().to_dict('records')}, indent=2, default=float))
            log.info('Univariate AP in %s', uni_path)

        if not uni_path.exists():
            raise SystemExit(f'No univariate AP at {uni_path}: run --stage univariate first.')
        uj = json.loads(uni_path.read_text())
        kinds = {c: k for c, k in uj['kinds'].items() if c not in exclude}
        uni = pd.DataFrame(uj['features']).set_index('feature').loc[list(kinds)]
        num = [c for c, k in kinds.items() if k == 'numerical']

        rank_path = corr_path.with_suffix('.meta.json')
        if run['corr']:
            train = dev_rows(spark, a.training_set, cfg).where(F.col(SPLIT) == 'train').select(*num)
            t = time.time()
            r = train.agg(F.count(F.lit(1)).alias('_n'),
                          *[F.approx_count_distinct(c, 0.02).alias(c) for c in num]).collect()[0].asDict()
            n_rows = int(r.pop('_n'))
            masks = null_masks(train, num)
            log.info('%s train rows; null patterns: %s (%.0fs)', f'{n_rows:,}',
                     {m: [c for c in masks if masks[c] == m] for m in dict.fromkeys(masks.values())} or 'none',
                     time.time() - t)
            tables = rank_tables(train, num, masks, r)
            t = time.time()
            sums, pairs = cross_sums(train, num, masks, tables)
            log.info('Cross sums for %s pairs in %.0fs', len(pairs), time.time() - t)
            corr, n = spearman(num, masks, tables, sums, pairs)
            corr_path.parent.mkdir(parents=True, exist_ok=True)
            corr.to_csv(corr_path)
            n.astype('int64').to_csv(pairs_path)
            rank_path.write_text(json.dumps({
                'train_rows': n_rows, 'masks': masks,
                'rank_groups': {c: {'groups': tables[c]['groups'], 'exact': tables[c]['exact'],
                                    'approx_distinct': r[c]} for c in num}}, indent=2))
            log.info('Spearman matrix in %s', corr_path)
    finally:
        if spark is not None:
            spark.stop()

    if run['filter']:
        if not corr_path.exists():
            raise SystemExit(f'No correlation matrix at {corr_path}: run --stage corr first.')
        corr = pd.read_csv(corr_path, index_col=0)
        rank_meta = json.loads(rank_path.read_text())
        cols = [c for c in corr.columns if c in num]
        missing = sorted(set(num) - set(cols))
        if missing:
            log.warning('Not in the saved matrix, left out of the filter: %s', ', '.join(missing))
        ap = uni['univariate_ap_valid']
        kept, steps = corr_filter(corr.loc[cols, cols], ap, a.corr_threshold)
        cat = [c for c, k in kinds.items() if k == 'categorical']
        candidates = kept + cat
        log.info('Correlation filter: %s of %s numeric features kept; %s candidates with the %s categorical',
                 len(kept), len(cols), len(candidates), len(cat))
        result = {
            'meta': {'split': cfg, 'train_rows': rank_meta['train_rows'], 'corr_threshold': a.corr_threshold,
                     'univariate_bins': uj['meta']['bins'], 'max_exact_distinct': MAX_EXACT_DISTINCT,
                     'rank_buckets': RANK_BUCKETS, 'method': 'Spearman, pairwise non-null train rows',
                     'corr_rule': 'highest max |corr|, then highest median |corr|, then lowest univariate AP',
                     'corr_csv': str(corr_path), 'univariate_json': str(uni_path),
                     'generated': datetime.now().isoformat(timespec='seconds'),
                     'source': 'src/jobs/12_gbm_corr_selection_main.py'},
            'kinds': kinds, 'numeric': cols, 'categorical': cat, 'kept_numeric': kept,
            'candidates': candidates, 'masks': rank_meta['masks'], 'rank_groups': rank_meta['rank_groups'],
            'corr_steps': steps.to_dict('records'),
        }
        sel = Path(a.selection_json)
        sel.parent.mkdir(parents=True, exist_ok=True)
        sel.write_text(json.dumps(result, indent=2, default=float))
        write_doc(Path(a.doc), result, uni)
        log.info('Selection in %s, report in %s', sel, a.doc)

if __name__ == '__main__':
    main()