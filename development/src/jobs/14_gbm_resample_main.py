import argparse
import json
import logging
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, functions as F
from sklearn.neighbors import NearestNeighbors
from helpers.config import settings
from helpers import hdfs
from helpers.gbm import SPLIT, TARGET, load_split, row_id, sample_meta_path, uniform, with_split
from helpers.period import month_partitions
from helpers.report import md_table, write_report
from helpers.spark_io import curated_path, spark_session

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s | %(message)s')
log = logging.getLogger('gbm_resample_main')

SMOTE_RATIO = 0.1
SMOTE_K = 5
ENN_NEIGHBORS = 3
CHUNKS_PER_MONTH = 10
SYNTHETIC_KEEP = 0.1
DISCRETE_MAX = 1_000
RANDOM_STATE = 42
VERSIONS = ('smoteenn', 'smotetomek')
READ_ATTEMPTS = 3

class Space:
    """The distance space for SMOTE and the cleaners: numerics with missing values imputed by the weighted median
    and standardised, plus a 0/1 missing flag for each column that has missing values."""
    def __init__(self, num: List[str], med: np.ndarray, mean: np.ndarray, sd: np.ndarray, flags: List[str]):
        self.num, self.med, self.mean, self.sd, self.flags = num, med, mean, sd, flags
        self.flag_idx = np.array([num.index(c) for c in flags], dtype=int)

    @classmethod
    def fit(cls, sample: pd.DataFrame, num: List[str], w: np.ndarray) -> 'Space':
        """Weighted medians, means and standard deviations of the numerics on the sample."""
        x = sample[num].to_numpy(np.float64)
        med, mean, sd = np.empty(len(num)), np.empty(len(num)), np.empty(len(num))
        for j in range(len(num)):
            ok = ~np.isnan(x[:, j])
            v, ww = x[ok, j], w[ok]
            med[j] = np.quantile(v, 0.5, weights=ww, method='inverted_cdf')
            z = np.where(ok, x[:, j], med[j])
            mean[j] = np.average(z, weights=w)
            sd[j] = np.sqrt(np.average((z - mean[j]) ** 2, weights=w)) or 1
        flags = [c for j, c in enumerate(num) if np.isnan(x[:, j]).any()]
        return cls(num, med, mean, sd, flags)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Rows mapped into the space."""
        miss = np.isnan(x)
        z = (np.where(miss, self.med, x) - self.mean) / self.sd
        return np.hstack([z, miss[:, self.flag_idx]]).astype(np.float32)

    def to_dict(self) -> dict:
        return {'numeric': self.num, 'median': self.med.tolist(), 'mean': self.mean.tolist(), 'sd': self.sd.tolist(),
                'missing_flags': self.flags}

def without_self(ind: np.ndarray, own: np.ndarray, k: int) -> np.ndarray:
    """The first k neighbours of each row with the row itself removed."""
    order = np.argsort(ind == own[:, None], axis=1, kind='stable')
    return np.take_along_axis(ind, order, axis=1)[:, :k]

def most_frequent(v: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The most frequent value per row, ties broken at random."""
    count = (v[:, :, None] == v[:, None, :]).sum(axis=2)
    best = np.argmax(count + rng.random(v.shape) * 0.5, axis=1)
    return v[np.arange(len(v)), best]

def snap(new: np.ndarray, values: List[np.ndarray]) -> np.ndarray:
    """Each non-missing synthetic value moved to the nearest of its column's discrete values (columns without a
    value list are left as they are)."""
    out = new.copy()
    for j, vals in enumerate(values):
        if vals is None or len(vals) == 0:
            continue
        col = new[:, j]
        ok = ~np.isnan(col)
        i = np.clip(np.searchsorted(vals, col[ok]), 1, max(len(vals) - 1, 1))
        lo, hi = vals[i - 1], vals[np.minimum(i, len(vals) - 1)]
        out[ok, j] = np.where(col[ok] - lo <= hi - col[ok], lo, hi)
    return out

def read_month(spark, uri: str, y: int, m: int, cfg: dict, feats: List[str], num: List[str], goods_fraction: float,
               chunks: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """One month's train rows (row id, label, whether in the goods design, chunk and numerics) and its bads with
    every feature, both sorted by row id."""
    df = with_split(spark.read.parquet(uri).withColumn('year', F.lit(y)).withColumn('month', F.lit(m)), cfg)
    df = df.where(F.col(SPLIT) == 'train')
    rows = df.select(row_id().alias('rid'), F.col(TARGET).cast('int').alias(TARGET),
                     (uniform(F.col('loan_identifier'), F.col('period'), salt='gbm_goods')
                      < F.lit(goods_fraction)).alias('design'),
                     F.floor(uniform(F.col('loan_identifier'), F.col('period'), salt='smote_chunk')
                             * F.lit(chunks)).cast('int').alias('chunk'),
                     *[F.col(c).cast('float').alias(c) for c in num]).toPandas()
    bads = df.where(F.col(TARGET) == 1).select(
        row_id().alias('rid'), *[F.col(c).cast('double').alias(c) if c in num else F.col(c) for c in feats]
    ).toPandas()
    return (rows.sort_values('rid', kind='mergesort').reset_index(drop=True),
            bads.sort_values('rid', kind='mergesort').reset_index(drop=True))

def smote(bads: pd.DataFrame, space: Space, num: List[str], cat: List[str], n_new: int, discrete: List[np.ndarray],
          rng: np.random.Generator, k: int) -> pd.DataFrame:
    """n_new synthetic bads interpolated between a bad and one of its k nearest bads in the space: numerics on the
    line between them (kept from the base where either is missing, then snapped), strings the most frequent
    among the base's neighbours."""
    x = bads[num].to_numpy(np.float64)
    z = space.transform(x)
    k = min(k, len(bads) - 1)
    ind = NearestNeighbors(n_neighbors=k + 1, algorithm='brute', n_jobs=-1).fit(z).kneighbors(
        z, return_distance=False
    )
    nbr = without_self(ind, np.arange(len(bads)), k)
    base = rng.integers(0, len(bads), n_new)
    pick = nbr[base, rng.integers(0, k, n_new)]
    lam = rng.random(n_new)[:, None]
    xb, xn = x[base], x[pick]
    new = np.where(np.isnan(xb) | np.isnan(xn), xb, xb + lam * (xn - xb))
    out = pd.DataFrame(snap(new, discrete), columns=num)
    for c in cat:
        codes = pd.Categorical(bads[c])
        c_nbr = codes.codes[nbr[base]]
        out[c] = pd.Categorical.from_codes(most_frequent(c_nbr, rng), codes.categories)
    return out

def clean_month(rows: pd.DataFrame, syn_z: np.ndarray, syn_chunk: np.ndarray, space: Space, num: List[str],
                chunks: int) -> Dict[str, np.ndarray]:
    """Within each chunk, the ids of the design goods ENN removes (any of its ENN_NEIGHBORS neighbours is bad) and
    those Tomek links remove (its nearest neighbour is a bad whose nearest neighbour is that good)."""
    removed = {v: [] for v in VERSIONS}
    z_all = space.transform(rows[num].to_numpy(np.float64))
    y_all, design, rid = rows[TARGET].to_numpy(), rows['design'].to_numpy(), rows['rid'].to_numpy()
    for c in range(chunks):
        real = np.flatnonzero(rows['chunk'].to_numpy() == c)
        zs = syn_z[syn_chunk == c]
        ref = np.vstack([z_all[real], zs])
        y = np.r_[y_all[real], np.ones(len(zs), dtype=y_all.dtype)]
        goods = np.flatnonzero(np.r_[design[real] & (y_all[real] == 0), np.zeros(len(zs), bool)])
        bads = np.flatnonzero(y == 1)
        if not len(goods) or not len(bads):
            continue
        q = np.r_[goods, bads]
        ind = NearestNeighbors(n_neighbors=ENN_NEIGHBORS + 1, algorithm='brute', n_jobs=-1).fit(ref).kneighbors(
            ref[q], return_distance=False
        )
        nb = without_self(ind, q, ENN_NEIGHBORS)
        g_nb = nb[:len(goods)]
        enn = (y[g_nb] == 1).any(axis=1)
        nearest = np.full(len(ref), -1)
        nearest[q] = nb[:, 0]
        first = g_nb[:, 0]
        tomek = (y[first] == 1) & (nearest[first] == goods)
        removed['smoteenn'].append(rid[real[goods[enn]]])
        removed['smotetomek'].append(rid[real[goods[tomek]]])
    return {v: np.concatenate(r) if r else np.array([], dtype=np.int64) for v, r in removed.items()}

def write_doc(path: Path, meta: dict, stats: pd.DataFrame) -> None:
    c = meta['config']
    tot = stats.sum(numeric_only=True)
    pre = (f"SMOTEENN and SMOTETomek on **every row of the train split** ({len(stats)} months, "
           f"{int(tot['rows']):,} rows), for versions 9 and 10 of `15_gbm_tuning_main`, as `10_lr_model_main` does "
           f"for the logistic regression but on the {len(c['features'])} raw features.\n\n"
           f"- **SMOTE**: per month on all of the month's train bads, k={c['smote_k']}, up to {c['smote_ratio']:g} bads "
           f"per good: {int(tot['synthetic']):,} synthetic bads in all.\n"
           f"- **ENN** (n_neighbors={c['enn_neighbors']}, kind_sel='all') and **Tomek links**, goods only, on the real "
           f"and synthetic rows of {c['chunks_per_month']} seeded random chunks per month.\n"
           f"- **Distance space**: the {len(meta['space']['numeric'])} numeric features median-imputed and standardised "
           f"(weighted train statistics), plus a missing flag for each of the {len(meta['space']['missing_flags'])} "
           f"nullable ones; strings are not in the distance. A synthetic numeric is interpolated where both parents "
           f"have it, else copied from the base bad; columns with at most {c['discrete_max']:,} values are snapped to "
           f"them; a string takes the most frequent value among the base bad's neighbours.\n"
           f"- The XGBoost fit uses every bad and {c['goods_fraction']:.0%} of goods, so the cleaners were evaluated "
           f"for those goods ({int(tot['design_goods']):,}) -- each removed exactly as if every row had been cleaned -- "
           f"and {c['synthetic_keep']:.0%} of the synthetic bads were kept ({int(tot['synthetic_kept']):,}, weighted "
           f"{1 / c['synthetic_keep']:g}).\n\n"
           f"Removed sampled goods: ENN {int(tot['removed_smoteenn']):,} ({tot['removed_smoteenn'] / tot['design_goods']:.2%}), "
           f"Tomek {int(tot['removed_smotetomek']):,} ({tot['removed_smotetomek'] / tot['design_goods']:.2%}).")
    s = stats.copy()
    s['ENN removed %'] = s['removed_smoteenn'] / s['design_goods']
    s['Tomek removed %'] = s['removed_smotetomek'] / s['design_goods']
    write_report(path, 'XGBoost: SMOTEENN / SMOTETomek on the Train Split', 'src/jobs/14_gbm_resample_main.py',
                 [('By month', md_table(s.reset_index(), floatfmt='{:.4f}'))], pre)

def main():
    p = argparse.ArgumentParser(description='gbm_resample_main')
    p.add_argument('--training-set', default=curated_path('training_set'))
    p.add_argument('--split-json', default=str(settings.model_dir / 'gbm_split.json'))
    p.add_argument('--elimination-json', default=str(settings.model_dir / 'gbm_elimination.json'),
                   help='the suggested features')
    p.add_argument('--selection-json', default=str(settings.model_dir / 'gbm_corr_selection.json'))
    p.add_argument('--elimination-sample', default=curated_path('gbm_elimination_sample'),
                   help='its train rows give the distance-space statistics')
    p.add_argument('--out', default=curated_path('gbm_resampled'), help='on HDFS')
    p.add_argument('--doc', default=str(Path(__file__).resolve().parents[2] / 'docs' / 'gbm_resample.md'))
    p.add_argument('--chunks-per-month', type=int, default=CHUNKS_PER_MONTH)
    p.add_argument('--resume', action='store_true', help='skip months already in --out')
    a = p.parse_args()

    cfg = load_split(Path(a.split_json))['config']
    feats = json.loads(Path(a.elimination_json).read_text())['suggested_features']
    kinds = json.loads(Path(a.selection_json).read_text())['kinds']
    num = [c for c in feats if kinds[c] == 'numerical']
    cat = [c for c in feats if kinds[c] == 'categorical']
    smeta = json.loads(sample_meta_path(a.elimination_sample).read_text())
    gf = float(smeta['goods_fraction'])
    out = a.out.rstrip('/')

    with hdfs.local_copy(a.elimination_sample) as local:
        sample = pd.read_parquet(local, columns=[SPLIT, 'sample_weight', *num])
    sample = sample[sample[SPLIT] == 'train']
    space = Space.fit(sample, num, sample['sample_weight'].to_numpy(np.float64))
    del sample
    config = {'smote_ratio': SMOTE_RATIO, 'smote_k': SMOTE_K, 'enn_neighbors': ENN_NEIGHBORS, 'enn_kind': 'all',
              'clean_strategy': 'majority', 'chunks_per_month': a.chunks_per_month, 'synthetic_keep': SYNTHETIC_KEEP,
              'discrete_max': DISCRETE_MAX, 'random_state': RANDOM_STATE, 'goods_fraction': gf, 'features': feats}
    log.info('%s features (%s numeric, %s strings); distance space %s dims', len(feats), len(num), len(cat),
             len(num) + len(space.flags))

    spark = spark_session('gbm_resample', conf={'spark.sql.execution.arrow.pyspark.enabled': 'true',
                                                'spark.driver.maxResultSize': '4g'})
    stage = Path(tempfile.mkdtemp(prefix='gbm_resampled_'))
    try:
        parts = [(y, m, uri) for y, m, uri in month_partitions(spark, a.training_set)
                 if y * 100 + m <= int(cfg['end_ym'])]
        done = {x.rsplit('/', 2)[-2] for x in hdfs.glob(f'{out}/month=*/stats.json')} if a.resume else set()
        t0 = time.time()
        for i, (y, m, uri) in enumerate(parts, 1):
            ym = y * 100 + m
            if f'month={ym}' in done:
                continue
            d = stage / f'month={ym}'
            d.mkdir()
            t = time.time()
            for attempt in range(1, READ_ATTEMPTS + 1):
                try:
                    rows, bads = read_month(spark, uri, y, m, cfg, feats, num, gf, a.chunks_per_month)
                    break
                except Exception as exc:
                    if attempt == READ_ATTEMPTS:
                        raise
                    log.warning('%s-%02d: read failed (%s), retry %s of %s', y, m,
                                str(exc).splitlines()[0][:120], attempt, READ_ATTEMPTS - 1)
                    time.sleep(30)
            n_bad = int((rows[TARGET] == 1).sum())
            n_good = len(rows) - n_bad
            rng = np.random.default_rng(RANDOM_STATE + ym)
            n_new = max(int(np.ceil(SMOTE_RATIO * n_good)) - n_bad, 0) if n_bad >= 2 else 0
            discrete = []
            for c in num:
                v = np.unique(rows[c].to_numpy(np.float64))
                v = v[~np.isnan(v)]
                discrete.append(v if len(v) <= DISCRETE_MAX else None)
            syn = smote(bads, space, num, cat, n_new, discrete, rng, SMOTE_K) if n_new else bads.iloc[:0][feats]
            syn_z = space.transform(syn[num].to_numpy(np.float64))
            syn_chunk = rng.integers(0, a.chunks_per_month, len(syn))
            t_smote = time.time() - t
            removed = clean_month(rows, syn_z, syn_chunk, space, num, a.chunks_per_month)
            keep = rng.random(len(syn)) < SYNTHETIC_KEEP
            kept = syn[keep].reset_index(drop=True)
            kept.insert(0, 'ym', np.int32(ym))
            kept.insert(1, TARGET, np.int8(1))
            kept.insert(2, 'sample_weight', np.float32(1 / SYNTHETIC_KEEP))
            kept.to_parquet(d / 'synthetic.parquet', index=False)
            pd.concat([pd.DataFrame({'rid': r, 'version': v}) for v, r in removed.items()],
                      ignore_index=True).to_parquet(d / 'removed.parquet', index=False)
            st = {'ym': ym, 'rows': len(rows), 'bads': n_bad, 'goods': n_good, 'synthetic': len(syn),
                  'synthetic_kept': int(keep.sum()), 'design_goods': int((rows['design'] & (rows[TARGET] == 0)).sum()),
                  'removed_smoteenn': len(removed['smoteenn']), 'removed_smotetomek': len(removed['smotetomek']),
                  'seconds': round(time.time() - t)}
            (d / 'stats.json').write_text(json.dumps(st))
            hdfs.put([d / 'synthetic.parquet', d / 'removed.parquet'], f'{out}/{d.name}')
            hdfs.put([d / 'stats.json'], f'{out}/{d.name}')
            shutil.rmtree(d)
            log.info('[%s/%s] %s-%02d: %s rows, %s bads + %s synthetic (%s kept); of %s sampled goods ENN removes %s, '
                     'Tomek %s; SMOTE %.0fs, total %.0fs', i, len(parts), y, m, f'{len(rows):,}', f'{n_bad:,}',
                     f'{len(syn):,}', f"{st['synthetic_kept']:,}", f"{st['design_goods']:,}",
                     f"{st['removed_smoteenn']:,}", f"{st['removed_smotetomek']:,}", t_smote, time.time() - t)
            del rows, bads, syn, syn_z
        log.info('All months in %.0f min', (time.time() - t0) / 60)
    finally:
        spark.stop()
        shutil.rmtree(stage, ignore_errors=True)

    stats = pd.DataFrame(hdfs.read_jsons(f'{out}/month=*/stats.json')).set_index('ym').sort_index()
    meta = {'config': config, 'space': space.to_dict(), 'months': len(stats),
            'totals': {k: float(v) for k, v in stats.sum(numeric_only=True).items()},
            'generated': datetime.now().isoformat(timespec='seconds'), 'source': 'src/jobs/14_gbm_resample_main.py'}
    hdfs.write_text(f'{out}/meta.json', json.dumps(meta, indent=2))
    write_doc(Path(a.doc), meta, stats)
    log.info('Resampled rows in %s, report in %s', out, a.doc)

if __name__ == '__main__':
    main()