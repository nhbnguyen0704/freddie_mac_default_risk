import json
import tempfile
from pathlib import Path
import mlflow
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, functions as F
from scipy.special import expit
from io_helpers.read import ROOT, config, local_path, read_json, tracking
from logic.aggregation import LEVELS, shares
from logic.transformation import TwoSegmentModel
from spark import spark_session

DECILES = [i / 10 for i in range(1, 10)]
DESCRIPTION = ('Two-segment XGBoost merge (clean_regional on clean rows, recency_12 on the rest), Platt per segment '
               'on valid, F1-optimal cut chosen on valid. Known limitation: training rows exclude loans that later '
               'prepay without going bad, and deployment scores them. On 2023-01 with realised 25-month outcomes, '
               'AP is 0.0961 and precision at the cut 0.174 on every scored row, against 0.2168 and 0.365 on the '
               'training rows; expect a flag list about 2.6 times as long. See PLAN.md, Known limitation.')


def bundle(art: str) -> dict:
    """Features, best iterations and Platt maps of both segments, the train categories and the cut."""
    platt = read_json(f'{art}/segment_merge.json')['merge']['platt_valid']
    clean = read_json(f'{art}/regional_models/fitted.json')['clean_regional']
    delinquent = read_json(f'{art}/gbm_tuning_models/recency_12.json')
    segments = {'clean': {'features': clean['features'], 'best_iteration': clean['best_iteration'],
                          'platt': platt['clean']},
                'delinquent': {'features': read_json(f'{art}/gbm_tuning.json')['meta']['features'],
                               'best_iteration': delinquent['best_iteration'], 'platt': platt['delinquent']}}
    cats = read_json(f'{art}/gbm_univariate.json')['categories']
    return {'segments': segments, 'categories': {c: cats[c] for c in clean['features'] if c in cats},
            'threshold': read_json(f'{art}/merge_threshold.json')['threshold']}

def score_baseline(art: str, segments: dict) -> dict:
    """Per segment, PD decile edges and bin shares on job 22's valid rows."""
    t = pd.read_parquet(local_path(f'{art}/segment_merge_tables.parquet'))
    out = {}
    for seg, g in t[t['split'] == 'valid'].groupby('segment'):
        a, b = segments[seg]['platt']
        p, n = expit(a + b * g['m'].to_numpy()), g['n'].to_numpy(float)
        order = np.argsort(p)
        cum = np.cumsum(n[order]) / n.sum()
        edges = sorted({float(p[order][np.searchsorted(cum, q)]) for q in DECILES})
        bins = pd.Series(n).groupby(np.searchsorted(edges, p, side='left')).sum()
        out[seg] = {'edges': edges, 'shares': {str(k): v / n.sum() for k, v in bins.items()}}
    return out

def train_rows(spark, cfg: dict, art: str) -> DataFrame:
    """The train split of the training set, recomputed from gbm_split.json as development does, with the regional
    features joined as 20_regional_features_main.with_regional."""
    split = read_json(f'{art}/gbm_split.json')['config']
    u = F.pmod(F.xxhash64(F.col('loan_identifier'), F.lit(split['salt'])), F.lit(1 << 31)) / F.lit(float(1 << 31))
    df = (spark.read.parquet(cfg['training_set'])
          .where((F.col('year') * 100 + F.col('month') <= int(split['end_ym'])) & (u >= float(split['valid_share']))))
    for lv, col in LEVELS.items():
        df = df.join(F.broadcast(spark.read.parquet(f"{cfg['regional_features']}/level={lv}")), [col, 'period'], 'left')
    return df

def feature_baseline(cfg: dict, art: str, b: dict) -> dict:
    """Decile edges (numerics) or train categories (strings) and bin shares of every model feature on the train split."""
    feats = b['segments']['clean']['features']
    spark = spark_session('register_baseline', cfg['spark'])
    df = train_rows(spark, cfg, art)
    numeric = [c for c in feats if c not in b['categories']]
    edges = df.approxQuantile(numeric, DECILES, 1e-3)
    specs = {c: {'edges': sorted(set(edges[numeric.index(c)]))} if c in numeric
             else {'categories': b['categories'][c]} for c in feats}
    for c, s in shares(df, specs).items():
        specs[c]['shares'] = s
    spark.stop()
    return specs

def metrics(art: str) -> dict:
    merge = read_json(f'{art}/segment_merge.json')['merge']
    thr = read_json(f'{art}/merge_threshold.json')
    out = {f'merged_ap_{sp}': merge[sp]['platt_valid'] for sp in ('valid', 'test')}
    out |= {f'segment_ap_{k.replace("/", "_")}': v for k, v in merge['segment_ap'].items()}
    out |= {f'{k}_{sp}': thr[sp][k] for sp in ('valid', 'test') for k in ('f1', 'precision', 'recall', 'flagged_share')}
    return out

def main() -> None:
    cfg = config()
    art = cfg['development_artifacts']
    b = bundle(art)
    b['baseline'] = {'scores': score_baseline(art, b['segments']), 'features': feature_baseline(cfg, art, b)}
    params = read_json(f'{art}/gbm_tuning_models/recency_12.json')['params']
    tracking(cfg)
    with tempfile.TemporaryDirectory() as tmp, mlflow.start_run(run_name='register_model'):
        path = Path(tmp) / 'bundle.json'
        path.write_text(json.dumps(b, indent=2))
        mlflow.log_params(params | {f'best_iteration_{s}': v['best_iteration'] for s, v in b['segments'].items()}
                          | {'threshold': b['threshold'], 'dev_end': '2022-09'})
        mlflow.log_metrics(metrics(art))
        info = mlflow.pyfunc.log_model(
            name='model', python_model=TwoSegmentModel(), code_paths=[str(ROOT / 'logic')],
            artifacts={'bundle': str(path),
                       'clean': str(local_path(f'{art}/regional_models/clean_regional.ubj')),
                       'delinquent': str(local_path(f'{art}/gbm_tuning_models/recency_12.ubj')),
                       'label_spec': str(local_path(cfg['label_spec']))},
            registered_model_name=cfg['mlflow']['model_name'])
    client = mlflow.MlflowClient()
    client.update_model_version(cfg['mlflow']['model_name'], info.registered_model_version, DESCRIPTION)
    client.set_registered_model_alias(cfg['mlflow']['model_name'], cfg['mlflow']['alias'],
                                      info.registered_model_version)


if __name__ == '__main__':
    main()
