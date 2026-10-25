import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
from pyspark.sql import functions as F
from helpers.gbm import DEV_END, SPLIT, SPLIT_JSON, SPLIT_SALT, TARGET, VALID_SHARE, with_split, ym
from helpers.report import md_table, write_report
from helpers.spark_io import curated_path, resolve_path, spark_session

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s | %(message)s')
log = logging.getLogger('gbm_split_main')

def census(df) -> pd.DataFrame:
    """Rows and bads per month and split."""
    return (df.groupBy(ym().alias('ym'), SPLIT)
            .agg(F.count(F.lit(1)).alias('rows'), F.sum(TARGET).alias('bads'))
            .toPandas())

def loans(df) -> pd.DataFrame:
    """The approximate number of distinct loans per split."""
    return (df.groupBy(SPLIT).agg(F.approx_count_distinct('loan_identifier', 0.01).alias('loans'))
            .toPandas().set_index(SPLIT))

def tables(c: pd.DataFrame):
    """The monthly train / valid comparison (rows, shares of rows and bads, bad rates) over the development months,
    and the totals per split."""
    c = c.copy()
    c['month'] = c['ym'].map(lambda v: f'{v // 100}-{v % 100:02d}')
    rows = c.pivot_table(index='month', columns=SPLIT, values='rows', aggfunc='sum', fill_value=0)
    bads = c.pivot_table(index='month', columns=SPLIT, values='bads', aggfunc='sum', fill_value=0)
    dev = rows.index[rows.get('test', 0) == 0]
    m = pd.DataFrame({
        'train rows': rows.loc[dev, 'train'], 'valid rows': rows.loc[dev, 'valid'],
        'valid share of rows': rows.loc[dev, 'valid'] / rows.loc[dev, ['train', 'valid']].sum(axis=1),
        'valid share of bads': bads.loc[dev, 'valid'] / bads.loc[dev, ['train', 'valid']].sum(axis=1),
        'train bad rate': bads.loc[dev, 'train'] / rows.loc[dev, 'train'],
        'valid bad rate': bads.loc[dev, 'valid'] / rows.loc[dev, 'valid'],
    })
    m['valid / train bad rate'] = m['valid bad rate'] / m['train bad rate']
    tot = c.groupby(SPLIT)[['rows', 'bads']].sum()
    tot['bad rate'] = tot['bads'] / tot['rows']
    tot['share of rows'] = tot['rows'] / tot['rows'].sum()
    return m, tot

def write_doc(path: Path, result: dict, monthly: pd.DataFrame, total: pd.DataFrame) -> None:
    cfg = result['config']
    s = result['summary']
    preamble = (f"Train / valid / test split for the XGBoost model. **Test** is every month after "
                f"{cfg['dev_end']}, the logistic regression's test months. Inside the train window "
                f"({cfg['first_month']} .. {cfg['dev_end']}) each **loan** goes wholly to valid with probability "
                f"{cfg['valid_share']:.0%}, from a seeded hash of `loan_identifier` (salt `{cfg['salt']}`), "
                f"and to train otherwise, so no loan has months on both sides and the valid AP is not lifted by "
                f"the same loan's neighbouring months in train.\n\n"
                f"Loans are not stratified explicitly; with ~{s['dev_loans']:,.0f} loans the hash gives every "
                f"month an 80/20 split and the same bad rate up to sampling noise. The realised spread is below: "
                f"valid share of rows {s['valid_row_share_min']:.2%} .. {s['valid_row_share_max']:.2%} by month, "
                f"valid / train bad rate {s['bad_rate_ratio_min']:.3f} .. {s['bad_rate_ratio_max']:.3f}.\n\n"
                f"Nothing is written: later jobs rebuild the split from `{SPLIT_JSON.name}` "
                f"(`helpers.gbm.split_column`).")
    tot = total.reset_index().rename(columns={SPLIT: 'split'})
    tot['loans (approx.)'] = tot['split'].map(result['loans']).round().astype('Int64')
    monthly = monthly.reset_index()
    write_report(path, 'XGBoost: Train / Valid / Test Split', 'src/jobs/11_gbm_split_main.py',
                 [('Totals', md_table(tot, floatfmt='{:.4f}')),
                  ('By month (train window)', md_table(monthly, floatfmt='{:.4f}'))], preamble)

def main() -> None:
    p = argparse.ArgumentParser(description='gbm_split_main')
    p.add_argument('--training-set', default=curated_path('training_set'))
    p.add_argument('--dev-end', default=DEV_END, help='last train/valid month; later months are test')
    p.add_argument('--valid-share', type=float, default=VALID_SHARE, help='share of loans that go to valid')
    p.add_argument('--salt', default=SPLIT_SALT)
    p.add_argument('--split-json', default=str(SPLIT_JSON))
    p.add_argument('--doc', default=str(Path(__file__).resolve().parents[2] / 'docs' / 'gbm_split.md'))
    a = p.parse_args()
    if not 0 < a.valid_share < 1:
        raise SystemExit('--valid-share must be in (0, 1).')

    end = pd.Timestamp(a.dev_end)
    cfg = {'dev_end': f'{end:%Y-%m}', 'end_ym': end.year * 100 + end.month,
           'valid_share': a.valid_share, 'salt': a.salt, 'unit': 'loan_identifier',
           'training_set': a.training_set}

    spark = spark_session('gbm_split')
    try:
        t = time.time()
        df = with_split(spark.read.parquet(resolve_path(a.training_set)), cfg)
        c = census(df)
        n_loans = loans(df.where(F.col(SPLIT) != 'test'))
        log.info('Census in %.0fs', time.time() - t)
    finally:
        spark.stop()

    monthly, total = tables(c)
    first = monthly.index.min()
    cfg['first_month'] = first
    summary = {
        'dev_loans': float(n_loans['loans'].sum()),
        'valid_row_share_min': float(monthly['valid share of rows'].min()),
        'valid_row_share_max': float(monthly['valid share of rows'].max()),
        'valid_bad_share_min': float(monthly['valid share of bads'].min()),
        'valid_bad_share_max': float(monthly['valid share of bads'].max()),
        'bad_rate_ratio_min': float(monthly['valid / train bad rate'].min()),
        'bad_rate_ratio_max': float(monthly['valid / train bad rate'].max()),
    }
    for s, r in total.iterrows():
        log.info('%-5s %s rows, %s bads, bad rate %.3f%%', s, f"{int(r['rows']):,}", f"{int(r['bads']):,}",
                 100 * r['bad rate'])
    log.info('Valid share of rows by month %.2f%% .. %.2f%%; valid / train bad rate %.3f .. %.3f',
             100 * summary['valid_row_share_min'], 100 * summary['valid_row_share_max'],
             summary['bad_rate_ratio_min'], summary['bad_rate_ratio_max'])

    result = {'config': cfg,
              'totals': {s: {k: float(v) for k, v in r.items()} for s, r in total.iterrows()},
              'loans': {s: float(v) for s, v in n_loans['loans'].items()},
              'summary': summary,
              'generated': datetime.now().isoformat(timespec='seconds'),
              'source': 'src/jobs/11_gbm_split_main.py'}
    out = Path(a.split_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    write_doc(Path(a.doc), result, monthly, total)
    log.info('Split in %s, report in %s', out, a.doc)

if __name__ == '__main__':
    main()