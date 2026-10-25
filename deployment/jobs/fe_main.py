import argparse
from pyspark.sql import functions as F
from io_helpers.read import config, month_index, months, partition_index
from io_helpers.write import write_month
from logic.aggregation import LEVELS, loan_features
from logic.preprocessing import observations
from logic.transformation import interactions
from spark import spark_session


def main() -> None:
    p = argparse.ArgumentParser(description='fe_main')
    p.add_argument('--asof', required=True, help='YYYY-MM')
    a = p.parse_args()
    cfg = config()
    t = month_index(a.asof)
    spark = spark_session('fe', cfg['spark'])
    perf = months(spark, cfg['performance'], t - cfg['history_months'] + 1, t)
    obs = observations(perf, spark.read.parquet(cfg['origination']), t)
    windows = loan_features(perf).where((F.year('period') * 12 + F.month('period')) == t)
    feat = interactions(obs.join(windows, ['loan_identifier', 'period'], 'left'))
    for lv, col in LEVELS.items():
        r = spark.read.parquet(f"{cfg['output_root']}/regional_monthly/level={lv}").where(partition_index() == t)
        feat = feat.join(F.broadcast(r.drop('year', 'month')), [col, 'period'], 'left')
    write_month(feat, f"{cfg['output_root']}/scoring_features", cfg['files_per_month'])
    spark.stop()


if __name__ == '__main__':
    main()
