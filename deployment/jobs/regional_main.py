import argparse
from pyspark.sql import functions as F
from io_helpers.read import config, month_index, months
from io_helpers.write import write_month
from logic.aggregation import LEVELS, regional
from spark import spark_session


def main() -> None:
    p = argparse.ArgumentParser(description='regional_main')
    p.add_argument('--asof', required=True, help='YYYY-MM')
    a = p.parse_args()
    cfg = config()
    t = month_index(a.asof)
    spark = spark_session('regional', cfg['spark'])
    active = (months(spark, cfg['performance'], t - cfg['regional_lag_months'], t)
              .where(F.col('zero_balance_code').isNull() & F.col('dpd').isNotNull())
              .select('loan_identifier', 'period', 'dpd')
              .join(spark.read.parquet(cfg['origination']).select('loan_identifier', *LEVELS.values()),
                    'loan_identifier'))
    for lv, g in regional(active).items():
        g = g.where((F.year('period') * 12 + F.month('period')) == t)
        write_month(g, f"{cfg['output_root']}/regional_monthly/level={lv}", 1)
    spark.stop()


if __name__ == '__main__':
    main()
