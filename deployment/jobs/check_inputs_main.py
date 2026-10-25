import argparse
from io_helpers.query import missing_months
from io_helpers.read import config, month_index
from spark import spark_session


def main() -> None:
    p = argparse.ArgumentParser(description='check_inputs_main')
    p.add_argument('--asof', required=True, help='YYYY-MM')
    a = p.parse_args()
    cfg = config()
    t = month_index(a.asof)
    spark = spark_session('check_inputs', cfg['spark'])
    first = t - max(cfg['history_months'] - 1, cfg['regional_lag_months'])
    missing = missing_months(spark, cfg['performance'], first, t)
    spark.stop()
    if missing:
        raise SystemExit(f"Performance months missing for {a.asof}: {', '.join(missing)}")


if __name__ == '__main__':
    main()
