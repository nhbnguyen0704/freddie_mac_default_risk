import argparse
from typing import Iterator
import pandas as pd
import xgboost as xgb
from pyspark.sql import functions as F
from io_helpers.read import champion, config, month_index, months
from io_helpers.write import write_month
from logic.transformation import score
from spark import spark_session

KEYS = ['loan_identifier', 'period']
SCHEMA = 'loan_identifier string, period date, segment string, margin double, pd double, flag boolean'


def main() -> None:
    p = argparse.ArgumentParser(description='predict_main')
    p.add_argument('--asof', required=True, help='YYYY-MM')
    a = p.parse_args()
    cfg = config()
    t = month_index(a.asof)
    model, version = champion(cfg)
    spark = spark_session('predict', cfg['spark'])
    bundle = model.bundle
    raw = spark.sparkContext.broadcast({s: b.save_raw('ubj') for s, b in model.boosters.items()})

    def batches(it: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        boosters = {}
        for s, r in raw.value.items():
            boosters[s] = xgb.Booster()
            boosters[s].load_model(bytearray(r))
        for pdf in it:
            yield pd.concat([pdf[KEYS], score(pdf, boosters, bundle)], axis=1)

    feats = sorted({c for s in bundle['segments'].values() for c in s['features']} | {'dpd_cnt30_l3m'})
    scored = (months(spark, f"{cfg['output_root']}/scoring_features", t, t).select(*KEYS, *feats)
              .mapInPandas(batches, SCHEMA).cache())
    cuts = scored.approxQuantile('pd', [i / 10 for i in range(1, 10)], 1e-4)
    decile = F.lit(10) - sum((F.col('pd') > q).cast('int') for q in cuts)
    write_month(scored.withColumn('decile', decile).withColumn('model_version', F.lit(version)),
                f"{cfg['output_root']}/scores", cfg['files_per_month'])
    spark.stop()


if __name__ == '__main__':
    main()
