import argparse
import mlflow
from pyspark.sql import Column, DataFrame, functions as F
from io_helpers.query import missing_months, month_label
from io_helpers.read import champion, config, local_path, month_index, months
from io_helpers.write import write_json
from logic.aggregation import outcome_metrics, psi, shares
from logic.preprocessing import workout
from spark import spark_session

PSI_TRIGGER = 0.25
PROXY_MONTHS = 6
LABEL_MONTHS = 25


def with_outcome(spark, cfg: dict, s: int, horizon: int, bad: Column) -> DataFrame:
    """The scores of month s with bad = 1 when the loan meets the condition in any month s+1 .. s+horizon."""
    perf = months(spark, cfg['performance'], s + 1, s + horizon)
    outcome = perf.groupBy('loan_identifier').agg(F.max(bad.cast('int')).alias('bad'))
    return (months(spark, f"{cfg['output_root']}/scores", s, s)
            .join(outcome, 'loan_identifier', 'left').fillna({'bad': 0}))

def drift(spark, cfg: dict, t: int, baseline: dict) -> dict:
    """Volumes, flag rate and mean PD per segment, score PSI per segment and feature PSI for month t."""
    scores = months(spark, f"{cfg['output_root']}/scores", t, t)
    out = {}
    for r in scores.groupBy('segment').agg(F.count(F.lit(1)).alias('n'), F.avg('pd').alias('pd'),
                                           F.avg(F.col('flag').cast('int')).alias('flag')).collect():
        out |= {f"rows_{r['segment']}": r['n'], f"mean_pd_{r['segment']}": r['pd'], f"flag_rate_{r['segment']}": r['flag']}
    out['flag_rate'] = scores.agg(F.avg(F.col('flag').cast('int'))).first()[0]
    for seg, spec in baseline['scores'].items():
        actual = shares(scores.where(F.col('segment') == seg), {'pd': spec})['pd']
        out[f'score_psi_{seg}'] = psi(spec['shares'], actual)
    feats = months(spark, f"{cfg['output_root']}/scoring_features", t, t)
    for c, a in shares(feats, baseline['features']).items():
        out[f'psi_{c}'] = psi(baseline['features'][c]['shares'], a)
    out['retrain_trigger'] = int(max(out[f'score_psi_{s}'] for s in baseline['scores']) > PSI_TRIGGER)
    return out

def main() -> None:
    p = argparse.ArgumentParser(description='monitor_main')
    p.add_argument('--asof', required=True, help='YYYY-MM')
    a = p.parse_args()
    cfg = config()
    t = month_index(a.asof)
    model, version = champion(cfg)
    spark = spark_session('monitor', cfg['spark'])
    metrics = drift(spark, cfg, t, model.bundle['baseline'])
    params = {'asof': a.asof, 'model_version': version}
    scores = f"{cfg['output_root']}/scores"
    for name, back, bad in (
            ('proxy', PROXY_MONTHS, F.coalesce(F.col('dpd') >= 2, F.lit(False))),
            ('backtest', LABEL_MONTHS, F.coalesce(F.col('dpd') >= 3, F.lit(False))
             | F.coalesce(F.trim('zero_balance_code').isin('02', '03', '09'), F.lit(False)) | workout())):
        if not missing_months(spark, scores, t - back, t - back):
            metrics |= outcome_metrics(with_outcome(spark, cfg, t - back, back, bad), name)
            params[f'{name}_month'] = month_label(t - back)
    spark.stop()
    with mlflow.start_run(run_name=a.asof, tags={'model_version': version}):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
    write_json(local_path(f'runs/{a.asof}.json'), {'params': params, 'metrics': metrics})


if __name__ == '__main__':
    main()
