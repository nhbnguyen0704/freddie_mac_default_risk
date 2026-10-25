import argparse
import importlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict
import pandas as pd
from scipy.special import expit
from helpers.config import settings

g19 = importlib.import_module('jobs.19_error_analysis_main')


def merged_probability(tables: pd.DataFrame, platt: Dict[str, list]) -> pd.DataFrame:
    """(split, p) -> rows, bads, with each segment's margin mapped by its Platt fit on valid."""
    a = tables['segment'].map(lambda s: platt[s][0])
    b = tables['segment'].map(lambda s: platt[s][1])
    p = expit(a + b * tables['m'])
    return tables.assign(p=p).groupby(['split', 'p'], as_index=False)[['n', 'bad']].sum()

def at_threshold(g: pd.DataFrame, threshold: float) -> Dict[str, float]:
    """Precision, recall and F1 when every row with p >= threshold is flagged."""
    flag = g['p'] >= threshold
    tp, flagged, bads = float(g.loc[flag, 'bad'].sum()), float(g.loc[flag, 'n'].sum()), float(g['bad'].sum())
    precision, recall = tp / flagged, tp / bads
    return {'f1': 2 * precision * recall / (precision + recall), 'precision': precision, 'recall': recall,
            'flagged': int(flagged), 'flagged_share': flagged / float(g['n'].sum()), 'true_positives': int(tp),
            'missed_bads': int(bads - tp)}

def main() -> None:
    """The F1-optimal cut on the merged probability, chosen on valid, with precision, recall and F1 at that cut on
    test, to merge_threshold.json."""
    p = argparse.ArgumentParser(description='merge_threshold_main')
    p.add_argument('--tables', default=str(settings.model_dir / 'segment_merge_tables.parquet'),
                   help='written by 22_segment_merge_main')
    p.add_argument('--merge-json', default=str(settings.model_dir / 'segment_merge.json'))
    p.add_argument('--result-json', default=str(settings.model_dir / 'merge_threshold.json'))
    a = p.parse_args()

    platt = json.loads(Path(a.merge_json).read_text())['merge']['platt_valid']
    g = merged_probability(pd.read_parquet(a.tables), platt)
    valid = g19.f1_threshold(g[g['split'] == 'valid'], 'p')
    result = {'threshold': valid['threshold'], 'chosen_on': 'valid', 'platt_valid': platt, 'valid': valid,
              'test': at_threshold(g[g['split'] == 'test'], valid['threshold']),
              'source': 'src/jobs/24_merge_threshold_main.py', 'generated': datetime.now().isoformat(timespec='seconds')}
    Path(a.result_json).write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
