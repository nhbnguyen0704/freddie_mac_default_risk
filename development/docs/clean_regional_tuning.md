# Clean Regional Model: Re-Tuned

_Generated 2026-10-14 22:33 by `src/jobs/23_clean_regional_tuning_main.py`._

Job 20's **clean_regional** (clean rows, 53 features with the regional ones) reused `recency_12`'s hyperparameters. **clean_regional_tuned** re-tunes them as job 15 tunes a version: the same search space, Optuna TPE, median pruning, 30 trials (7 complete, 23 pruned), objective weighted AP on the clean valid sample. Trial 0 is `recency_12`'s hyperparameters; the best is trial 0 (valid AP on the sample 0.0465, 572 rounds).

- Clean AP is on every clean valid / test row. The merged ranking is job 22's: the clean model on clean rows, `all_rows_recency_12` on delinquent rows (its scores reused from job 22). Current single model: test AP 0.2174.
- The valid sample picks the trial and stops each fit, so valid AP is optimistic; test is untouched.

## Before and after re-tuning

| measure                        |   clean_regional |   clean_regional_tuned | change   |
|:-------------------------------|-----------------:|-----------------------:|:---------|
| clean valid AP                 |           0.0467 |                 0.0467 | +0.0000  |
| clean test AP                  |           0.0782 |                 0.0782 | +0.0000  |
| merged valid AP, raw           |           0.2565 |                 0.2565 | +0.0000  |
| merged valid AP, Platt (valid) |           0.2569 |                 0.2569 | +0.0000  |
| merged test AP, raw            |           0.2215 |                 0.2215 | +0.0000  |
| merged test AP, Platt (valid)  |           0.2216 |                 0.2216 | +0.0000  |

## Hyperparameters

| hyperparameter   |   recency_12 (reused by job 20) |   clean_regional_tuned |
|:-----------------|--------------------------------:|-----------------------:|
| learning_rate    |                         0.03649 |                0.03649 |
| max_depth        |                               9 |                      9 |
| min_child_weight |                           9.997 |                  9.997 |
| subsample        |                          0.7869 |                 0.7869 |
| colsample_bytree |                          0.4055 |                 0.4055 |
| gamma            |                          0.1835 |                 0.1835 |
| reg_lambda       |                           6.291 |                  6.291 |
| reg_alpha        |                           6.614 |                  6.614 |
