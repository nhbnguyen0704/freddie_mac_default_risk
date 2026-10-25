# Model Score Correlation on Test

_Generated 2026-10-11 16:32 by `src/jobs/18_model_correlation_main.py`._

Correlation of the predicted P(bad) of the best version of each model (by test AP on every test row) over **all 187,470,934 test rows** (after 2022-09; 2,240,522 bads). Each row is scored by the three models in one pass: the logistic regression on its WoE columns (`training_set_woe_selected`), XGBoost and the random forest on the raw columns (`training_set`), joined on (loan, period).

- **Pearson (probability)**: on P(bad) itself; dominated by the few high-risk rows.
- **Pearson (logit)**: on log-odds (LR logit, XGBoost margin, RF P(bad) clipped to [1e-06, 1 - 1e-06]); spreads the low-risk mass out.
- **Spearman**: Pearson of the ranks, ties sharing their mid-rank; scores rounded as in the AP tables (logit and margin to 1e-4, RF P(bad) to 1e-6). Rank agreement is what matters for AP.
- Test AP recomputed from these scores: Logistic regression (recency_6) 0.1399, XGBoost (recency_12) 0.2174, Random forest (undersample) 0.1895 (matches each model's own job).

## Spearman (rank)

|                                 |   Logistic regression (recency_6) |   XGBoost (recency_12) |   Random forest (undersample) |
|:--------------------------------|----------------------------------:|-----------------------:|------------------------------:|
| Logistic regression (recency_6) |                                 1 |                 0.9133 |                        0.8967 |
| XGBoost (recency_12)            |                            0.9133 |                      1 |                        0.9033 |
| Random forest (undersample)     |                            0.8967 |                 0.9033 |                             1 |

## Pearson (probability)

|                                 |   Logistic regression (recency_6) |   XGBoost (recency_12) |   Random forest (undersample) |
|:--------------------------------|----------------------------------:|-----------------------:|------------------------------:|
| Logistic regression (recency_6) |                                 1 |                 0.7467 |                        0.7721 |
| XGBoost (recency_12)            |                            0.7467 |                      1 |                        0.9115 |
| Random forest (undersample)     |                            0.7721 |                 0.9115 |                             1 |

## Pearson (logit)

|                                 |   Logistic regression (recency_6) |   XGBoost (recency_12) |   Random forest (undersample) |
|:--------------------------------|----------------------------------:|-----------------------:|------------------------------:|
| Logistic regression (recency_6) |                                 1 |                 0.8886 |                        0.8506 |
| XGBoost (recency_12)            |                            0.8886 |                      1 |                        0.9072 |
| Random forest (undersample)     |                            0.8506 |                 0.9072 |                             1 |

## Pearson (logit), bads only

|                                 |   Logistic regression (recency_6) |   XGBoost (recency_12) |   Random forest (undersample) |
|:--------------------------------|----------------------------------:|-----------------------:|------------------------------:|
| Logistic regression (recency_6) |                                 1 |                 0.8645 |                        0.8838 |
| XGBoost (recency_12)            |                            0.8645 |                      1 |                        0.9417 |
| Random forest (undersample)     |                            0.8838 |                 0.9417 |                             1 |

## Pearson (logit), goods only

|                                 |   Logistic regression (recency_6) |   XGBoost (recency_12) |   Random forest (undersample) |
|:--------------------------------|----------------------------------:|-----------------------:|------------------------------:|
| Logistic regression (recency_6) |                                 1 |                  0.886 |                        0.8451 |
| XGBoost (recency_12)            |                             0.886 |                      1 |                        0.9023 |
| Random forest (undersample)     |                            0.8451 |                 0.9023 |                             1 |
