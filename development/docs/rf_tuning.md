# Random Forest: Tuned Versions

_Generated 2026-10-11 13:39 by `src/jobs/17_rf_tuning_main.py`._

Random forest (scikit-learn) on the 23 features suggested by `16_rf_perm_elimination_main`, tuned with Optuna (TPE, seed 42) in 9 versions, **ranked by average precision on every test row** (after 2022-09). The tuning design, the versions' weights and the SMOTE rows are those of `15_gbm_tuning_main` / `14_gbm_resample_main`; focal loss has no forest counterpart and is left out, and scale_pos_weight becomes a tuned weight on the bads (`class_weight`, compared with XGBoost's `scale_pos_weight`).

- **Tuning data**: every train bad and 2% of train goods (13,239,314 rows, `rf_tuning_sample`), goods weighted 50; each version's weights multiply these. Objective: weighted AP on the valid sample drawn the same way (3,319,892 rows).
- **Per trial**: the forest is grown 50 trees at a time up to its n_estimators; the valid AP after each chunk feeds median pruning (after 5 complete trials). Searched: n_estimators 100-500, max_features 0.05-0.6, min_samples_leaf 5-5000, max_samples (bootstrap rows per tree) 200k-2M, max_depth 8-48, and the bad weight 1-150 for `class_weight`. 30 trials per version (complete + pruned).
- **Strings** as codes ordered by weighted train bad rate (from `16_rf_perm_elimination_main`).
- **Scored** with the best trial's forest on every train (460,914,721), valid (115,353,351) and test (187,470,934) row, unweighted; P(bad) rounded to 1e-6. No-skill AP is the bad rate: train 0.0089, valid 0.0089, test 0.0120.
- The valid sample scores the trials, so valid AP is optimistic; test is untouched until this comparison. XGBoost test AP from `artifacts/gbm_tuning.json`.

## Versions, best test AP first

| version      | what                                                          |   trials |   pruned |   trees |   valid AP (sample, tuning) |   train AP |   valid AP |   test AP |   train - test | test vs baseline   |   XGBoost test AP | RF - XGBoost   |
|:-------------|:--------------------------------------------------------------|---------:|---------:|--------:|----------------------------:|-----------:|-----------:|----------:|---------------:|:-------------------|------------------:|:---------------|
| undersample  | random undersampling of goods in lower-rate months            |       17 |       13 |     350 |                      0.2614 |     0.2667 |     0.2588 |    0.1895 |         0.0771 | +0.0035            |            0.2040 | -0.0144        |
| recency_6    | recency weight, half-life 6 months                            |       21 |        9 |     200 |                      0.2404 |     0.2410 |     0.2384 |    0.1894 |         0.0516 | +0.0034            |            0.2028 | -0.0134        |
| oversample   | random oversampling of bads in lower-rate months              |       23 |        7 |     400 |                      0.2603 |     0.2776 |     0.2581 |    0.1893 |         0.0883 | +0.0032            |            0.2084 | -0.0191        |
| class_weight | weight on bads tuned                                          |       17 |       13 |     150 |                      0.2639 |     0.2673 |     0.2614 |    0.1891 |         0.0781 | +0.0031            |            0.2077 | -0.0185        |
| recency_12   | recency weight, half-life 12 months                           |       15 |       15 |     250 |                      0.2518 |     0.2609 |     0.2491 |    0.1890 |         0.0720 | +0.0029            |            0.2174 | -0.0284        |
| recency_25   | recency weight, half-life 25 months                           |       18 |       12 |     100 |                      0.2566 |     0.2571 |     0.2540 |    0.1890 |         0.0681 | +0.0029            |            0.2098 | -0.0208        |
| baseline     | no reweighting                                                |       19 |       11 |     150 |                      0.2627 |     0.2679 |     0.2603 |    0.1860 |         0.0819 | +0.0000            |            0.2077 | -0.0217        |
| smotetomek   | SMOTE then Tomek links (train split, per month)               |       16 |       14 |     350 |                      0.2441 |     0.2486 |     0.2405 |    0.1777 |         0.0709 | -0.0084            |            0.1714 | +0.0063        |
| smoteenn     | SMOTE then edited nearest neighbours (train split, per month) |       12 |       18 |     350 |                      0.2314 |     0.2315 |     0.2279 |    0.1732 |         0.0583 | -0.0128            |            0.1457 | +0.0276        |

## Best parameters

| version      |   n_estimators |   max_features |   min_samples_leaf |   max_samples |   max_depth |   bad_weight |
|:-------------|---------------:|---------------:|-------------------:|--------------:|------------:|-------------:|
| baseline     |            150 |         0.3237 |                  5 |     1,822,802 |          18 |           -- |
| class_weight |            150 |         0.4825 |                  7 |     1,962,453 |          14 |        3.249 |
| recency_6    |            200 |         0.1471 |                  5 |     1,659,202 |          15 |           -- |
| recency_12   |            250 |         0.4029 |                  5 |     1,932,876 |          31 |           -- |
| recency_25   |            100 |         0.5814 |                  6 |     1,870,511 |          10 |           -- |
| oversample   |            400 |         0.2199 |                  5 |     1,996,419 |          39 |           -- |
| undersample  |            350 |         0.3907 |                  6 |     1,901,772 |          18 |           -- |
| smoteenn     |            350 |          0.211 |                 28 |     1,813,267 |          33 |           -- |
| smotetomek   |            350 |        0.05542 |                  7 |     1,873,611 |          48 |           -- |
