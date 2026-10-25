# XGBoost: Tuned Versions

_Generated 2026-10-06 23:54 by `src/jobs/15_gbm_tuning_main.py`._

XGBoost on the 38 features suggested by `13_gbm_shap_elimination_main`, tuned with Optuna (TPE, seed 42) in 10 versions, **ranked by average precision on every test row** (after 2022-09).

- **Tuning data**: every train bad and 2% of train goods (13,239,314 rows), goods weighted 50 so the rows stand for the whole train split; each version's weights multiply these. Objective: weighted AP on the valid sample drawn the same way (3,319,892 rows), unchanged across versions.
- **Per trial**: at most 2000 rounds, early stopping after 50 rounds without a better valid AUC-PR, and median pruning (valid AUC-PR every 10 rounds, after 5 complete trials and 100 rounds). 30 trials per version (complete + pruned).
- **Scored** with the best trial's model on every train (460,914,721), valid (115,353,351) and test (187,470,934) row, unweighted; margin rounded to 1e-4. No-skill AP is the bad rate: train 0.0089, valid 0.0089, test 0.0120.
- Over/undersampling lift every train month to the highest monthly train bad rate (1.4641%) as seeded frequency weights; recency weights are 0.5^(months before 2022-09 / half-life); SMOTEENN / SMOTETomek come from `14_gbm_resample_main` (see `docs/gbm_resample.md`).
- The valid sample both stops each fit early and scores the trials, so valid AP is optimistic; test is untouched until this comparison.

## Versions, best test AP first

| version          | what                                                          |   trials |   pruned |   rounds |   valid AP (sample, tuning) |   train AP |   valid AP |   test AP |   train - test | test vs baseline   |
|:-----------------|:--------------------------------------------------------------|---------:|---------:|---------:|----------------------------:|-----------:|-----------:|----------:|---------------:|:-------------------|
| recency_12       | recency weight, half-life 12 months                           |        9 |       21 |      395 |                      0.2588 |     0.2811 |     0.2564 |    0.2174 |         0.0637 | +0.0097            |
| recency_25       | recency weight, half-life 25 months                           |       14 |       16 |      418 |                      0.2623 |     0.2833 |     0.2597 |    0.2098 |         0.0735 | +0.0021            |
| oversample       | random oversampling of bads in lower-rate months              |       16 |       14 |    1,036 |                      0.2638 |     0.2871 |     0.2607 |    0.2084 |         0.0787 | +0.0006            |
| baseline         | logloss, no reweighting                                       |        7 |       23 |      446 |                      0.2690 |     0.2992 |     0.2663 |    0.2077 |         0.0915 | +0.0000            |
| scale_pos_weight | logloss, scale_pos_weight tuned                               |       15 |       15 |      256 |                      0.2679 |     0.3040 |     0.2642 |    0.2077 |         0.0964 | -0.0001            |
| undersample      | random undersampling of goods in lower-rate months            |       14 |       16 |      281 |                      0.2615 |     0.2819 |     0.2587 |    0.2040 |         0.0779 | -0.0037            |
| recency_6        | recency weight, half-life 6 months                            |       16 |       14 |      280 |                      0.2390 |     0.2422 |     0.2360 |    0.2028 |         0.0394 | -0.0050            |
| focal_loss       | focal loss, alpha and gamma tuned                             |       29 |        1 |      862 |                      0.2604 |     0.2898 |     0.2575 |    0.1981 |         0.0917 | -0.0096            |
| smotetomek       | SMOTE then Tomek links (train split, per month)               |       29 |        1 |       59 |                      0.2279 |     0.2319 |     0.2255 |    0.1714 |         0.0605 | -0.0363            |
| smoteenn         | SMOTE then edited nearest neighbours (train split, per month) |       12 |       18 |      930 |                      0.2429 |     0.2503 |     0.2386 |    0.1457 |         0.1046 | -0.0621            |

## Best parameters

| version          |   learning_rate |   max_depth |   min_child_weight |   subsample |   colsample_bytree |   gamma |   reg_lambda |   reg_alpha |   scale_pos_weight |   focal_alpha |   focal_gamma |
|:-----------------|----------------:|------------:|-------------------:|------------:|-------------------:|--------:|-------------:|------------:|-------------------:|--------------:|--------------:|
| baseline         |         0.03836 |           8 |               69.4 |      0.8684 |             0.4356 |  0.9129 |         49.5 |     0.03533 |                 -- |            -- |            -- |
| scale_pos_weight |         0.08123 |           8 |              9.309 |       0.577 |             0.4865 |   3.873 |      0.01007 |      0.1015 |              4.055 |            -- |            -- |
| focal_loss       |         0.06089 |           6 |               9.84 |       0.784 |             0.4572 |    3.99 |      0.02204 |       4.422 |                 -- |        0.8616 |       0.07929 |
| recency_6        |         0.08761 |           4 |              387.4 |      0.6121 |             0.6114 |   3.273 |     0.006565 |    0.001045 |                 -- |            -- |            -- |
| recency_12       |         0.03649 |           9 |              9.997 |      0.7869 |             0.4055 |  0.1835 |        6.291 |       6.614 |                 -- |            -- |            -- |
| recency_25       |         0.03249 |           7 |              295.8 |      0.8629 |             0.7592 |   2.946 |      0.08371 |    0.007809 |                 -- |            -- |            -- |
| oversample       |         0.03933 |           6 |              649.8 |       0.818 |             0.4261 |   1.049 |     0.007214 |      0.2203 |                 -- |            -- |            -- |
| undersample      |         0.03455 |           7 |              20.07 |      0.8867 |             0.7896 |   2.388 |        84.37 |       2.237 |                 -- |            -- |            -- |
| smoteenn         |          0.1728 |           5 |              10.96 |      0.6759 |             0.4873 |   4.863 |        35.15 |      0.1737 |                 -- |            -- |            -- |
| smotetomek       |         0.03121 |           9 |              183.7 |      0.6215 |             0.4132 |   3.654 |      0.05177 |       5.078 |                 -- |            -- |            -- |
