# XGBoost Feature Selection: Spearman Correlation Filter

_Generated 2026-10-05 01:28 by `src/jobs/12_gbm_corr_selection_main.py`._

Correlation filter before the XGBoost elimination, on the **raw** features of the **train** split (2022-09 and before, loans not in valid; 460,914,721 rows). Valid and test are not used, except to score the univariate AP.

1. **Univariate AP** (tie breaker): each feature alone as a lookup scorer -- the train bad rate of its category; of its value for a numeric feature with at most 1,000 distinct values; otherwise of its 100 equal-frequency train bins, with any value holding more than 1/100 of the rows as a bin of its own; missing as a group of its own -- and AP over every valid row.
2. **Spearman** between every pair of the 62 numeric features, on the train rows where **both** are non-null, each ranked again within those rows (midranks for ties), as pandas or R with pairwise-complete observations. Columns with more than 1,000,000 distinct values are ranked on a 50,000-quantile grid of train (a value's rank is its bucket's midrank, so it moves by at most ~1/50,000 of the rows): `current_actual_upb`, `upb_pct_of_original`, `estimated_property_value`, `property_value_change`, `upb_pct_of_original_roc_l3m`, `upb_pct_of_original_roc_l6m`, `upb_pct_of_original_roc_l12m`, `property_value_roc_l3m`, `property_value_roc_l6m`, `property_value_roc_l12m`; every other column on its exact values. 
3. **Filter**: while any pair has |Spearman| > 0.8, drop the feature with the highest max |corr|, then the highest median |corr|, then the lowest univariate AP (24 dropped).

The 17 string features have no Spearman correlation and all go forward; so do numeric features with no correlation above the threshold.

**55 candidates for the elimination**: 38 numeric + 17 categorical.

## Correlation filter

|   round | dropped                     |   max_corr | with                            |    corr |   median_corr |   univariate_ap |   partner_univariate_ap |   features_left |
|--------:|:----------------------------|-----------:|:--------------------------------|--------:|--------------:|----------------:|------------------------:|----------------:|
|       1 | dpd60_streak_l6m            |          1 | dpd60_streak_l12m               |       1 |        0.0215 |          0.0703 |                  0.0705 |              61 |
|       2 | dpd60_streak_l3m            |          1 | dpd60_streak_l12m               |       1 |        0.0198 |          0.0698 |                  0.0705 |              60 |
|       3 | dpd30_streak_l12m           |          1 | dpd30_streak_l6m                |       1 |        0.0445 |          0.1117 |                  0.1106 |              59 |
|       4 | dpd30_streak_l6m            |          1 | dpd30_streak_l3m                |       1 |        0.0404 |          0.1106 |                  0.1097 |              58 |
|       5 | dpd_cnt60_l12m              |          1 | months_since_last_dpd60         |      -1 |        0.0292 |          0.1027 |                  0.1238 |              57 |
|       6 | dpd30_streak_l3m            |          1 | dpd                             |       1 |        0.0331 |          0.1097 |                  0.1199 |              56 |
|       7 | original_interest_rate      |          1 | current_interest_rate           |       1 |        0.0761 |          0.0235 |                  0.0235 |              55 |
|       8 | dpd_cnt30_l3m               |          1 | dpd_max_l3m                     |       1 |        0.0325 |          0.1254 |                  0.1292 |              54 |
|       9 | dpd_cnt30_l6m               |          1 | dpd_max_l6m                     |       1 |        0.0306 |          0.1339 |                  0.1261 |              53 |
|      10 | dpd_cnt30_l12m              |     0.9999 | dpd_max_l12m                    |  0.9999 |        0.0338 |          0.1349 |                  0.1172 |              52 |
|      11 | months_since_last_dpd30     |     0.9999 | dpd_max_l12m                    | -0.9999 |        0.0337 |          0.1231 |                  0.1172 |              51 |
|      12 | upb_pct_of_original_roc_l6m |     0.9797 | upb_pct_of_original_roc_l12m    |  0.9797 |        0.0397 |          0.0202 |                  0.0169 |              50 |
|      13 | original_loan_to_value      |     0.9738 | original_combined_loan_to_value |  0.9738 |        0.0256 |          0.0124 |                  0.0128 |              49 |
|      14 | dpd_max_l12m                |     0.9577 | n_roll_forward_l12m             |  0.9577 |        0.0334 |          0.1172 |                   0.127 |              48 |
|      15 | upb_pct_of_original_roc_l3m |     0.9408 | upb_pct_of_original_roc_l12m    |  0.9408 |         0.037 |          0.0236 |                  0.0169 |              47 |
|      16 | original_upb                |     0.9119 | current_actual_upb              |  0.9119 |        0.0484 |          0.0135 |                  0.0138 |              46 |
|      17 | dpd_max_l6m                 |      0.904 | n_roll_forward_l6m              |   0.904 |         0.027 |          0.1261 |                  0.1071 |              45 |
|      18 | n_curtail_l6m               |     0.8931 | n_curtail_l12m                  |  0.8931 |        0.0166 |          0.0101 |                  0.0112 |              44 |
|      19 | n_roll_forward_l12m         |      0.887 | n_cure_l12m                     |   0.887 |        0.0302 |           0.127 |                  0.0498 |              43 |
|      20 | loan_age                    |     0.8636 | upb_pct_of_original             | -0.8636 |        0.0492 |          0.0165 |                   0.012 |              42 |
|      21 | dpd_cnt60_l6m               |      0.836 | dpd_cnt60_l3m                   |   0.836 |        0.0158 |          0.0996 |                  0.0905 |              41 |
|      22 | dpd_max_l3m                 |     0.8351 | n_roll_forward_l6m              |  0.8351 |        0.0232 |          0.1292 |                  0.1071 |              40 |
|      23 | ltv_ratio_to_original       |     0.8345 | property_value_change           | -0.8345 |        0.0708 |           0.011 |                  0.0114 |              39 |
|      24 | estimated_loan_to_value     |       0.81 | ltv_x_dti                       |    0.81 |        0.1045 |          0.0123 |                  0.0117 |              38 |

## Candidates for the elimination

`current_actual_upb`, `current_interest_rate`, `remaining_months_to_legal_maturity`, `classic_fico`, `mortgage_insurance_percentage`, `number_of_units`, `original_combined_loan_to_value`, `original_debt_to_income_ratio`, `original_loan_term`, `number_of_borrowers`, `dpd_cnt60_l3m`, `current_interest_rate_roc_l3m`, `current_interest_rate_roc_l6m`, `current_interest_rate_roc_l12m`, `dpd`, `months_since_last_dpd60`, `current_streak_l3m`, `current_streak_l6m`, `current_streak_l12m`, `dpd60_streak_l12m`, `n_roll_forward_l3m`, `n_cure_l3m`, `n_roll_forward_l6m`, `n_cure_l6m`, `n_cure_l12m`, `n_curtail_l3m`, `n_curtail_l12m`, `servicer_changed_l12m`, `rate_gap`, `ltv_x_dti`, `upb_pct_of_original`, `negative_equity_flag`, `estimated_property_value`, `property_value_change`, `upb_pct_of_original_roc_l12m`, `property_value_roc_l3m`, `property_value_roc_l6m`, `property_value_roc_l12m`, `servicer_name`, `first_time_homebuyer_indicator`, `metropolitan_statistical_area_or_metropolitan_division`, `occupancy_status`, `channel`, `prepayment_penalty_indicator`, `amortization_type`, `property_state`, `property_type`, `postal_code`, `loan_purpose`, `seller_name`, `super_conforming_flag`, `special_eligibility_program`, `harp_indicator`, `property_valuation_method`, `interest_only_indicator`

## Univariate AP (valid), best first

| feature                                                | kind        | status                |   univariate_ap_valid |   univariate_ap_train |   groups |   missing_share_train |   unseen_in_train_valid_rows |
|:-------------------------------------------------------|:------------|:----------------------|----------------------:|----------------------:|---------:|----------------------:|-----------------------------:|
| dpd_cnt30_l12m                                         | numerical   | dropped (correlation) |                0.1349 |                0.1361 |       13 |                     0 |                            0 |
| dpd_cnt30_l6m                                          | numerical   | dropped (correlation) |                0.1339 |                0.1348 |        7 |                     0 |                            0 |
| dpd_max_l3m                                            | numerical   | dropped (correlation) |                0.1292 |                0.1298 |       33 |                     0 |                            0 |
| n_roll_forward_l12m                                    | numerical   | dropped (correlation) |                 0.127 |                0.1274 |       11 |                     0 |                            0 |
| dpd_max_l6m                                            | numerical   | dropped (correlation) |                0.1261 |                0.1269 |       40 |                     0 |                            0 |
| dpd_cnt30_l3m                                          | numerical   | dropped (correlation) |                0.1254 |                0.1259 |        4 |                     0 |                            0 |
| months_since_last_dpd60                                | numerical   | candidate             |                0.1238 |                0.1246 |       13 |                     0 |                            0 |
| months_since_last_dpd30                                | numerical   | dropped (correlation) |                0.1231 |                0.1233 |       13 |                     0 |                            0 |
| dpd                                                    | numerical   | candidate             |                0.1199 |                0.1202 |        3 |                     0 |                            0 |
| dpd_max_l12m                                           | numerical   | dropped (correlation) |                0.1172 |                0.1182 |       46 |                     0 |                            0 |
| dpd30_streak_l12m                                      | numerical   | dropped (correlation) |                0.1117 |                 0.112 |       13 |                     0 |                            0 |
| dpd30_streak_l6m                                       | numerical   | dropped (correlation) |                0.1106 |                0.1109 |        7 |                     0 |                            0 |
| dpd30_streak_l3m                                       | numerical   | dropped (correlation) |                0.1097 |                  0.11 |        4 |                     0 |                            0 |
| n_roll_forward_l6m                                     | numerical   | candidate             |                0.1071 |                0.1072 |        5 |                     0 |                            0 |
| dpd_cnt60_l12m                                         | numerical   | dropped (correlation) |                0.1027 |                0.1039 |       13 |                     0 |                            0 |
| current_streak_l12m                                    | numerical   | candidate             |                0.1012 |                0.1013 |       13 |                     0 |                            0 |
| dpd_cnt60_l6m                                          | numerical   | dropped (correlation) |                0.0996 |                0.1005 |        7 |                     0 |                            0 |
| current_streak_l6m                                     | numerical   | candidate             |                0.0994 |                0.0995 |        7 |                     0 |                            0 |
| current_streak_l3m                                     | numerical   | candidate             |                0.0969 |                0.0969 |        4 |                     0 |                            0 |
| dpd_cnt60_l3m                                          | numerical   | candidate             |                0.0905 |                0.0911 |        4 |                     0 |                            0 |
| n_roll_forward_l3m                                     | numerical   | candidate             |                0.0736 |                0.0734 |        3 |                     0 |                            0 |
| dpd60_streak_l12m                                      | numerical   | candidate             |                0.0705 |                0.0709 |       13 |                     0 |                            0 |
| dpd60_streak_l6m                                       | numerical   | dropped (correlation) |                0.0703 |                0.0707 |        7 |                     0 |                            0 |
| dpd60_streak_l3m                                       | numerical   | dropped (correlation) |                0.0698 |                0.0702 |        4 |                     0 |                            0 |
| n_cure_l12m                                            | numerical   | candidate             |                0.0498 |                0.0502 |        7 |                     0 |                            0 |
| n_cure_l6m                                             | numerical   | candidate             |                0.0356 |                0.0358 |        4 |                     0 |                            0 |
| classic_fico                                           | numerical   | candidate             |                0.0262 |                 0.027 |      402 |                0.0003 |                            0 |
| upb_pct_of_original_roc_l3m                            | numerical   | dropped (correlation) |                0.0236 |                0.0233 |      101 |                0.1543 |                            0 |
| original_interest_rate                                 | numerical   | dropped (correlation) |                0.0235 |                0.0236 |       94 |                     0 |                            0 |
| current_interest_rate                                  | numerical   | candidate             |                0.0235 |                0.0236 |       94 |                     0 |                            0 |
| n_cure_l3m                                             | numerical   | candidate             |                0.0219 |                0.0219 |        2 |                     0 |                            0 |
| upb_pct_of_original_roc_l6m                            | numerical   | dropped (correlation) |                0.0202 |                0.0199 |      101 |                0.1543 |                            0 |
| remaining_months_to_legal_maturity                     | numerical   | candidate             |                0.0179 |                0.0187 |      389 |                     0 |                            0 |
| upb_pct_of_original_roc_l12m                           | numerical   | candidate             |                0.0169 |                0.0166 |      101 |                0.1543 |                            0 |
| loan_age                                               | numerical   | dropped (correlation) |                0.0165 |                0.0166 |      284 |                     0 |                            0 |
| seller_name                                            | categorical | candidate             |                0.0154 |                0.0154 |      105 |                     0 |                            0 |
| current_actual_upb                                     | numerical   | candidate             |                0.0138 |                 0.014 |      101 |                     0 |                            0 |
| original_upb                                           | numerical   | dropped (correlation) |                0.0135 |                0.0136 |      104 |                     0 |                            0 |
| estimated_property_value                               | numerical   | candidate             |                0.0135 |                0.0134 |      101 |                0.2797 |                            0 |
| original_combined_loan_to_value                        | numerical   | candidate             |                0.0128 |                 0.013 |      429 |                     0 |                            0 |
| original_debt_to_income_ratio                          | numerical   | candidate             |                0.0128 |                0.0128 |       66 |                0.1331 |                            0 |
| channel                                                | categorical | candidate             |                0.0128 |                0.0128 |        5 |                     0 |                            0 |
| servicer_name                                          | categorical | candidate             |                0.0128 |                0.0129 |       44 |                     0 |                            0 |
| original_loan_to_value                                 | numerical   | dropped (correlation) |                0.0124 |                0.0126 |      401 |                     0 |                            0 |
| postal_code                                            | categorical | candidate             |                0.0124 |                0.0127 |      900 |                     0 |                           22 |
| estimated_loan_to_value                                | numerical   | dropped (correlation) |                0.0123 |                0.0124 |      261 |                0.2785 |                            0 |
| metropolitan_statistical_area_or_metropolitan_division | categorical | candidate             |                0.0121 |                0.0123 |      462 |                0.1411 |                            0 |
| upb_pct_of_original                                    | numerical   | candidate             |                 0.012 |                0.0123 |      103 |                     0 |                            0 |
| ltv_x_dti                                              | numerical   | candidate             |                0.0117 |                0.0117 |      101 |                0.3603 |                            0 |
| property_value_change                                  | numerical   | candidate             |                0.0114 |                0.0114 |      101 |                0.2797 |                            0 |
| n_curtail_l12m                                         | numerical   | candidate             |                0.0112 |                0.0112 |       12 |                     0 |                            0 |
| number_of_borrowers                                    | numerical   | candidate             |                0.0111 |                 0.011 |        6 |                     0 |                            0 |
| ltv_ratio_to_original                                  | numerical   | dropped (correlation) |                 0.011 |                 0.011 |      105 |                0.2785 |                            0 |
| property_state                                         | categorical | candidate             |                0.0109 |                0.0108 |       54 |                     0 |                            0 |
| mortgage_insurance_percentage                          | numerical   | candidate             |                0.0107 |                0.0106 |       43 |                     0 |                            0 |
| property_value_roc_l3m                                 | numerical   | candidate             |                0.0105 |                0.0105 |      101 |                0.3962 |                            0 |
| original_loan_term                                     | numerical   | candidate             |                0.0102 |                0.0102 |      263 |                     0 |                            0 |
| negative_equity_flag                                   | numerical   | candidate             |                0.0101 |                0.0102 |        3 |                0.2785 |                            0 |
| n_curtail_l6m                                          | numerical   | dropped (correlation) |                0.0101 |                0.0101 |        6 |                     0 |                            0 |
| property_value_roc_l6m                                 | numerical   | candidate             |                0.0101 |                0.0101 |      101 |                0.3859 |                            0 |
| property_value_roc_l12m                                | numerical   | candidate             |                0.0101 |                0.0101 |      101 |                 0.375 |                            0 |
| property_valuation_method                              | categorical | candidate             |                0.0098 |                0.0098 |        4 |                     0 |                            0 |
| property_type                                          | categorical | candidate             |                0.0098 |                0.0098 |        6 |                     0 |                            0 |
| loan_purpose                                           | categorical | candidate             |                0.0097 |                0.0097 |        3 |                     0 |                            0 |
| harp_indicator                                         | categorical | candidate             |                0.0096 |                0.0096 |        2 |                     0 |                            0 |
| n_curtail_l3m                                          | numerical   | candidate             |                0.0095 |                0.0095 |        3 |                     0 |                            0 |
| occupancy_status                                       | categorical | candidate             |                0.0093 |                0.0092 |        3 |                     0 |                            0 |
| special_eligibility_program                            | categorical | candidate             |                0.0093 |                0.0092 |        4 |                0.9764 |                            0 |
| first_time_homebuyer_indicator                         | categorical | candidate             |                0.0092 |                0.0091 |        3 |                     0 |                            0 |
| servicer_changed_l12m                                  | numerical   | candidate             |                0.0092 |                0.0092 |        2 |                     0 |                            0 |
| super_conforming_flag                                  | categorical | candidate             |                 0.009 |                 0.009 |        2 |                     0 |                            0 |
| current_interest_rate_roc_l12m                         | numerical   | candidate             |                 0.009 |                 0.009 |       10 |                0.0203 |                            0 |
| current_interest_rate_roc_l6m                          | numerical   | candidate             |                 0.009 |                 0.009 |        7 |                0.0203 |                            0 |
| current_interest_rate_roc_l3m                          | numerical   | candidate             |                 0.009 |                 0.009 |        6 |                0.0203 |                            0 |
| rate_gap                                               | numerical   | candidate             |                 0.009 |                 0.009 |       31 |                     0 |                            0 |
| number_of_units                                        | numerical   | candidate             |                 0.009 |                0.0089 |        5 |                0.0001 |                            0 |
| prepayment_penalty_indicator                           | categorical | candidate             |                 0.009 |                0.0089 |        2 |                     0 |                            0 |
| amortization_type                                      | categorical | candidate             |                0.0089 |                0.0089 |        1 |                     0 |                            0 |
| interest_only_indicator                                | categorical | candidate             |                0.0089 |                0.0089 |        1 |                     0 |                            0 |
