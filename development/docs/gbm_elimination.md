# XGBoost Feature Selection: SHAP Backward Elimination

_Generated 2026-10-05 15:11 by `src/jobs/13_gbm_shap_elimination_main.py`._

Backward elimination with XGBoost on the **raw** features, from the 55 candidates left by `12_gbm_corr_selection_main`. Each step fits one model, scores it, and drops the feature with the lowest **mean |SHAP|** (lowest univariate AP on a tie), until 1 remain.

- **Fit** on every train bad and 2% of train goods (goods weighted 50): 13,239,314 rows standing for 460,703,345. Early stopping on the valid sample, drawn the same way (3,319,892 rows), weighted AUC-PR, patience 50, at most 2000 rounds. Fixed parameters: tree_method=hist, learning_rate=0.1, max_depth=6, min_child_weight=1.0, subsample=0.8, colsample_bytree=0.8, max_bin=256, seed=42.
- **Mean |SHAP|** (TreeSHAP, weighted) on 332,193 rows of the valid sample.
- **AP full**: the model scored on **every** train and valid row (train 460,914,721, valid 115,353,351), unweighted, margin rounded to 1e-4. AP sample: on the weighted samples.

**Elbow**: best full valid AP 0.2545 with 55 features; the fewest features within 0.5% of it: **38** (valid AP 0.2537). This is a suggestion; the curve is below.

## Elimination curve

|   step |   features |   rounds |   AP full train |   AP full valid |   train - valid |   AP sample train |   AP sample valid | dropped next                                           |   its mean \|SHAP\| |
|-------:|-----------:|---------:|----------------:|----------------:|----------------:|------------------:|------------------:|:-------------------------------------------------------|--------------------:|
|      1 |         55 |      152 |          0.2806 |          0.2545 |          0.0261 |            0.3139 |            0.2572 | amortization_type                                      |              0.0000 |
|      2 |         54 |      148 |          0.2788 |           0.254 |          0.0248 |            0.3086 |             0.256 | interest_only_indicator                                |              0.0000 |
|      3 |         53 |      208 |          0.2825 |          0.2512 |          0.0313 |            0.3204 |            0.2531 | current_interest_rate_roc_l3m                          |              0.0000 |
|      4 |         52 |      281 |          0.2845 |          0.2486 |          0.0359 |            0.3303 |            0.2514 | rate_gap                                               |              0.0000 |
|      5 |         51 |      177 |          0.2774 |           0.249 |          0.0284 |            0.3162 |            0.2515 | prepayment_penalty_indicator                           |              0.0000 |
|      6 |         50 |      206 |           0.283 |          0.2517 |          0.0314 |              0.32 |            0.2534 | super_conforming_flag                                  |              0.0000 |
|      7 |         49 |      274 |           0.284 |          0.2491 |          0.0349 |            0.3292 |            0.2524 | current_interest_rate_roc_l6m                          |              0.0004 |
|      8 |         48 |      322 |           0.287 |          0.2483 |          0.0387 |             0.336 |            0.2526 | harp_indicator                                         |              0.0005 |
|      9 |         47 |      203 |          0.2788 |          0.2479 |          0.0309 |            0.3194 |            0.2526 | current_interest_rate_roc_l12m                         |              0.0004 |
|     10 |         46 |      183 |          0.2789 |          0.2501 |          0.0288 |             0.317 |            0.2524 | first_time_homebuyer_indicator                         |              0.0001 |
|     11 |         45 |      274 |          0.2854 |          0.2502 |          0.0352 |            0.3298 |            0.2523 | dpd60_streak_l12m                                      |              0.0006 |
|     12 |         44 |      200 |          0.2789 |          0.2487 |          0.0302 |            0.3169 |             0.251 | negative_equity_flag                                   |              0.0010 |
|     13 |         43 |      194 |          0.2819 |          0.2526 |          0.0293 |            0.3177 |             0.255 | special_eligibility_program                            |              0.0016 |
|     14 |         42 |      228 |          0.2833 |          0.2509 |          0.0324 |            0.3242 |            0.2526 | n_roll_forward_l3m                                     |              0.0023 |
|     15 |         41 |      339 |          0.2916 |          0.2515 |            0.04 |            0.3379 |            0.2538 | dpd_cnt60_l3m                                          |              0.0043 |
|     16 |         40 |      241 |          0.2873 |          0.2532 |          0.0341 |            0.3275 |            0.2558 | channel                                                |              0.0033 |
|     17 |         39 |      210 |          0.2825 |          0.2509 |          0.0316 |            0.3223 |            0.2534 | property_value_roc_l6m                                 |              0.0029 |
|     18 |         38 |      214 |          0.2852 |          0.2537 |          0.0315 |            0.3219 |            0.2558 | number_of_units                                        |              0.0039 |
|     19 |         37 |      211 |          0.2834 |          0.2519 |          0.0315 |            0.3197 |            0.2545 | n_cure_l6m                                             |              0.0059 |
|     20 |         36 |      294 |          0.2868 |          0.2499 |          0.0369 |            0.3303 |            0.2529 | servicer_changed_l12m                                  |              0.0069 |
|     21 |         35 |      183 |           0.277 |          0.2475 |          0.0296 |            0.3145 |            0.2495 | current_streak_l3m                                     |              0.0050 |
|     22 |         34 |      211 |           0.282 |          0.2492 |          0.0328 |            0.3212 |            0.2512 | n_cure_l3m                                             |              0.0053 |
|     23 |         33 |      304 |          0.2847 |          0.2462 |          0.0385 |            0.3319 |            0.2479 | property_state                                         |              0.0104 |
|     24 |         32 |      181 |            0.28 |           0.251 |           0.029 |            0.3163 |            0.2539 | mortgage_insurance_percentage                          |              0.0070 |
|     25 |         31 |      241 |          0.2826 |          0.2492 |          0.0335 |            0.3233 |            0.2509 | current_streak_l6m                                     |              0.0088 |
|     26 |         30 |      274 |          0.2843 |          0.2485 |          0.0358 |            0.3305 |            0.2517 | property_value_change                                  |              0.0129 |
|     27 |         29 |      212 |          0.2807 |          0.2479 |          0.0328 |            0.3195 |            0.2495 | property_value_roc_l3m                                 |              0.0108 |
|     28 |         28 |      335 |          0.2886 |          0.2489 |          0.0397 |            0.3372 |            0.2511 | property_type                                          |              0.0160 |
|     29 |         27 |      322 |          0.2857 |          0.2472 |          0.0386 |            0.3334 |            0.2495 | property_value_roc_l12m                                |              0.0245 |
|     30 |         26 |      295 |          0.2844 |          0.2481 |          0.0363 |            0.3306 |            0.2509 | original_loan_term                                     |              0.0227 |
|     31 |         25 |      224 |          0.2786 |          0.2456 |          0.0331 |            0.3195 |            0.2479 | dpd                                                    |              0.0137 |
|     32 |         24 |      331 |          0.2851 |          0.2468 |          0.0383 |            0.3302 |            0.2489 | n_roll_forward_l6m                                     |              0.0185 |
|     33 |         23 |      168 |          0.2782 |          0.2508 |          0.0274 |            0.3105 |             0.253 | n_curtail_l3m                                          |              0.0320 |
|     34 |         22 |      330 |          0.2829 |          0.2453 |          0.0376 |            0.3244 |            0.2476 | estimated_property_value                               |              0.0309 |
|     35 |         21 |      219 |          0.2784 |          0.2466 |          0.0318 |            0.3177 |            0.2481 | remaining_months_to_legal_maturity                     |              0.0379 |
|     36 |         20 |      198 |          0.2776 |          0.2469 |          0.0307 |            0.3148 |            0.2497 | property_valuation_method                              |              0.0473 |
|     37 |         19 |      257 |          0.2797 |           0.244 |          0.0357 |            0.3228 |            0.2452 | months_since_last_dpd60                                |              0.0498 |
|     38 |         18 |      236 |          0.2493 |            0.21 |          0.0394 |            0.2903 |            0.2113 | upb_pct_of_original                                    |              0.0512 |
|     39 |         17 |      441 |          0.2444 |          0.1922 |          0.0522 |            0.3007 |            0.1943 | ltv_x_dti                                              |              0.0652 |
|     40 |         16 |      360 |          0.2421 |          0.1955 |          0.0466 |            0.2924 |            0.1954 | occupancy_status                                       |              0.0651 |
|     41 |         15 |      182 |           0.232 |          0.1987 |          0.0333 |             0.268 |            0.1985 | loan_purpose                                           |              0.0721 |
|     42 |         14 |      220 |          0.2275 |          0.1905 |          0.0371 |            0.2691 |            0.1897 | metropolitan_statistical_area_or_metropolitan_division |              0.0844 |
|     43 |         13 |      296 |          0.2352 |          0.1989 |          0.0364 |            0.2748 |            0.1983 | seller_name                                            |              0.0878 |
|     44 |         12 |      173 |          0.2217 |          0.1985 |          0.0232 |            0.2496 |               0.2 | servicer_name                                          |              0.0839 |
|     45 |         11 |      441 |          0.2158 |          0.1864 |          0.0294 |            0.2581 |            0.1881 | n_curtail_l12m                                         |              0.1222 |
|     46 |         10 |      207 |          0.2199 |          0.1957 |          0.0242 |            0.2507 |            0.1967 | postal_code                                            |              0.1307 |
|     47 |          9 |      348 |          0.2112 |          0.2011 |          0.0102 |            0.2257 |            0.2023 | n_cure_l12m                                            |              0.1478 |
|     48 |          8 |      504 |          0.2109 |          0.1991 |          0.0118 |            0.2275 |            0.2016 | current_actual_upb                                     |              0.1608 |
|     49 |          7 |      290 |          0.1997 |          0.1932 |          0.0065 |            0.2091 |             0.195 | original_combined_loan_to_value                        |              0.1486 |
|     50 |          6 |      518 |          0.1991 |          0.1914 |          0.0077 |            0.2126 |            0.1937 | original_debt_to_income_ratio                          |              0.1902 |
|     51 |          5 |      332 |          0.1921 |          0.1875 |          0.0046 |            0.2018 |             0.189 | current_interest_rate                                  |              0.3057 |
|     52 |          4 |      123 |          0.1781 |           0.176 |          0.0021 |            0.1807 |             0.178 | number_of_borrowers                                    |              0.3718 |
|     53 |          3 |      131 |          0.1747 |          0.1728 |          0.0018 |            0.1764 |            0.1742 | upb_pct_of_original_roc_l12m                           |              0.4860 |
|     54 |          2 |       56 |          0.1381 |          0.1367 |          0.0014 |            0.1378 |            0.1374 | current_streak_l12m                                    |              0.3857 |
|     55 |          1 |       88 |          0.0265 |          0.0261 |          0.0004 |            0.0265 |            0.0261 | --                                                     |                  -- |

## Suggested features (38), by mean |SHAP|

| feature                                                |   mean \|SHAP\| |
|:-------------------------------------------------------|----------------:|
| classic_fico                                           |         0.50772 |
| number_of_borrowers                                    |         0.35571 |
| current_interest_rate                                  |         0.26136 |
| upb_pct_of_original_roc_l12m                           |         0.20639 |
| original_debt_to_income_ratio                          |         0.14282 |
| original_combined_loan_to_value                        |         0.11939 |
| n_cure_l12m                                            |         0.11936 |
| postal_code                                            |         0.10812 |
| n_curtail_l12m                                         |         0.10074 |
| current_actual_upb                                     |         0.09004 |
| servicer_name                                          |          0.0825 |
| upb_pct_of_original                                    |         0.07777 |
| current_streak_l12m                                    |         0.07611 |
| metropolitan_statistical_area_or_metropolitan_division |         0.07467 |
| occupancy_status                                       |         0.06449 |
| loan_purpose                                           |         0.05868 |
| ltv_x_dti                                              |         0.05417 |
| seller_name                                            |          0.0491 |
| property_valuation_method                              |         0.04535 |
| remaining_months_to_legal_maturity                     |         0.03797 |
| months_since_last_dpd60                                |         0.03615 |
| n_curtail_l3m                                          |         0.03038 |
| dpd                                                    |         0.02619 |
| estimated_property_value                               |         0.02357 |
| n_cure_l6m                                             |         0.02086 |
| property_value_roc_l12m                                |           0.019 |
| original_loan_term                                     |         0.01818 |
| current_streak_l6m                                     |         0.01572 |
| property_type                                          |         0.01326 |
| property_value_roc_l3m                                 |          0.0123 |
| property_state                                         |         0.01225 |
| n_roll_forward_l6m                                     |         0.01188 |
| current_streak_l3m                                     |         0.01153 |
| property_value_change                                  |         0.01053 |
| mortgage_insurance_percentage                          |         0.00975 |
| n_cure_l3m                                             |         0.00953 |
| servicer_changed_l12m                                  |         0.00567 |
| number_of_units                                        |         0.00395 |
