# Random Forest Feature Selection: Permutation Backward Elimination

_Generated 2026-10-09 22:00 by `src/jobs/16_rf_perm_elimination_main.py`._

Backward elimination with a **random forest** (scikit-learn) on the **raw** features, from the 55 candidates left by `12_gbm_corr_selection_main` -- the split, candidates and sample are the XGBoost ones. Each step fits one forest, scores it, and drops the feature with the lowest **permutation importance** (lowest univariate AP on a tie), until 1 remain.

- **Fit** on the elimination sample of `13_gbm_shap_elimination_main`: every train bad and 2% of train goods (goods weighted 50), 13,239,314 rows standing for 460,703,345. Fixed parameters: n_estimators=300, criterion=gini, max_features=sqrt, min_samples_leaf=100, max_samples=1000000, bootstrap=True, random_state=42. Each tree is grown on a bootstrap of max_samples rows; the weights act inside it.
- **Strings** as codes ordered by their weighted train bad rate (missing a category of its own, unseen categories as missing), so a split on the code is a split of the categories into low- and high-rate groups.
- **Permutation importance**: weighted AP on 332,193 rows of the valid sample less the AP with the feature's column shuffled, mean of 3 shuffles (sd in the progress file). TreeSHAP is not used: sklearn's trees are grown to full depth and its cost is quadratic in it.
- **AP full valid**: the forest scored on **every** valid row (115,353,351) at every step, unweighted, P(bad) rounded to 1e-6. **AP full train** (every train row, 460,914,721) only for the suggested step: scoring sklearn forests is slow enough that train at every step would quadruple the run. AP sample: on the weighted samples (train AP is in-bag, so optimistic).

**Elbow**: best full valid AP 0.2412 with 25 features; the fewest features within 0.5% of it: **23** (valid AP 0.2403). This is a suggestion; the curve is below.

## Elimination curve

|   step |   features |   AP full train |   AP full valid |   train - valid |   AP sample train |   AP sample valid | dropped next                                           |   its importance |
|-------:|-----------:|----------------:|----------------:|----------------:|------------------:|------------------:|:-------------------------------------------------------|-----------------:|
|      1 |         55 |              -- |          0.2397 |              -- |            0.2439 |            0.2424 | property_value_roc_l3m                                 |          -0.0001 |
|      2 |         54 |              -- |          0.2396 |              -- |             0.244 |            0.2421 | servicer_changed_l12m                                  |          -0.0000 |
|      3 |         53 |              -- |          0.2401 |              -- |            0.2445 |            0.2427 | negative_equity_flag                                   |          -0.0000 |
|      4 |         52 |              -- |          0.2398 |              -- |            0.2442 |            0.2424 | property_value_roc_l6m                                 |          -0.0005 |
|      5 |         51 |              -- |          0.2402 |              -- |            0.2444 |            0.2428 | current_interest_rate_roc_l6m                          |          -0.0000 |
|      6 |         50 |              -- |          0.2401 |              -- |            0.2444 |             0.243 | property_state                                         |          -0.0000 |
|      7 |         49 |              -- |          0.2403 |              -- |            0.2445 |            0.2429 | property_valuation_method                              |          -0.0000 |
|      8 |         48 |              -- |          0.2396 |              -- |            0.2436 |            0.2424 | current_interest_rate_roc_l3m                          |          -0.0000 |
|      9 |         47 |              -- |          0.2395 |              -- |            0.2437 |            0.2422 | current_interest_rate_roc_l12m                         |          -0.0000 |
|     10 |         46 |              -- |          0.2395 |              -- |            0.2436 |            0.2423 | prepayment_penalty_indicator                           |          -0.0000 |
|     11 |         45 |              -- |            0.24 |              -- |            0.2442 |            0.2425 | super_conforming_flag                                  |          -0.0000 |
|     12 |         44 |              -- |          0.2401 |              -- |            0.2443 |            0.2428 | amortization_type                                      |           0.0000 |
|     13 |         43 |              -- |          0.2404 |              -- |            0.2448 |            0.2433 | interest_only_indicator                                |           0.0000 |
|     14 |         42 |              -- |          0.2403 |              -- |            0.2446 |            0.2431 | property_type                                          |          -0.0000 |
|     15 |         41 |              -- |          0.2399 |              -- |            0.2442 |            0.2426 | rate_gap                                               |           0.0000 |
|     16 |         40 |              -- |          0.2406 |              -- |            0.2447 |            0.2435 | number_of_units                                        |           0.0000 |
|     17 |         39 |              -- |          0.2407 |              -- |            0.2451 |            0.2436 | special_eligibility_program                            |           0.0000 |
|     18 |         38 |              -- |          0.2407 |              -- |             0.245 |            0.2436 | n_curtail_l3m                                          |           0.0000 |
|     19 |         37 |              -- |          0.2409 |              -- |            0.2452 |            0.2438 | first_time_homebuyer_indicator                         |           0.0000 |
|     20 |         36 |              -- |          0.2407 |              -- |            0.2451 |            0.2435 | harp_indicator                                         |           0.0000 |
|     21 |         35 |              -- |          0.2404 |              -- |            0.2446 |            0.2431 | original_loan_term                                     |           0.0001 |
|     22 |         34 |              -- |          0.2404 |              -- |            0.2447 |            0.2433 | mortgage_insurance_percentage                          |           0.0001 |
|     23 |         33 |              -- |          0.2405 |              -- |            0.2447 |            0.2432 | property_value_change                                  |           0.0001 |
|     24 |         32 |              -- |          0.2405 |              -- |            0.2447 |            0.2435 | occupancy_status                                       |           0.0002 |
|     25 |         31 |              -- |          0.2404 |              -- |            0.2448 |            0.2432 | loan_purpose                                           |           0.0002 |
|     26 |         30 |              -- |          0.2408 |              -- |            0.2451 |            0.2433 | metropolitan_statistical_area_or_metropolitan_division |           0.0003 |
|     27 |         29 |              -- |          0.2408 |              -- |            0.2451 |            0.2437 | channel                                                |           0.0003 |
|     28 |         28 |              -- |           0.241 |              -- |            0.2452 |            0.2441 | property_value_roc_l12m                                |           0.0004 |
|     29 |         27 |              -- |          0.2411 |              -- |            0.2454 |             0.244 | postal_code                                            |           0.0006 |
|     30 |         26 |              -- |           0.241 |              -- |            0.2451 |             0.244 | seller_name                                            |           0.0007 |
|     31 |         25 |              -- |          0.2412 |              -- |            0.2452 |             0.244 | n_curtail_l12m                                         |           0.0008 |
|     32 |         24 |              -- |          0.2403 |              -- |            0.2444 |            0.2433 | original_debt_to_income_ratio                          |           0.0011 |
|     33 |         23 |          0.2427 |          0.2403 |          0.0024 |            0.2441 |            0.2431 | servicer_name                                          |           0.0015 |
|     34 |         22 |              -- |          0.2397 |              -- |            0.2435 |            0.2426 | original_combined_loan_to_value                        |           0.0018 |
|     35 |         21 |              -- |           0.239 |              -- |            0.2426 |            0.2416 | n_cure_l3m                                             |           0.0021 |
|     36 |         20 |              -- |          0.2394 |              -- |             0.243 |             0.242 | estimated_property_value                               |           0.0028 |
|     37 |         19 |              -- |           0.239 |              -- |            0.2427 |            0.2416 | n_roll_forward_l3m                                     |           0.0024 |
|     38 |         18 |              -- |           0.239 |              -- |            0.2429 |            0.2419 | upb_pct_of_original                                    |           0.0024 |
|     39 |         17 |              -- |          0.2379 |              -- |            0.2417 |            0.2409 | current_actual_upb                                     |           0.0027 |
|     40 |         16 |              -- |          0.2369 |              -- |            0.2405 |            0.2398 | remaining_months_to_legal_maturity                     |           0.0036 |
|     41 |         15 |              -- |          0.2339 |              -- |            0.2372 |            0.2371 | ltv_x_dti                                              |           0.0043 |
|     42 |         14 |              -- |          0.2329 |              -- |            0.2361 |            0.2357 | number_of_borrowers                                    |           0.0046 |
|     43 |         13 |              -- |          0.2315 |              -- |            0.2346 |            0.2346 | n_cure_l6m                                             |           0.0045 |
|     44 |         12 |              -- |          0.2316 |              -- |            0.2349 |            0.2343 | dpd60_streak_l12m                                      |           0.0036 |
|     45 |         11 |              -- |          0.2321 |              -- |            0.2353 |            0.2344 | n_roll_forward_l6m                                     |           0.0107 |
|     46 |         10 |              -- |           0.231 |              -- |            0.2348 |            0.2333 | current_interest_rate                                  |           0.0117 |
|     47 |          9 |              -- |          0.2262 |              -- |            0.2296 |            0.2291 | current_streak_l6m                                     |           0.0108 |
|     48 |          8 |              -- |          0.2253 |              -- |            0.2285 |            0.2281 | classic_fico                                           |           0.0170 |
|     49 |          7 |              -- |          0.2168 |              -- |            0.2194 |            0.2199 | current_streak_l3m                                     |           0.0163 |
|     50 |          6 |              -- |           0.217 |              -- |            0.2197 |            0.2203 | dpd_cnt60_l3m                                          |           0.0213 |
|     51 |          5 |              -- |          0.2168 |              -- |            0.2198 |            0.2206 | upb_pct_of_original_roc_l12m                           |           0.0273 |
|     52 |          4 |              -- |           0.195 |              -- |            0.1966 |            0.1972 | current_streak_l12m                                    |           0.0250 |
|     53 |          3 |              -- |          0.1868 |              -- |            0.1885 |             0.189 | n_cure_l12m                                            |           0.0238 |
|     54 |          2 |              -- |          0.1606 |              -- |            0.1618 |            0.1627 | dpd                                                    |           0.0739 |
|     55 |          1 |              -- |          0.1238 |              -- |            0.1252 |            0.1259 | --                                                     |               -- |

## Suggested features (23), by permutation importance

| feature                            |   permutation importance |      sd | in XGBoost set   |
|:-----------------------------------|-------------------------:|--------:|:-----------------|
| months_since_last_dpd60            |                  0.03961 |  0.0034 | yes              |
| current_streak_l6m                 |                  0.02432 | 0.00108 | yes              |
| current_streak_l3m                 |                  0.02345 | 0.00123 | yes              |
| current_streak_l12m                |                  0.01939 | 0.00153 | yes              |
| dpd_cnt60_l3m                      |                  0.01809 | 0.00127 | no               |
| n_cure_l12m                        |                  0.01573 |   5e-05 | yes              |
| dpd                                |                  0.01335 | 0.00148 | yes              |
| classic_fico                       |                  0.01269 |  0.0003 | yes              |
| upb_pct_of_original_roc_l12m       |                  0.00975 | 0.00091 | yes              |
| dpd60_streak_l12m                  |                  0.00952 | 0.00106 | no               |
| n_roll_forward_l6m                 |                  0.00874 | 0.00042 | yes              |
| n_cure_l6m                         |                  0.00493 | 0.00023 | yes              |
| current_interest_rate              |                  0.00475 | 0.00054 | yes              |
| upb_pct_of_original                |                  0.00389 | 0.00066 | yes              |
| n_roll_forward_l3m                 |                   0.0034 |  0.0001 | no               |
| number_of_borrowers                |                  0.00285 |   9e-05 | yes              |
| remaining_months_to_legal_maturity |                  0.00201 |  0.0003 | yes              |
| original_combined_loan_to_value    |                  0.00193 | 0.00016 | yes              |
| ltv_x_dti                          |                  0.00187 | 0.00079 | yes              |
| current_actual_upb                 |                  0.00187 | 0.00076 | yes              |
| estimated_property_value           |                  0.00162 | 0.00033 | yes              |
| n_cure_l3m                         |                  0.00152 |   7e-05 | yes              |
| servicer_name                      |                  0.00148 |   8e-05 | yes              |

## Against the XGBoost selection

In both: 20. RF only: `dpd60_streak_l12m`, `dpd_cnt60_l3m`, `n_roll_forward_l3m`. XGBoost only: `loan_purpose`, `metropolitan_statistical_area_or_metropolitan_division`, `mortgage_insurance_percentage`, `n_curtail_l12m`, `n_curtail_l3m`, `number_of_units`, `occupancy_status`, `original_debt_to_income_ratio`, `original_loan_term`, `postal_code`, `property_state`, `property_type`, `property_valuation_method`, `property_value_change`, `property_value_roc_l12m`, `property_value_roc_l3m`, `seller_name`, `servicer_changed_l12m`.
