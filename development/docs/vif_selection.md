# Feature Selection: Correlation and VIF

_Generated 2026-09-29 23:30 by `src/jobs/09_lr_vif_selection_main.py`._

Feature selection on the WoE features, fitted on the **train** months only (2016-01-2022-09, 576,268,072 rows) and applied unchanged to the test months. Starting from 78 binned features:

0. **IV**: drop IV < 0.01 (25 dropped) and features with a constant WoE on train (0 dropped).
1. **Correlation**: while any pair has |corr| > 0.8, drop the feature with the highest max |corr|, then the highest median |corr|, then the lowest IV (21 dropped).
2. **VIF**: while any VIF > 5, drop the highest VIF, lowest IV on a tie (1 dropped).

**31 of 78 features kept.**

## Kept features

| feature                                                    |    vif |     iv |   max_abs_corr |
|:-----------------------------------------------------------|-------:|-------:|---------------:|
| estimated_property_value_woe                               | 3.3845 | 0.1815 |         0.6068 |
| ltv_x_dti_woe                                              | 3.0142 | 0.1261 |         0.5278 |
| upb_pct_of_original_roc_l3m_woe                            | 2.8335 | 0.3725 |         0.7098 |
| original_interest_rate_woe                                 | 2.8119 | 0.5413 |         0.6142 |
| original_upb_woe                                           | 2.6509 | 0.1328 |         0.6068 |
| n_cure_l6m_woe                                             | 2.6326 | 0.5685 |         0.7526 |
| upb_pct_of_original_woe                                    | 2.6253 | 0.1701 |         0.7098 |
| loan_age_woe                                               | 2.4727 | 0.1933 |         0.6142 |
| property_value_change_woe                                  | 2.3713 | 0.0656 |         0.5891 |
| n_cure_l12m_woe                                            | 2.3581 | 0.7984 |         0.7526 |
| original_loan_to_value_woe                                 | 2.2992 | 0.0947 |         0.5953 |
| dpd_cnt30_l3m_woe                                          | 2.0152 | 1.2526 |         0.5721 |
| original_loan_term_woe                                     | 1.9596 | 0.0711 |         0.3741 |
| property_value_roc_l12m_woe                                | 1.9277 | 0.0231 |         0.5891 |
| harp_indicator_woe                                         | 1.8615 | 0.0238 |         0.4833 |
| original_debt_to_income_ratio_woe                          | 1.8418 | 0.1164 |          0.498 |
| mortgage_insurance_percentage_woe                          | 1.7681 | 0.0369 |         0.5953 |
| remaining_months_to_legal_maturity_woe                     | 1.6219 | 0.0352 |         0.4277 |
| property_valuation_method_woe                              | 1.5503 | 0.0623 |         0.4532 |
| current_streak_l3m_woe                                     | 1.5135 | 0.8335 |         0.5259 |
| channel_woe                                                | 1.4776 | 0.1508 |         0.5188 |
| property_state_woe                                         | 1.2683 | 0.0434 |         0.3224 |
| loan_purpose_woe                                           | 1.2149 | 0.0154 |         0.3345 |
| metropolitan_statistical_area_or_metropolitan_division_woe | 1.2109 | 0.0419 |         0.3224 |
| classic_fico_woe                                           | 1.1632 |  0.724 |         0.2586 |
| servicer_name_woe                                          | 1.1294 | 0.0643 |         0.1799 |
| seller_name_woe                                            | 1.1177 | 0.0558 |         0.2598 |
| number_of_borrowers_woe                                    | 1.0709 |  0.129 |         0.1338 |
| occupancy_status_woe                                       | 1.0666 | 0.0151 |          0.115 |
| n_curtail_l3m_woe                                          | 1.0649 | 0.0591 |         0.0901 |
| property_type_woe                                          | 1.0579 | 0.0348 |         0.1827 |

## Correlation filter

|   round | dropped                             |   max_corr | with                             |   corr |   median_corr |     iv |   partner_iv |   features_left |
|--------:|:------------------------------------|-----------:|:---------------------------------|-------:|--------------:|-------:|-------------:|----------------:|
|       1 | dpd_max_l3m_woe                     |          1 | dpd_cnt30_l3m_woe                |      1 |        0.0278 | 1.2526 |       1.2526 |              52 |
|       2 | dpd_max_l6m_woe                     |          1 | dpd_cnt30_l6m_woe                |      1 |        0.0307 | 1.3081 |       1.3081 |              51 |
|       3 | current_interest_rate_woe           |          1 | original_interest_rate_woe       |      1 |        0.1135 |  0.541 |       0.5413 |              50 |
|       4 | current_actual_upb_woe              |          1 | upb_pct_of_original_woe          |      1 |        0.0142 |   0.17 |       0.1701 |              49 |
|       5 | current_streak_l12m_woe             |     0.9665 | current_streak_l6m_woe           | 0.9665 |        0.0976 | 0.8887 |        0.869 |              48 |
|       6 | months_since_last_dpd30_woe         |     0.9621 | dpd_max_l12m_woe                 | 0.9621 |        0.0334 | 1.4587 |       1.3176 |              47 |
|       7 | original_combined_loan_to_value_woe |     0.9617 | original_loan_to_value_woe       | 0.9617 |        0.0452 | 0.1087 |       0.0947 |              46 |
|       8 | dpd_cnt30_l12m_woe                  |     0.9592 | dpd_max_l12m_woe                 | 0.9592 |        0.0349 | 1.4706 |       1.3176 |              45 |
|       9 | dpd_max_l12m_woe                    |     0.9577 | n_roll_forward_l12m_woe          | 0.9577 |        0.0317 | 1.3176 |       1.2672 |              44 |
|      10 | ltv_ratio_to_original_woe           |     0.9543 | negative_equity_flag_woe         | 0.9543 |        0.0721 | 0.0424 |        0.038 |              43 |
|      11 | upb_pct_of_original_roc_l6m_woe     |     0.9536 | upb_pct_of_original_roc_l12m_woe | 0.9536 |        0.0433 | 0.3158 |       0.2554 |              42 |
|      12 | n_curtail_l12m_woe                  |      0.946 | n_curtail_l6m_woe                |  0.946 |        0.0353 | 0.0558 |       0.0648 |              41 |
|      13 | current_streak_l6m_woe              |     0.9442 | current_streak_l3m_woe           | 0.9442 |        0.0645 |  0.869 |       0.8335 |              40 |
|      14 | property_value_roc_l6m_woe          |     0.9287 | property_value_roc_l12m_woe      | 0.9287 |        0.0275 | 0.0252 |       0.0231 |              39 |
|      15 | n_curtail_l6m_woe                   |     0.9121 | n_curtail_l3m_woe                | 0.9121 |        0.0341 | 0.0648 |       0.0591 |              38 |
|      16 | dpd_cnt30_l6m_woe                   |     0.9039 | n_roll_forward_l6m_woe           | 0.9039 |        0.0274 | 1.3081 |       1.1417 |              37 |
|      17 | n_roll_forward_l12m_woe             |     0.8862 | n_cure_l12m_woe                  | 0.8862 |        0.0281 | 1.2672 |       0.7984 |              36 |
|      18 | upb_pct_of_original_roc_l12m_woe    |     0.8817 | upb_pct_of_original_roc_l3m_woe  | 0.8817 |        0.0407 | 0.2554 |       0.3725 |              35 |
|      19 | property_value_roc_l3m_woe          |     0.8597 | property_value_roc_l12m_woe      | 0.8597 |         0.032 |  0.023 |       0.0231 |              34 |
|      20 | n_roll_forward_l6m_woe              |     0.8344 | dpd_cnt30_l3m_woe                | 0.8344 |        0.0227 | 1.1417 |       1.2526 |              33 |
|      21 | negative_equity_flag_woe            |     0.8014 | property_value_change_woe        | 0.8014 |        0.0707 |  0.038 |       0.0656 |              32 |

## VIF filter

|   round | dropped                     |    vif |     iv |   tied_at_max |   features_left |
|--------:|:----------------------------|-------:|-------:|--------------:|----------------:|
|       1 | estimated_loan_to_value_woe | 6.3295 | 0.0829 |             1 |              31 |

## Dropped for IV < 0.01

| feature                            |     iv |
|:-----------------------------------|-------:|
| super_conforming_flag_woe          |  0.009 |
| special_eligibility_program_woe    | 0.0083 |
| servicer_changed_l12m_woe          | 0.0053 |
| first_time_homebuyer_indicator_woe | 0.0043 |
| current_interest_rate_roc_l12m_woe | 0.0022 |
| current_interest_rate_roc_l6m_woe  | 0.0022 |
| current_interest_rate_roc_l3m_woe  | 0.0022 |
| prepayment_penalty_indicator_woe   | 0.0014 |
| number_of_units_woe                |      0 |
| amortization_type_woe              |      0 |
| dpd30_streak_l12m_woe              |      0 |
| dpd_cnt60_l6m_woe                  |      0 |
| rate_gap_woe                       |      0 |
| dpd30_streak_l6m_woe               |      0 |
| n_roll_forward_l3m_woe             |      0 |
| interest_only_indicator_woe        |      0 |
| dpd_cnt60_l3m_woe                  |      0 |
| dpd_cnt60_l12m_woe                 |      0 |
| dpd60_streak_l6m_woe               |      0 |
| dpd60_streak_l12m_woe              |      0 |
| dpd_woe                            |      0 |
| months_since_last_dpd60_woe        |      0 |
| dpd30_streak_l3m_woe               |      0 |
| dpd60_streak_l3m_woe               |      0 |
| n_cure_l3m_woe                     |      0 |
