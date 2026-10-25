# Regional Features in the All-Rows Model

_Generated 2026-10-14 17:44 by `src/jobs/21_regional_all_rows_main.py`._

Job 20's 15 regional delinquency features (3-digit zip, MSA and state of month t, from the raw performance tape; see `docs/regional_features.md`) added to the all-rows XGBoost model.

- **all_rows_recency_12** is job 15's model as it is. **all_rows_regional** is fitted on the same rows (every train bad and job 15's 2% draw of goods, weighted 50), with the same hyperparameters, recency weights (half-life 12 months) and early stopping on the same valid sample; only the features differ (38 vs 53). all_rows_regional stopped at 487 rounds (valid AP on the sample 0.2592).
- The hyperparameters were tuned for all_rows_recency_12 (best of 30 trials on the valid sample), so valid AP favours it; test is untouched by either.
- **Segments** as in `19_error_analysis_main`: clean = no 30+ DPD in the last 3 months and no 60+ in the last 12; delinquent = the rest. No-skill AP is the bad rate.
- Regional features take **4.2%** of all_rows_regional's total gain.

## Average precision on every valid and test row

| rows              | count       | bad rate   |   all_rows_recency_12 AP |   all_rows_regional AP | change   | relative   |
|:------------------|:------------|:-----------|-------------------------:|-----------------------:|:---------|:-----------|
| valid, all rows   | 115,353,351 | 0.8939%    |                   0.2564 |                 0.2573 | +0.0009  | +0.3%      |
| test, all rows    | 187,470,934 | 1.1951%    |                   0.2174 |                 0.2173 | -0.0001  | -0.0%      |
| valid, clean      | 113,744,502 | 0.5879%    |                   0.0460 |                 0.0475 | +0.0016  | +3.4%      |
| test, clean       | 184,961,128 | 0.9103%    |                   0.0721 |                 0.0753 | +0.0032  | +4.5%      |
| valid, delinquent | 1,608,849   | 22.5262%   |                   0.5800 |                 0.5795 | -0.0005  | -0.1%      |
| test, delinquent  | 2,509,806   | 22.1886%   |                   0.5698 |                 0.5625 | -0.0073  | -1.3%      |

## Clean rows: against the clean-only models of job 20

| model               | trained on          |   clean valid AP |   clean test AP |
|:--------------------|:--------------------|-----------------:|----------------:|
| clean_base          | clean rows (job 20) |           0.0448 |          0.0745 |
| clean_regional      | clean rows (job 20) |           0.0467 |          0.0782 |
| all_rows_recency_12 | every row           |            0.046 |          0.0721 |
| all_rows_regional   | every row           |           0.0475 |          0.0753 |

## all_rows_regional: gain share, top 25

| feature                                                |   gain share |   regional |
|:-------------------------------------------------------|-------------:|-----------:|
| current_streak_l3m                                     |       0.1739 |      False |
| classic_fico                                           |       0.1229 |      False |
| n_roll_forward_l6m                                     |       0.1159 |      False |
| postal_code                                            |       0.0839 |      False |
| metropolitan_statistical_area_or_metropolitan_division |       0.0527 |      False |
| n_cure_l12m                                            |       0.0526 |      False |
| current_interest_rate                                  |       0.0362 |      False |
| number_of_borrowers                                    |       0.0329 |      False |
| seller_name                                            |         0.03 |      False |
| months_since_last_dpd60                                |       0.0239 |      False |
| servicer_name                                          |       0.0229 |      False |
| upb_pct_of_original_roc_l12m                           |        0.021 |      False |
| ltv_x_dti                                              |       0.0189 |      False |
| upb_pct_of_original                                    |       0.0183 |      False |
| current_streak_l6m                                     |       0.0163 |      False |
| property_state                                         |       0.0142 |      False |
| original_combined_loan_to_value                        |       0.0141 |      False |
| original_debt_to_income_ratio                          |       0.0131 |      False |
| current_actual_upb                                     |       0.0127 |      False |
| estimated_property_value                               |       0.0114 |      False |
| dpd                                                    |        0.009 |      False |
| current_streak_l12m                                    |       0.0072 |      False |
| property_valuation_method                              |       0.0071 |      False |
| occupancy_status                                       |        0.006 |      False |
| n_cure_l3m                                             |        0.006 |      False |
