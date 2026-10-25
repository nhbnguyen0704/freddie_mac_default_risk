# Regional Features on the Clean Segment

_Generated 2026-10-14 13:53 by `src/jobs/20_regional_features_main.py`._

Do regional delinquency features help rank the **clean** loans (no 30+ DPD in the last 3 months, no 60+ in the last 12), where `19_error_analysis_main` found the models miss 97-99% of bads? Clean rows: valid 113,744,502 (bad rate 0.5879%), test 184,961,128 (bad rate 0.9103%); no-skill AP is the bad rate.

- **Regional features** of month t, for the 3-digit zip, the MSA and the state, over every loan active in t on the raw performance tape: active loans, share 30+ DPD, share 90+ DPD, share newly 30 DPD (averaged over t-2..t) and the 6-month change of the 30+ share. Month t is not leakage: a kept row is not bad in t, so its label covers t+1..t+25. The raw tape, not training_set, because training_set drops rows on what happens after t.
- **Clean models**: XGBoost with `recency_12`'s hyperparameters and recency weights (half-life 12 months), on every clean train bad and job 15's 2% draw of clean goods (weighted 50), early stopped on the clean valid sample; the same with the 15 regional features added. Hyperparameters are not re-tuned for either, so the comparison is like for like.
- **all_rows_recency_12** is the current XGBoost model (fitted on every row), scored on the same clean rows.
- Regional features take **6.5%** of the clean_regional model's total gain.

## Average precision on every clean row

| model               | trained on         |   features |   clean valid AP |   clean test AP | test vs all-rows model   |
|:--------------------|:-------------------|-----------:|-----------------:|----------------:|:-------------------------|
| all_rows_recency_12 | every row (job 15) |         38 |           0.0460 |          0.0721 | +0.0000                  |
| clean_base          | clean rows         |         38 |           0.0448 |          0.0745 | +0.0024                  |
| clean_regional      | clean rows         |         53 |           0.0467 |          0.0782 | +0.0062                  |

## clean_regional: gain share, top 25

| feature                                                |   gain share |   regional |
|:-------------------------------------------------------|-------------:|-----------:|
| classic_fico                                           |       0.1925 |      False |
| postal_code                                            |         0.13 |      False |
| metropolitan_statistical_area_or_metropolitan_division |       0.0861 |      False |
| number_of_borrowers                                    |        0.054 |      False |
| current_interest_rate                                  |       0.0533 |      False |
| seller_name                                            |       0.0501 |      False |
| servicer_name                                          |       0.0369 |      False |
| n_cure_l12m                                            |       0.0353 |      False |
| ltv_x_dti                                              |       0.0335 |      False |
| upb_pct_of_original_roc_l12m                           |       0.0325 |      False |
| upb_pct_of_original                                    |       0.0266 |      False |
| original_combined_loan_to_value                        |       0.0262 |      False |
| original_debt_to_income_ratio                          |       0.0249 |      False |
| property_state                                         |       0.0242 |      False |
| current_actual_upb                                     |       0.0238 |      False |
| estimated_property_value                               |       0.0174 |      False |
| property_valuation_method                              |       0.0117 |      False |
| loan_purpose                                           |       0.0096 |      False |
| occupancy_status                                       |       0.0092 |      False |
| remaining_months_to_legal_maturity                     |       0.0089 |      False |
| state_dpd90_rate                                       |       0.0084 |       True |
| msa_loans                                              |       0.0073 |       True |
| property_value_change                                  |       0.0064 |      False |
| msa_dpd90_rate                                         |       0.0061 |       True |
| state_dpd30_rate                                       |        0.006 |       True |
