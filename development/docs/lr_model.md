
# Logistic Regression: Weighting Versions

_Generated 2026-10-04 18:37 by `src/jobs/10_lr_model_main.py`._

Logistic regression on the WoE features kept by `09_lr_vif_selection_main`, in 8 versions that differ only in how the train rows (2016-01-2022-09) are weighted. Each is fitted by maximum likelihood (Newton/IRLS) on every train row with its weights, then backward eliminated: drop the highest p-value (lowest IV on a tie) and refit until every p-value is at most 0.05; SMOTEENN and SMOTETomek, where present, are fitted on resampled rows instead (see the notes under All versions). Every version is scored, unchanged and unweighted, on all 576,268,072 train rows and all 187,470,934 test rows (after 2022-09).

Scored by average precision (AP), the step-wise area under the precision-recall curve (sklearn's average_precision_score). No-skill AP is the bad rate: train 0.0089, test 0.0120.

## Test average precision (AP), best first

| version     |   test AP | test vs baseline   |
|:------------|----------:|:-------------------|
| recency_6   |    0.1399 | +1.97%             |
| recency_12  |    0.1397 | +1.85%             |
| recency_25  |     0.139 | +1.33%             |
| oversample  |    0.1378 | +0.46%             |
| undersample |    0.1377 | +0.39%             |
| baseline    |    0.1372 | +0.00%             |
| smotetomek  |    0.1339 | -2.40%             |
| smoteenn    |    0.1269 | -7.50%             |

## All versions

| version     | weighting                                                           |   features |   train AP |   test AP |   train - test | test vs baseline   |   test mean PD |   weighted bad rate |   effective rows |
|:------------|:--------------------------------------------------------------------|-----------:|-----------:|----------:|---------------:|:-------------------|---------------:|--------------------:|-----------------:|
| baseline    | no weighting                                                        |         31 |     0.1683 |    0.1372 |         0.0311 | +0.00%             |         0.0081 |              0.0089 |      5.76268e+08 |
| oversample  | random oversampling of bads in lower-rate months                    |         30 |     0.1617 |    0.1378 |         0.0239 | +0.46%             |         0.0149 |              0.0146 |      5.72682e+08 |
| undersample | random undersampling of goods in lower-rate months                  |         31 |     0.1627 |    0.1377 |          0.025 | +0.39%             |         0.0148 |              0.0146 |      3.51136e+08 |
| recency_25  | recency weight, half-life 25 months                                 |         31 |     0.1639 |     0.139 |         0.0249 | +1.33%             |         0.0081 |              0.0076 |       4.1554e+08 |
| recency_12  | recency weight, half-life 12 months                                 |         30 |     0.1581 |    0.1397 |         0.0184 | +1.85%             |         0.0082 |              0.0072 |      2.65765e+08 |
| recency_6   | recency weight, half-life 6 months                                  |         30 |     0.1524 |    0.1399 |         0.0126 | +1.97%             |         0.0087 |              0.0073 |       1.6048e+08 |
| smoteenn    | SMOTE then edited nearest neighbours, per month, on every train row |         31 |     0.1458 |    0.1269 |         0.0189 | -7.50%             |         0.0952 |              0.1025 |      5.57321e+08 |
| smotetomek  | SMOTE then Tomek links, per month, on every train row               |         31 |     0.1541 |    0.1339 |         0.0202 | -2.40%             |         0.0755 |              0.0916 |      6.23197e+08 |

- Over/undersampling are exact random resamples, applied as frequency weights (0, 1, 2, ...) drawn from a seeded hash of (loan, month): the same rows every Newton scan. Both lift every lower month to the highest monthly train bad rate (1.4634%).
- Weighted versions use robust (sandwich) standard errors, so a duplicated bad counts once in the p-values; the baseline keeps the model-based ones.
- Resampling raises the fitted bad rate, so those versions' mean PD overstates the real bad rate. That moves the intercept, not the ranking: average precision is unaffected by it.
- Effective rows is Kish's (sum w)^2 / sum w^2: how many equally weighted rows the weights are worth.
- SMOTEENN / SMOTETomek (imblearn) run on every train row, one month at a time. SMOTE (k=5) works on all of the month's bads together and adds synthetic bads up to 0.1 bads per good, each snapped to the nearest real WoE value per feature. Then ENN (n_neighbors=3, kind_sel='all') or Tomek links removes majority-class rows, with the month cut into 10 seeded random chunks (one per Spark task) and neighbours searched within a chunk. They refit the baseline's final features with no elimination: p-values on synthetic rows are too small to select on. Their weighted rows and effective rows count the resampled rows, synthetic ones included: smoteenn 505,346,372 real + 51,974,460 synthetic; smotetomek 571,222,643 real + 51,974,460 synthetic.

## Version: baseline

no weighting. Intercept -4.650418; 31 features; standard errors model-based.

| feature                                                    |      coef |   std_err |      z |    p_value |      iv |   sign_ok |
|:-----------------------------------------------------------|----------:|----------:|-------:|-----------:|--------:|----------:|
| loan_age_woe                                               |    0.1307 |  0.001677 |  77.96 |          0 |  0.1933 |     False |
| remaining_months_to_legal_maturity_woe                     |   -0.8287 |  0.002948 | -281.1 |          0 | 0.03518 |      True |
| servicer_name_woe                                          |    -0.317 |  0.001867 | -169.8 |          0 | 0.06431 |      True |
| classic_fico_woe                                           |   -0.5402 | 0.0005984 | -902.7 |          0 |   0.724 |      True |
| metropolitan_statistical_area_or_metropolitan_division_woe |   -0.2454 |  0.002543 | -96.49 |          0 | 0.04191 |      True |
| mortgage_insurance_percentage_woe                          |   -0.2428 |  0.003138 | -77.38 |          0 | 0.03689 |      True |
| occupancy_status_woe                                       |   -0.9265 |   0.00438 | -211.5 |          0 | 0.01506 |      True |
| original_debt_to_income_ratio_woe                          |   -0.4088 |  0.001796 | -227.6 |          0 |  0.1164 |      True |
| original_upb_woe                                           |   -0.2844 |  0.001971 | -144.3 |          0 |  0.1328 |      True |
| original_loan_to_value_woe                                 |   -0.3937 |  0.002239 | -175.8 |          0 | 0.09469 |      True |
| original_interest_rate_woe                                 |   -0.4096 |  0.001082 | -378.7 |          0 |  0.5413 |      True |
| channel_woe                                                |  0.008865 |  0.001322 |  6.707 |  1.988e-11 |  0.1508 |     False |
| property_state_woe                                         |   -0.1739 |  0.002591 |  -67.1 |          0 | 0.04335 |      True |
| property_type_woe                                          |    -0.073 |  0.002717 | -26.87 | 5.151e-159 | 0.03479 |      True |
| loan_purpose_woe                                           |   -0.8088 |  0.004448 | -181.8 |          0 | 0.01537 |      True |
| original_loan_term_woe                                     |   -0.1115 |  0.002603 | -42.83 |          0 | 0.07107 |      True |
| number_of_borrowers_woe                                    |    -0.812 |  0.001397 | -581.2 |          0 |   0.129 |      True |
| seller_name_woe                                            |   -0.1056 |  0.002064 | -51.16 |          0 |  0.0558 |      True |
| harp_indicator_woe                                         |   0.05629 |   0.00416 |  13.53 |  1.001e-41 | 0.02383 |     False |
| property_valuation_method_woe                              |    0.4205 |  0.002491 |  168.8 |          0 | 0.06234 |     False |
| dpd_cnt30_l3m_woe                                          |   -0.5305 | 0.0007413 | -715.7 |          0 |   1.253 |      True |
| current_streak_l3m_woe                                     |   -0.2488 | 0.0008889 | -279.9 |          0 |  0.8335 |      True |
| n_cure_l6m_woe                                             |    0.2065 | 0.0007712 |  267.7 |          0 |  0.5685 |     False |
| n_cure_l12m_woe                                            |   -0.3954 | 0.0006864 | -576.1 |          0 |  0.7984 |      True |
| n_curtail_l3m_woe                                          |   -0.4471 |  0.002459 | -181.8 |          0 | 0.05911 |      True |
| ltv_x_dti_woe                                              |   -0.1829 |  0.002147 | -85.16 |          0 |  0.1261 |      True |
| upb_pct_of_original_woe                                    |   -0.8146 |   0.01574 | -51.77 |          0 |  0.1701 |      True |
| estimated_property_value_woe                               | -0.004326 |  0.001851 | -2.338 |     0.0194 |  0.1815 |      True |
| property_value_change_woe                                  |   -0.1316 |  0.002586 |  -50.9 |          0 | 0.06561 |      True |
| upb_pct_of_original_roc_l3m_woe                            |   -0.4201 | 0.0008377 | -501.5 |          0 |  0.3725 |      True |
| property_value_roc_l12m_woe                                |   -0.4032 |  0.004286 | -94.08 |          0 | 0.02313 |      True |

_No feature removed: every p-value was already within the threshold._

## Version: oversample

random oversampling of bads in lower-rate months. Intercept -4.121634; 30 features; standard errors robust (sandwich).

| feature                                                    |     coef |   std_err |      z |    p_value |      iv |   sign_ok |
|:-----------------------------------------------------------|---------:|----------:|-------:|-----------:|--------:|----------:|
| remaining_months_to_legal_maturity_woe                     |  -0.6765 |  0.003211 | -210.7 |          0 | 0.03518 |      True |
| servicer_name_woe                                          |  -0.1051 |   0.00198 | -53.06 |          0 | 0.06431 |      True |
| classic_fico_woe                                           |  -0.5683 | 0.0006839 |   -831 |          0 |   0.724 |      True |
| metropolitan_statistical_area_or_metropolitan_division_woe |  -0.2368 |  0.002786 | -85.01 |          0 | 0.04191 |      True |
| mortgage_insurance_percentage_woe                          |  -0.3346 |  0.003388 | -98.75 |          0 | 0.03689 |      True |
| occupancy_status_woe                                       |  -0.8503 |  0.004829 | -176.1 |          0 | 0.01506 |      True |
| original_debt_to_income_ratio_woe                          |  -0.4217 |  0.001928 | -218.7 |          0 |  0.1164 |      True |
| original_upb_woe                                           |  -0.4132 |  0.002182 | -189.4 |          0 |  0.1328 |      True |
| original_loan_to_value_woe                                 |  -0.4065 |  0.002441 | -166.5 |          0 | 0.09469 |      True |
| original_interest_rate_woe                                 |  -0.3145 |   0.00106 | -296.7 |          0 |  0.5413 |      True |
| channel_woe                                                |  0.05274 |  0.001359 |  38.81 |          0 |  0.1508 |     False |
| property_state_woe                                         |  -0.1848 |  0.002831 | -65.26 |          0 | 0.04335 |      True |
| property_type_woe                                          |  -0.1376 |   0.00296 | -46.47 |          0 | 0.03479 |      True |
| loan_purpose_woe                                           |  -0.9602 |  0.004871 | -197.1 |          0 | 0.01537 |      True |
| original_loan_term_woe                                     |    -0.23 |  0.002797 | -82.24 |          0 | 0.07107 |      True |
| number_of_borrowers_woe                                    |  -0.8817 |  0.001511 | -583.4 |          0 |   0.129 |      True |
| seller_name_woe                                            | -0.07263 |  0.002254 | -32.23 | 6.985e-228 |  0.0558 |      True |
| harp_indicator_woe                                         |   0.1873 |  0.004529 |  41.36 |          0 | 0.02383 |     False |
| property_valuation_method_woe                              |    0.657 |  0.002418 |  271.7 |          0 | 0.06234 |     False |
| dpd_cnt30_l3m_woe                                          |  -0.5047 |   0.00084 | -600.8 |          0 |   1.253 |      True |
| current_streak_l3m_woe                                     |  -0.2806 | 0.0008674 | -323.5 |          0 |  0.8335 |      True |
| n_cure_l6m_woe                                             |   0.2058 | 0.0009743 |  211.2 |          0 |  0.5685 |     False |
| n_cure_l12m_woe                                            |  -0.4041 | 0.0008132 | -496.9 |          0 |  0.7984 |      True |
| n_curtail_l3m_woe                                          |  -0.4385 |  0.002633 | -166.6 |          0 | 0.05911 |      True |
| ltv_x_dti_woe                                              |  -0.2204 |  0.002386 | -92.35 |          0 |  0.1261 |      True |
| upb_pct_of_original_woe                                    |  -0.6807 |  0.003347 | -203.3 |          0 |  0.1701 |      True |
| estimated_property_value_woe                               |   0.1126 |  0.002041 |  55.18 |          0 |  0.1815 |     False |
| property_value_change_woe                                  |  0.04796 |  0.003063 |  15.66 |  2.934e-55 | 0.06561 |     False |
| upb_pct_of_original_roc_l3m_woe                            |  -0.3737 | 0.0009327 | -400.6 |          0 |  0.3725 |      True |
| property_value_roc_l12m_woe                                |   0.1666 |  0.004912 |  33.91 | 5.182e-252 | 0.02313 |     False |

|   round | dropped      |   p_value |     iv |     coef |   tied_at_max |   features_left |
|--------:|:-------------|----------:|-------:|---------:|--------------:|----------------:|
|       1 | loan_age_woe |    0.0523 | 0.1933 | 0.003633 |             1 |              30 |

## Version: undersample

random undersampling of goods in lower-rate months. Intercept -4.158473; 31 features; standard errors robust (sandwich).

| feature                                                    |     coef |   std_err |      z |   p_value |      iv |   sign_ok |
|:-----------------------------------------------------------|---------:|----------:|-------:|----------:|--------:|----------:|
| loan_age_woe                                               |  0.01791 |  0.001799 |  9.952 |  2.46e-23 |  0.1933 |     False |
| remaining_months_to_legal_maturity_woe                     |  -0.7527 |  0.003141 | -239.6 |         0 | 0.03518 |      True |
| servicer_name_woe                                          |  -0.1171 |  0.001938 | -60.39 |         0 | 0.06431 |      True |
| classic_fico_woe                                           |  -0.5573 | 0.0006411 | -869.3 |         0 |   0.724 |      True |
| metropolitan_statistical_area_or_metropolitan_division_woe |  -0.2952 |  0.002644 | -111.7 |         0 | 0.04191 |      True |
| mortgage_insurance_percentage_woe                          |  -0.2982 |  0.003234 | -92.23 |         0 | 0.03689 |      True |
| occupancy_status_woe                                       |  -0.8288 |  0.004553 |   -182 |         0 | 0.01506 |      True |
| original_debt_to_income_ratio_woe                          |  -0.4336 |  0.001834 | -236.4 |         0 |  0.1164 |      True |
| original_upb_woe                                           |  -0.3597 |  0.002076 | -173.3 |         0 |  0.1328 |      True |
| original_loan_to_value_woe                                 |  -0.4316 |  0.002305 | -187.2 |         0 | 0.09469 |      True |
| original_interest_rate_woe                                 |  -0.3546 |  0.001148 |   -309 |         0 |  0.5413 |      True |
| channel_woe                                                |  0.04217 |  0.001434 |  29.41 | 4.38e-190 |  0.1508 |     False |
| property_state_woe                                         |  -0.2319 |  0.002663 | -87.09 |         0 | 0.04335 |      True |
| property_type_woe                                          |  -0.1125 |  0.002802 | -40.16 |         0 | 0.03479 |      True |
| loan_purpose_woe                                           |  -0.9194 |  0.004663 | -197.2 |         0 | 0.01537 |      True |
| original_loan_term_woe                                     |  -0.1783 |  0.002642 | -67.51 |         0 | 0.07107 |      True |
| number_of_borrowers_woe                                    |  -0.8506 |  0.001434 | -593.3 |         0 |   0.129 |      True |
| seller_name_woe                                            | -0.09131 |  0.002166 | -42.17 |         0 |  0.0558 |      True |
| harp_indicator_woe                                         |   0.1127 |  0.004342 |  25.96 | 1.42e-148 | 0.02383 |     False |
| property_valuation_method_woe                              |   0.7962 |  0.002327 |  342.1 |         0 | 0.06234 |     False |
| dpd_cnt30_l3m_woe                                          |  -0.4971 | 0.0008035 | -618.6 |         0 |   1.253 |      True |
| current_streak_l3m_woe                                     |  -0.2943 | 0.0008253 | -356.6 |         0 |  0.8335 |      True |
| n_cure_l6m_woe                                             |   0.2007 | 0.0009682 |  207.2 |         0 |  0.5685 |     False |
| n_cure_l12m_woe                                            |  -0.4069 | 0.0007994 |   -509 |         0 |  0.7984 |      True |
| n_curtail_l3m_woe                                          |  -0.4521 |  0.002453 | -184.3 |         0 | 0.05911 |      True |
| ltv_x_dti_woe                                              |  -0.1748 |   0.00231 | -75.69 |         0 |  0.1261 |      True |
| upb_pct_of_original_woe                                    |  -0.6822 |  0.004048 | -168.5 |         0 |  0.1701 |      True |
| estimated_property_value_woe                               |   0.1027 |  0.001968 |   52.2 |         0 |  0.1815 |     False |
| property_value_change_woe                                  |  0.03132 |  0.002988 |  10.48 | 1.032e-25 | 0.06561 |     False |
| upb_pct_of_original_roc_l3m_woe                            |  -0.4019 | 0.0009199 | -436.9 |         0 |  0.3725 |      True |
| property_value_roc_l12m_woe                                |   0.2126 |  0.004752 |  44.74 |         0 | 0.02313 |     False |

_No feature removed: every p-value was already within the threshold._

## Version: recency_25

recency weight, half-life 25 months. Intercept -4.654041; 31 features; standard errors robust (sandwich).

| feature                                                    |     coef |   std_err |      z |    p_value |      iv |   sign_ok |
|:-----------------------------------------------------------|---------:|----------:|-------:|-----------:|--------:|----------:|
| loan_age_woe                                               |   0.1565 |  0.002108 |  74.24 |          0 |  0.1933 |     False |
| remaining_months_to_legal_maturity_woe                     |  -0.5884 |  0.004052 | -145.2 |          0 | 0.03518 |      True |
| servicer_name_woe                                          |  -0.3341 |  0.002367 | -141.2 |          0 | 0.06431 |      True |
| classic_fico_woe                                           |  -0.6075 |  0.000786 | -772.9 |          0 |   0.724 |      True |
| metropolitan_statistical_area_or_metropolitan_division_woe |  -0.1417 |  0.003114 | -45.51 |          0 | 0.04191 |      True |
| mortgage_insurance_percentage_woe                          |  -0.3536 |  0.003871 | -91.34 |          0 | 0.03689 |      True |
| occupancy_status_woe                                       |   -1.029 |  0.005594 |   -184 |          0 | 0.01506 |      True |
| original_debt_to_income_ratio_woe                          |  -0.3957 |  0.002231 | -177.4 |          0 |  0.1164 |      True |
| original_upb_woe                                           |  -0.3392 |  0.002547 | -133.1 |          0 |  0.1328 |      True |
| original_loan_to_value_woe                                 |  -0.3408 |   0.00281 | -121.3 |          0 | 0.09469 |      True |
| original_interest_rate_woe                                 |  -0.3439 |  0.001387 | -247.9 |          0 |  0.5413 |      True |
| channel_woe                                                |  0.02926 |    0.0017 |  17.21 |  2.122e-66 |  0.1508 |     False |
| property_state_woe                                         | -0.06176 |  0.003206 | -19.26 |  1.125e-82 | 0.04335 |      True |
| property_type_woe                                          | -0.09921 |  0.003315 | -29.93 | 8.118e-197 | 0.03479 |      True |
| loan_purpose_woe                                           |   -1.011 |  0.005493 | -184.1 |          0 | 0.01537 |      True |
| original_loan_term_woe                                     |  -0.2207 |  0.003146 | -70.15 |          0 | 0.07107 |      True |
| number_of_borrowers_woe                                    |  -0.8894 |  0.001721 | -516.9 |          0 |   0.129 |      True |
| seller_name_woe                                            | -0.09607 |  0.002577 | -37.28 | 3.637e-304 |  0.0558 |      True |
| harp_indicator_woe                                         |   0.2885 |  0.005186 |  55.63 |          0 | 0.02383 |     False |
| property_valuation_method_woe                              |   0.3334 |  0.002638 |  126.4 |          0 | 0.06234 |     False |
| dpd_cnt30_l3m_woe                                          |  -0.5428 | 0.0009345 | -580.8 |          0 |   1.253 |      True |
| current_streak_l3m_woe                                     |  -0.2119 | 0.0009352 | -226.5 |          0 |  0.8335 |      True |
| n_cure_l6m_woe                                             |   0.2151 |  0.001098 |  195.9 |          0 |  0.5685 |     False |
| n_cure_l12m_woe                                            |  -0.3808 | 0.0009318 | -408.6 |          0 |  0.7984 |      True |
| n_curtail_l3m_woe                                          |  -0.4437 |  0.002936 | -151.1 |          0 | 0.05911 |      True |
| ltv_x_dti_woe                                              |  -0.2981 |  0.002686 |   -111 |          0 |  0.1261 |      True |
| upb_pct_of_original_woe                                    |  -0.7648 |  0.005847 | -130.8 |          0 |  0.1701 |      True |
| estimated_property_value_woe                               |  0.01952 |  0.002304 |  8.475 |  2.345e-17 |  0.1815 |     False |
| property_value_change_woe                                  | -0.04742 |  0.003355 | -14.13 |  2.392e-45 | 0.06561 |      True |
| upb_pct_of_original_roc_l3m_woe                            |  -0.3414 |  0.001101 |   -310 |          0 |  0.3725 |      True |
| property_value_roc_l12m_woe                                |  -0.4151 |  0.005329 |  -77.9 |          0 | 0.02313 |      True |

_No feature removed: every p-value was already within the threshold._

## Version: recency_12

recency weight, half-life 12 months. Intercept -4.667353; 30 features; standard errors robust (sandwich).

| feature                                                    |     coef |   std_err |      z |    p_value |      iv |   sign_ok |
|:-----------------------------------------------------------|---------:|----------:|-------:|-----------:|--------:|----------:|
| loan_age_woe                                               |    0.174 |  0.002876 |   60.5 |          0 |  0.1933 |     False |
| remaining_months_to_legal_maturity_woe                     |  -0.2964 |  0.006945 | -42.68 |          0 | 0.03518 |      True |
| servicer_name_woe                                          |  -0.3047 |  0.003573 | -85.29 |          0 | 0.06431 |      True |
| classic_fico_woe                                           |   -0.683 |  0.001055 | -647.1 |          0 |   0.724 |      True |
| metropolitan_statistical_area_or_metropolitan_division_woe |  -0.1052 |  0.004053 | -25.95 | 1.997e-148 | 0.04191 |      True |
| mortgage_insurance_percentage_woe                          |  -0.4334 |  0.005316 | -81.54 |          0 | 0.03689 |      True |
| occupancy_status_woe                                       |    -1.13 |   0.00782 | -144.6 |          0 | 0.01506 |      True |
| original_debt_to_income_ratio_woe                          |  -0.3662 |  0.003112 | -117.7 |          0 |  0.1164 |      True |
| original_upb_woe                                           |  -0.3998 |  0.003636 |   -110 |          0 |  0.1328 |      True |
| original_loan_to_value_woe                                 |  -0.3071 |  0.004051 | -75.81 |          0 | 0.09469 |      True |
| original_interest_rate_woe                                 |  -0.3085 |  0.001849 | -166.8 |          0 |  0.5413 |      True |
| channel_woe                                                |   0.0743 |  0.002569 |  28.92 | 6.091e-184 |  0.1508 |     False |
| property_type_woe                                          |  -0.1087 |  0.004299 | -25.29 | 4.242e-141 | 0.03479 |      True |
| loan_purpose_woe                                           |   -1.137 |   0.00712 | -159.7 |          0 | 0.01537 |      True |
| original_loan_term_woe                                     |  -0.2661 |   0.00427 | -62.33 |          0 | 0.07107 |      True |
| number_of_borrowers_woe                                    |  -0.9534 |  0.002317 | -411.5 |          0 |   0.129 |      True |
| seller_name_woe                                            | -0.08786 |  0.003367 | -26.09 | 4.454e-150 |  0.0558 |      True |
| harp_indicator_woe                                         |   0.5879 |  0.007489 |   78.5 |          0 | 0.02383 |     False |
| property_valuation_method_woe                              |   0.2384 |  0.003208 |   74.3 |          0 | 0.06234 |     False |
| dpd_cnt30_l3m_woe                                          |  -0.5371 |  0.001258 | -426.9 |          0 |   1.253 |      True |
| current_streak_l3m_woe                                     |  -0.1923 |  0.001184 | -162.5 |          0 |  0.8335 |      True |
| n_cure_l6m_woe                                             |   0.2166 |  0.001587 |  136.4 |          0 |  0.5685 |     False |
| n_cure_l12m_woe                                            |  -0.3726 |    0.0013 | -286.7 |          0 |  0.7984 |      True |
| n_curtail_l3m_woe                                          |  -0.4585 |  0.003877 | -118.3 |          0 | 0.05911 |      True |
| ltv_x_dti_woe                                              |  -0.4291 |  0.003723 | -115.3 |          0 |  0.1261 |      True |
| upb_pct_of_original_woe                                    |  -0.7397 |   0.00623 | -118.7 |          0 |  0.1701 |      True |
| estimated_property_value_woe                               |  0.06505 |  0.003163 |  20.57 |  5.523e-94 |  0.1815 |     False |
| property_value_change_woe                                  |  0.09197 |  0.005029 |  18.29 |  1.002e-74 | 0.06561 |     False |
| upb_pct_of_original_roc_l3m_woe                            |  -0.2817 |  0.001595 | -176.6 |          0 |  0.3725 |      True |
| property_value_roc_l12m_woe                                |  -0.4477 |  0.006811 | -65.73 |          0 | 0.02313 |      True |

|   round | dropped            |   p_value |      iv |      coef |   tied_at_max |   features_left |
|--------:|:-------------------|----------:|--------:|----------:|--------------:|----------------:|
|       1 | property_state_woe |    0.5912 | 0.04335 | -0.002268 |             1 |              30 |

## Version: recency_6

recency weight, half-life 6 months. Intercept -4.650219; 30 features; standard errors robust (sandwich).

| feature                                                    |     coef |   std_err |      z |    p_value |      iv |   sign_ok |
|:-----------------------------------------------------------|---------:|----------:|-------:|-----------:|--------:|----------:|
| loan_age_woe                                               |   0.2058 |  0.003807 |  54.05 |          0 |  0.1933 |     False |
| servicer_name_woe                                          |  -0.2737 |  0.004899 | -55.86 |          0 | 0.06431 |      True |
| classic_fico_woe                                           |  -0.7615 |  0.001394 | -546.4 |          0 |   0.724 |      True |
| metropolitan_statistical_area_or_metropolitan_division_woe |  -0.1068 |   0.00546 | -19.56 |  3.408e-85 | 0.04191 |      True |
| mortgage_insurance_percentage_woe                          |  -0.4758 |  0.007214 | -65.95 |          0 | 0.03689 |      True |
| occupancy_status_woe                                       |   -1.242 |   0.01096 | -113.3 |          0 | 0.01506 |      True |
| original_debt_to_income_ratio_woe                          |  -0.3345 |  0.004226 | -79.15 |          0 |  0.1164 |      True |
| original_upb_woe                                           |  -0.4189 |   0.00497 | -84.29 |          0 |  0.1328 |      True |
| original_loan_to_value_woe                                 |  -0.2993 |  0.005764 | -51.93 |          0 | 0.09469 |      True |
| original_interest_rate_woe                                 |  -0.3001 |  0.002335 | -128.5 |          0 |  0.5413 |      True |
| channel_woe                                                |   0.1144 |  0.003893 |  29.37 | 1.156e-189 |  0.1508 |     False |
| property_state_woe                                         |   0.0388 |  0.005472 |   7.09 |  1.341e-12 | 0.04335 |     False |
| property_type_woe                                          | -0.09337 |  0.005503 | -16.97 |  1.452e-64 | 0.03479 |      True |
| loan_purpose_woe                                           |   -1.186 |  0.009117 | -130.1 |          0 | 0.01537 |      True |
| original_loan_term_woe                                     |  -0.2316 |  0.005268 | -43.96 |          0 | 0.07107 |      True |
| number_of_borrowers_woe                                    |  -0.9941 |  0.003053 | -325.6 |          0 |   0.129 |      True |
| seller_name_woe                                            | -0.07837 |  0.004314 | -18.17 |  9.689e-74 |  0.0558 |      True |
| harp_indicator_woe                                         |   0.9094 |    0.0113 |  80.48 |          0 | 0.02383 |     False |
| property_valuation_method_woe                              |   0.1512 |  0.003971 |  38.08 |          0 | 0.06234 |     False |
| dpd_cnt30_l3m_woe                                          |  -0.5203 |    0.0017 | -306.1 |          0 |   1.253 |      True |
| current_streak_l3m_woe                                     |  -0.1833 |  0.001519 | -120.7 |          0 |  0.8335 |      True |
| n_cure_l6m_woe                                             |   0.2104 |  0.002277 |  92.37 |          0 |  0.5685 |     False |
| n_cure_l12m_woe                                            |  -0.3716 |  0.001808 | -205.6 |          0 |  0.7984 |      True |
| n_curtail_l3m_woe                                          |  -0.4875 |  0.005261 | -92.68 |          0 | 0.05911 |      True |
| ltv_x_dti_woe                                              |   -0.561 |  0.005128 | -109.4 |          0 |  0.1261 |      True |
| upb_pct_of_original_woe                                    |  -0.7213 |   0.00974 | -74.06 |          0 |  0.1701 |      True |
| estimated_property_value_woe                               |  0.09184 |  0.004464 |  20.57 |  4.856e-94 |  0.1815 |     False |
| property_value_change_woe                                  |   0.2575 |  0.007557 |  34.07 | 2.331e-254 | 0.06561 |     False |
| upb_pct_of_original_roc_l3m_woe                            |  -0.2599 |  0.002239 | -116.1 |          0 |  0.3725 |      True |
| property_value_roc_l12m_woe                                |  -0.5684 |   0.00846 | -67.18 |          0 | 0.02313 |      True |

|   round | dropped                                |   p_value |      iv |     coef |   tied_at_max |   features_left |
|--------:|:---------------------------------------|----------:|--------:|---------:|--------------:|----------------:|
|       1 | remaining_months_to_legal_maturity_woe |    0.3884 | 0.03518 | -0.01025 |             1 |              30 |

## Version: smoteenn

SMOTE then edited nearest neighbours, per month, on every train row. Intercept -1.785299; 31 features; standard errors model-based on resampled rows (optimistic).

| feature                                                    |     coef |   std_err |      z |   p_value |      iv |   sign_ok |
|:-----------------------------------------------------------|---------:|----------:|-------:|----------:|--------:|----------:|
| loan_age_woe                                               |  0.04878 | 0.0007048 |  69.21 |         0 |  0.1933 |     False |
| remaining_months_to_legal_maturity_woe                     |   -0.571 |  0.001381 | -413.5 |         0 | 0.03518 |      True |
| servicer_name_woe                                          |  -0.1536 | 0.0007771 | -197.7 |         0 | 0.06431 |      True |
| classic_fico_woe                                           |  -0.8025 | 0.0002277 |  -3524 |         0 |   0.724 |      True |
| metropolitan_statistical_area_or_metropolitan_division_woe |  -0.2107 | 0.0009895 |   -213 |         0 | 0.04191 |      True |
| mortgage_insurance_percentage_woe                          |  -0.2303 |  0.001216 | -189.5 |         0 | 0.03689 |      True |
| occupancy_status_woe                                       |   -1.705 |   0.00183 | -931.7 |         0 | 0.01506 |      True |
| original_debt_to_income_ratio_woe                          |  -0.5545 | 0.0007287 |   -761 |         0 |  0.1164 |      True |
| original_upb_woe                                           |  -0.4986 | 0.0008225 | -606.2 |         0 |  0.1328 |      True |
| original_loan_to_value_woe                                 |  -0.5874 | 0.0008928 | -657.9 |         0 | 0.09469 |      True |
| original_interest_rate_woe                                 |  -0.4491 | 0.0004239 |  -1059 |         0 |  0.5413 |      True |
| channel_woe                                                | -0.03434 | 0.0006078 |  -56.5 |         0 |  0.1508 |      True |
| property_state_woe                                         |  -0.4326 | 0.0009848 | -439.3 |         0 | 0.04335 |      True |
| property_type_woe                                          |  -0.3505 |  0.001009 | -347.4 |         0 | 0.03479 |      True |
| loan_purpose_woe                                           |   -1.303 |  0.001692 | -770.1 |         0 | 0.01537 |      True |
| original_loan_term_woe                                     |  -0.5215 | 0.0009909 | -526.2 |         0 | 0.07107 |      True |
| number_of_borrowers_woe                                    |   -1.206 | 0.0005309 |  -2271 |         0 |   0.129 |      True |
| seller_name_woe                                            | -0.01408 | 0.0008152 | -17.27 | 7.547e-67 |  0.0558 |      True |
| harp_indicator_woe                                         |   0.3795 |  0.001641 |  231.3 |         0 | 0.02383 |     False |
| property_valuation_method_woe                              |    0.838 | 0.0008081 |   1037 |         0 | 0.06234 |     False |
| dpd_cnt30_l3m_woe                                          |  -0.9591 |   0.00044 |  -2180 |         0 |   1.253 |      True |
| current_streak_l3m_woe                                     |  -0.2292 | 0.0003775 | -607.3 |         0 |  0.8335 |      True |
| n_cure_l6m_woe                                             | -0.03028 | 0.0005203 | -58.19 |         0 |  0.5685 |      True |
| n_cure_l12m_woe                                            |  -0.7659 | 0.0003703 |  -2069 |         0 |  0.7984 |      True |
| n_curtail_l3m_woe                                          |  -0.5622 | 0.0007964 | -705.9 |         0 | 0.05911 |      True |
| ltv_x_dti_woe                                              |   -0.393 | 0.0008412 | -467.2 |         0 |  0.1261 |      True |
| upb_pct_of_original_woe                                    |   -1.101 |  0.004081 | -269.8 |         0 |  0.1701 |      True |
| estimated_property_value_woe                               |   0.1426 |   0.00074 |  192.7 |         0 |  0.1815 |     False |
| property_value_change_woe                                  |   0.2605 |  0.001085 |    240 |         0 | 0.06561 |     False |
| upb_pct_of_original_roc_l3m_woe                            |  -0.3228 | 0.0004419 | -730.4 |         0 |  0.3725 |      True |
| property_value_roc_l12m_woe                                |  0.07067 |  0.001638 |  43.13 |         0 | 0.02313 |     False |

_No elimination: the baseline's final features, refitted on the resampled rows._

## Version: smotetomek

SMOTE then Tomek links, per month, on every train row. Intercept -2.224050; 31 features; standard errors model-based on resampled rows (optimistic).

| feature                                                    |       coef |   std_err |      z |    p_value |      iv |   sign_ok |
|:-----------------------------------------------------------|-----------:|----------:|-------:|-----------:|--------:|----------:|
| loan_age_woe                                               |    0.01354 | 0.0006297 |   21.5 | 1.545e-102 |  0.1933 |     False |
| remaining_months_to_legal_maturity_woe                     |    -0.6338 |  0.001193 | -531.4 |          0 | 0.03518 |      True |
| servicer_name_woe                                          |    -0.1297 | 0.0007149 | -181.5 |          0 | 0.06431 |      True |
| classic_fico_woe                                           |      -0.63 | 0.0002075 |  -3036 |          0 |   0.724 |      True |
| metropolitan_statistical_area_or_metropolitan_division_woe |    -0.1674 | 0.0009176 | -182.4 |          0 | 0.04191 |      True |
| mortgage_insurance_percentage_woe                          |    -0.1412 |  0.001125 | -125.4 |          0 | 0.03689 |      True |
| occupancy_status_woe                                       |     -1.532 |  0.001682 | -910.8 |          0 | 0.01506 |      True |
| original_debt_to_income_ratio_woe                          |    -0.4747 | 0.0006687 | -709.9 |          0 |  0.1164 |      True |
| original_upb_woe                                           |    -0.4093 | 0.0007513 | -544.8 |          0 |  0.1328 |      True |
| original_loan_to_value_woe                                 |     -0.508 | 0.0008236 | -616.9 |          0 | 0.09469 |      True |
| original_interest_rate_woe                                 |    -0.3357 | 0.0003883 | -864.4 |          0 |  0.5413 |      True |
| channel_woe                                                | -0.0006661 | 0.0005304 | -1.256 |     0.2091 |  0.1508 |      True |
| property_state_woe                                         |    -0.3811 | 0.0009165 | -415.8 |          0 | 0.04335 |      True |
| property_type_woe                                          |    -0.3232 | 0.0009448 | -342.1 |          0 | 0.03479 |      True |
| loan_purpose_woe                                           |     -1.093 |  0.001556 | -702.3 |          0 | 0.01537 |      True |
| original_loan_term_woe                                     |    -0.4438 | 0.0009106 | -487.3 |          0 | 0.07107 |      True |
| number_of_borrowers_woe                                    |    -0.9783 | 0.0004897 |  -1998 |          0 |   0.129 |      True |
| seller_name_woe                                            |   -0.01192 | 0.0007513 | -15.87 |  1.099e-56 |  0.0558 |      True |
| harp_indicator_woe                                         |     0.3328 |  0.001506 |    221 |          0 | 0.02383 |     False |
| property_valuation_method_woe                              |     0.6739 | 0.0007657 |  880.1 |          0 | 0.06234 |     False |
| dpd_cnt30_l3m_woe                                          |    -0.5274 |  0.000286 |  -1844 |          0 |   1.253 |      True |
| current_streak_l3m_woe                                     |    -0.2477 | 0.0003319 | -746.3 |          0 |  0.8335 |      True |
| n_cure_l6m_woe                                             |     0.1419 | 0.0003605 |  393.5 |          0 |  0.5685 |     False |
| n_cure_l12m_woe                                            |    -0.4876 | 0.0002855 |  -1708 |          0 |  0.7984 |      True |
| n_curtail_l3m_woe                                          |    -0.4656 | 0.0007683 |   -606 |          0 | 0.05911 |      True |
| ltv_x_dti_woe                                              |    -0.3324 | 0.0007693 | -432.1 |          0 |  0.1261 |      True |
| upb_pct_of_original_woe                                    |    -0.7839 |  0.003825 | -204.9 |          0 |  0.1701 |      True |
| estimated_property_value_woe                               |     0.1085 |  0.000676 |  160.5 |          0 |  0.1815 |     False |
| property_value_change_woe                                  |     0.2615 | 0.0009807 |  266.7 |          0 | 0.06561 |     False |
| upb_pct_of_original_roc_l3m_woe                            |    -0.3121 | 0.0003563 | -876.1 |          0 |  0.3725 |      True |
| property_value_roc_l12m_woe                                |     0.1008 |  0.001504 |  67.02 |          0 | 0.02313 |     False |

_No elimination: the baseline's final features, refitted on the resampled rows._
