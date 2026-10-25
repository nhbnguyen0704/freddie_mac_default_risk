# Error Analysis: Missed Bads on Test

_Generated 2026-10-11 17:32 by `src/jobs/19_error_analysis_main.py`._

Which test bads (after 2022-09; all 2,240,522 of them) each model's best version misses. A bad is **missed** (false negative) when it scores below the model's threshold, the one that maximizes F1 over every test row (rows scoring at or above it are flagged). Thresholds are on each model's own scale: LR logit, XGBoost margin, RF P(bad).

For each model a decision tree (depth 3, each leaf at least 1% of the bads) is fitted on every test bad to predict missed vs caught, on the 49 raw columns any of the three models reads, the same for all three. Categoricals are coded by the model's miss rate (categories with under 500 bads pooled as `__other__`, missing as `__missing__`), so a split is the best two-group partition of the categories; numerics keep missing values, routed by the tree. Tree AUC is in-sample over 2,240,522 rows with 8 leaves at most, so it is close to out-of-sample. Leaves are sorted by miss rate; **lift** is the leaf's miss rate over the model's overall miss rate (1 - recall).

## Thresholds (max F1 on every test row)

| model                           |   threshold |   flagged |   flagged share |   precision |   recall |     F1 |   missed bads |   tree AUC |
|:--------------------------------|------------:|----------:|----------------:|------------:|---------:|-------:|--------------:|-----------:|
| Logistic regression (recency_6) |      -2.474 | 1,914,013 |          0.0102 |      0.2593 |   0.2215 | 0.2389 |     1,744,238 |     0.9862 |
| XGBoost (recency_12)            |     -2.0753 | 1,755,975 |          0.0094 |      0.3186 |   0.2497 |   0.28 |     1,681,004 |     0.9426 |
| Random forest (undersample)     |      0.1605 | 1,692,696 |           0.009 |      0.3041 |   0.2298 | 0.2618 |     1,725,740 |     0.9843 |

## Missed by which models (test bads)

| missed by   |      bads |   share of bads |
|:------------|----------:|----------------:|
| lr, xgb, rf | 1,603,400 |          0.7156 |
| none        |   413,150 |          0.1844 |
| lr          |    64,509 |          0.0288 |
| lr, rf      |    58,602 |          0.0262 |
| xgb, rf     |    40,481 |          0.0181 |
| rf          |    23,257 |          0.0104 |
| xgb         |    19,396 |          0.0087 |
| lr, xgb     |    17,727 |          0.0079 |

## Logistic regression (recency_6): missed bads, overall miss rate 0.7785

| rule                                                                                                     |      bads |    missed |   miss rate |   lift |   share of misses |
|:---------------------------------------------------------------------------------------------------------|----------:|----------:|------------:|-------:|------------------:|
| dpd_cnt30_l3m <= 0.5 or missing AND classic_fico > 677.5 or missing AND n_cure_l12m <= 0.5 or missing    | 1,191,953 | 1,189,488 |      0.9979 | 1.2819 |            0.6820 |
| dpd_cnt30_l3m <= 0.5 or missing AND classic_fico <= 677.5 AND original_interest_rate <= 5.494 or missing |   295,087 |   289,153 |      0.9799 | 1.2587 |            0.1658 |
| dpd_cnt30_l3m <= 0.5 or missing AND classic_fico > 677.5 or missing AND n_cure_l12m > 0.5                |   114,299 |   110,494 |      0.9667 | 1.2418 |            0.0633 |
| dpd_cnt30_l3m <= 0.5 or missing AND classic_fico <= 677.5 AND original_interest_rate > 5.494             |   113,594 |    96,859 |      0.8527 | 1.0953 |            0.0555 |
| dpd_cnt30_l3m > 0.5 AND classic_fico > 758.5 AND ltv_x_dti <= 18.89                                      |    29,353 |    20,141 |      0.6862 | 0.8814 |            0.0115 |
| dpd_cnt30_l3m > 0.5 AND classic_fico > 758.5 AND ltv_x_dti > 18.89 or missing                            |    45,635 |    14,623 |      0.3204 | 0.4116 |            0.0084 |
| dpd_cnt30_l3m > 0.5 AND classic_fico <= 758.5 or missing AND number_of_borrowers > 1.5                   |   133,757 |    17,390 |      0.1300 | 0.1670 |            0.0100 |
| dpd_cnt30_l3m > 0.5 AND classic_fico <= 758.5 or missing AND number_of_borrowers <= 1.5 or missing       |   316,844 |     6,090 |      0.0192 | 0.0247 |            0.0035 |

```
|--- dpd_cnt30_l3m <= 0.5000
|   |--- classic_fico <= 677.5000
|   |   |--- original_interest_rate <= 5.4945
|   |   |   |--- class: True
|   |   |--- original_interest_rate >  5.4945
|   |   |   |--- class: True
|   |--- classic_fico >  677.5000
|   |   |--- n_cure_l12m <= 0.5000
|   |   |   |--- class: True
|   |   |--- n_cure_l12m >  0.5000
|   |   |   |--- class: True
|--- dpd_cnt30_l3m >  0.5000
|   |--- classic_fico <= 758.5000
|   |   |--- number_of_borrowers <= 1.5000
|   |   |   |--- class: False
|   |   |--- number_of_borrowers >  1.5000
|   |   |   |--- class: False
|   |--- classic_fico >  758.5000
|   |   |--- ltv_x_dti <= 18.8900
|   |   |   |--- class: True
|   |   |--- ltv_x_dti >  18.8900
|   |   |   |--- class: False
```

## XGBoost (recency_12): missed bads, overall miss rate 0.7503

| rule                                                                                                                                                                                              |      bads |    missed |   miss rate |   lift |   share of misses |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------:|----------:|------------:|-------:|------------------:|
| dpd_cnt30_l3m <= 0.5 or missing AND months_since_last_dpd60 > 11.5 or missing AND metropolitan_statistical_area_or_metropolitan_division in {48540, 46540, 35660, 38540, 21794, 40660, +364 more} | 1,634,466 | 1,582,160 |      0.9680 | 1.2902 |            0.9412 |
| dpd_cnt30_l3m <= 0.5 or missing AND months_since_last_dpd60 > 11.5 or missing AND metropolitan_statistical_area_or_metropolitan_division in {47764, 28880, 12054, 17410, 30500, 31924, +6 more}   |    49,164 |    26,430 |      0.5376 | 0.7165 |            0.0157 |
| dpd_cnt30_l3m > 0.5 AND dpd <= 0.5 AND months_since_last_dpd60 > 11.5 or missing                                                                                                                  |    92,317 |    40,985 |      0.4440 | 0.5917 |            0.0244 |
| dpd_cnt30_l3m <= 0.5 or missing AND months_since_last_dpd60 <= 11.5                                                                                                                               |    31,303 |     9,843 |      0.3144 | 0.4191 |            0.0059 |
| dpd_cnt30_l3m > 0.5 AND dpd > 0.5 or missing AND months_since_last_dpd60 > 11.5 or missing                                                                                                        |   213,877 |    20,383 |      0.0953 | 0.1270 |            0.0121 |
| dpd_cnt30_l3m > 0.5 AND dpd <= 0.5 AND months_since_last_dpd60 <= 11.5                                                                                                                            |    42,185 |     1,072 |      0.0254 | 0.0339 |            0.0006 |
| dpd_cnt30_l3m > 0.5 AND dpd > 0.5 or missing AND months_since_last_dpd60 <= 11.5                                                                                                                  |   177,210 |       131 |      0.0007 | 0.0010 |            0.0001 |

```
|--- dpd_cnt30_l3m <= 0.5000
|   |--- months_since_last_dpd60 <= 11.5000
|   |   |--- class: False
|   |--- months_since_last_dpd60 >  11.5000
|   |   |--- metropolitan_statistical_area_or_metropolitan_division <= 11.5000
|   |   |   |--- class: True
|   |   |--- metropolitan_statistical_area_or_metropolitan_division >  11.5000
|   |   |   |--- class: True
|--- dpd_cnt30_l3m >  0.5000
|   |--- dpd <= 0.5000
|   |   |--- months_since_last_dpd60 <= 11.5000
|   |   |   |--- class: False
|   |   |--- months_since_last_dpd60 >  11.5000
|   |   |   |--- class: False
|   |--- dpd >  0.5000
|   |   |--- months_since_last_dpd60 <= 11.5000
|   |   |   |--- class: False
|   |   |--- months_since_last_dpd60 >  11.5000
|   |   |   |--- class: False
```

## Random forest (undersample): missed bads, overall miss rate 0.7702

| rule                                                                                                           |      bads |    missed |   miss rate |   lift |   share of misses |
|:---------------------------------------------------------------------------------------------------------------|----------:|----------:|------------:|-------:|------------------:|
| dpd_cnt30_l3m <= 0.5 or missing AND months_since_last_dpd60 > 11.5 or missing AND n_cure_l6m <= 0.5 or missing | 1,617,558 | 1,606,609 |      0.9932 | 1.2895 |            0.9310 |
| dpd_cnt30_l3m <= 0.5 or missing AND months_since_last_dpd60 > 11.5 or missing AND n_cure_l6m > 0.5             |    66,072 |    56,104 |      0.8491 | 1.1024 |            0.0325 |
| dpd_cnt30_l3m > 0.5 AND dpd <= 0.5 AND months_since_last_dpd60 > 11.5 or missing                               |    92,317 |    47,689 |      0.5166 | 0.6707 |            0.0276 |
| dpd_cnt30_l3m <= 0.5 or missing AND months_since_last_dpd60 <= 11.5                                            |    31,303 |     6,912 |      0.2208 | 0.2867 |            0.0040 |
| dpd_cnt30_l3m > 0.5 AND dpd > 0.5 or missing AND current_interest_rate <= 3.107                                |    66,241 |     5,007 |      0.0756 | 0.0981 |            0.0029 |
| dpd_cnt30_l3m > 0.5 AND dpd > 0.5 or missing AND current_interest_rate > 3.107 or missing                      |   324,846 |     3,385 |      0.0104 | 0.0135 |            0.0020 |
| dpd_cnt30_l3m > 0.5 AND dpd <= 0.5 AND months_since_last_dpd60 <= 11.5                                         |    42,185 |        34 |      0.0008 | 0.0010 |            0.0000 |

```
|--- dpd_cnt30_l3m <= 0.5000
|   |--- months_since_last_dpd60 <= 11.5000
|   |   |--- class: False
|   |--- months_since_last_dpd60 >  11.5000
|   |   |--- n_cure_l6m <= 0.5000
|   |   |   |--- class: True
|   |   |--- n_cure_l6m >  0.5000
|   |   |   |--- class: True
|--- dpd_cnt30_l3m >  0.5000
|   |--- dpd <= 0.5000
|   |   |--- months_since_last_dpd60 <= 11.5000
|   |   |   |--- class: False
|   |   |--- months_since_last_dpd60 >  11.5000
|   |   |   |--- class: True
|   |--- dpd >  0.5000
|   |   |--- current_interest_rate <= 3.1065
|   |   |   |--- class: False
|   |   |--- current_interest_rate >  3.1065
|   |   |   |--- class: False
```
