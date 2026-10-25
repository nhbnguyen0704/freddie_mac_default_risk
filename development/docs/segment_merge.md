# Two-Segment Merge

_Generated 2026-10-14 19:04 by `src/jobs/22_segment_merge_main.py`._

One ranking from two models: **clean_regional** (job 20, fitted on clean rows, with the regional features) scores the clean rows, **all_rows_recency_12** (job 15, the current model) the delinquent rows. Clean = no 30+ DPD in the last 3 months and no 60+ in the last 12. Every valid and test row is scored.

- The two margins come from models fitted on different rows, so they need not be on one scale. **Raw** merges them as they are; **Platt (valid)** maps each segment's margin to a probability with a logistic fit on that segment's valid rows, the deployable version. The two **oracle** rows are fitted or chosen on test and only bound what a better calibration could reach.
- Within a segment, every merge keeps the model's ranking; only how the segments interleave changes.

## Average precision of the merged ranking

| ranking                                          |   valid AP |   test AP | test vs current   |
|:-------------------------------------------------|-----------:|----------:|:------------------|
| all_rows_recency_12 on every row (current)       |     0.2564 |    0.2174 | +0.0000           |
| merge, raw margins                               |     0.2565 |    0.2215 | +0.0041           |
| merge, Platt per segment fitted on valid         |     0.2569 |    0.2216 | +0.0043           |
| merge, Platt per segment fitted on test (oracle) |     0.2566 |    0.2215 | +0.0041           |
| merge, best offset on the clean margin (oracle)  |     0.2568 |    0.2216 | +0.0042           |

## AP within each segment

| rows             | bad rate   | model               |   AP within segment |
|:-----------------|:-----------|:--------------------|--------------------:|
| test/clean       | 0.9103%    | clean_regional      |              0.0782 |
| test/delinquent  | 22.1886%   | all_rows_recency_12 |              0.5698 |
| valid/clean      | 0.5879%    | clean_regional      |              0.0467 |
| valid/delinquent | 22.5262%   | all_rows_recency_12 |              0.5800 |

## Calibration: mean predicted probability vs bad rate

| split   | segment    | actual bad rate   | mean p, raw   | mean p, Platt (valid)   |
|:--------|:-----------|:------------------|:--------------|:------------------------|
| valid   | clean      | 0.5879%           | 0.5921%       | 0.5852%                 |
| valid   | delinquent | 22.5262%          | 23.3863%      | 22.5255%                |
| test    | clean      | 0.9103%           | 0.7387%       | 0.7216%                 |
| test    | delinquent | 22.1886%          | 21.8295%      | 21.1044%                |

## Offset added to the clean margin

|   clean offset |   valid AP |   test AP |
|---------------:|-----------:|----------:|
|             -3 |     0.2517 |    0.2122 |
|           -2.5 |     0.2525 |    0.2132 |
|             -2 |     0.2534 |    0.2147 |
|           -1.5 |     0.2545 |    0.2166 |
|             -1 |     0.2557 |    0.2188 |
|           -0.5 |     0.2566 |    0.2209 |
|              0 |     0.2565 |    0.2215 |
|            0.5 |     0.2536 |    0.2187 |
|              1 |     0.2451 |    0.2099 |
|            1.5 |     0.2283 |    0.1943 |
|              2 |     0.2025 |    0.1732 |
|            2.5 |     0.1695 |    0.1492 |
|              3 |     0.1327 |    0.1249 |
