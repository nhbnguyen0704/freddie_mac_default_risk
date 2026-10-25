# Feature dictionary

Every column of the training set that can serve as a model input: the fields carried by the observations table (`src/jobs/observations_main.py`, with the origination table joined on) and the features engineered in `src/fe_main.py`; `first_payment_date`, `maturity_date` and `vintage` stay in the observations table but are dropped from the training set. Each row reads **feature_abbreviation - feature_meaning - feature_group_based_on_domain_context - logic**.

- All features are as of the observation month: nothing uses a month after it.
- `dpd` is months past due (0 current, 1 = 30+ days, 2 = 60+, ...), parsed at ingest from `current_loan_delinquency_status`: 'XX' becomes null and 'RA' (REO acquisition) becomes 999.
- **Window features** (`_l3m`, `_l6m`, `_l12m`) come from `fe_main.py`. Each is an aggregate over the loan's own performance rows from the observation month back n-1 calendar months (`rangeBetween(-(n-1), 0)` on the month index). They are read from the performance panel, not the observations table, so the history includes months the build excluded as observations. A missing reporting month shortens the window rather than stretching it, and a loan younger than n months gets a partial window.

**79 features**: 32 fields from the observations table and 47 engineered in `fe_main.py`.

## Summary by group

| Group | Features |
|---|---:|
| Delinquency | 27 |
| Balance & amortisation | 9 |
| Collateral & equity | 12 |
| Pricing & rate | 6 |
| Borrower credit | 5 |
| Loan product & terms | 8 |
| Property & geography | 6 |
| Origination & servicing parties | 4 |
| Tenure & seasoning | 2 |

## Delinquency

| feature_abbreviation | feature_meaning | feature_group | logic |
|---|---|---|---|
| `dpd_max_l3m` | Worst delinquency status in the last 3 months | Delinquency | max(dpd) over the window. An REO-acquisition month ('RA') is coded 999 at ingest and dominates the max (window of 3 calendar months) |
| `dpd_cnt30_l3m` | Number of months at 30+ days past due in the last 3 months | Delinquency | sum(dpd >= 1) over the window; a null status counts 0. Stored as tinyint (0-12) (window of 3 calendar months) |
| `dpd_cnt60_l3m` | Number of months at 60+ days past due in the last 3 months | Delinquency | sum(dpd >= 2) over the window; a null status counts 0; a 60+ month also counts in cnt30. Stored as tinyint (0-12) (window of 3 calendar months) |
| `dpd_max_l6m` | Worst delinquency status in the last 6 months | Delinquency | max(dpd) over the window. An REO-acquisition month ('RA') is coded 999 at ingest and dominates the max (window of 6 calendar months) |
| `dpd_cnt30_l6m` | Number of months at 30+ days past due in the last 6 months | Delinquency | sum(dpd >= 1) over the window; a null status counts 0. Stored as tinyint (0-12) (window of 6 calendar months) |
| `dpd_cnt60_l6m` | Number of months at 60+ days past due in the last 6 months | Delinquency | sum(dpd >= 2) over the window; a null status counts 0; a 60+ month also counts in cnt30. Stored as tinyint (0-12) (window of 6 calendar months) |
| `dpd_max_l12m` | Worst delinquency status in the last 12 months | Delinquency | max(dpd) over the window. An REO-acquisition month ('RA') is coded 999 at ingest and dominates the max (window of 12 calendar months) |
| `dpd_cnt30_l12m` | Number of months at 30+ days past due in the last 12 months | Delinquency | sum(dpd >= 1) over the window; a null status counts 0. Stored as tinyint (0-12) (window of 12 calendar months) |
| `dpd_cnt60_l12m` | Number of months at 60+ days past due in the last 12 months | Delinquency | sum(dpd >= 2) over the window; a null status counts 0; a 60+ month also counts in cnt30. Stored as tinyint (0-12) (window of 12 calendar months) |
| `dpd` | Delinquency status in the observation month, in months past due (0 current, 1 = 30-59 days, 2 = 60-89 days) | Delinquency | Raw panel field for the observation month, carried by fe_main.py. Months at or beyond the label's DPD threshold are excluded as observations (Rule 2), so the values seen are below it |
| `months_since_last_dpd30` | Months since the loan was last 30+ days past due, within the last 12 months (0 = this month; 12 = not in the last 12 months) | Delinquency | observation month - latest month with dpd >= 1 in T-11..T, else 12. Computed on the panel; stored as tinyint |
| `months_since_last_dpd60` | Months since the loan was last 60+ days past due, within the last 12 months (0 = this month; 12 = not in the last 12 months) | Delinquency | observation month - latest month with dpd >= 2 in T-11..T, else 12. Computed on the panel; stored as tinyint |
| `current_streak_l3m` | Consecutive months current, ending at the observation month, capped at 3 | Delinquency | Months back from T while dpd == 0; an unreported month, a null status or the start of the loan's history ends the streak; at most 3. Computed on the panel; stored as tinyint |
| `dpd30_streak_l3m` | Consecutive months 30+ days past due, ending at the observation month, capped at 3 | Delinquency | Months back from T while dpd >= 1; an unreported month, a null status or the start of the loan's history ends the streak; at most 3. Computed on the panel; stored as tinyint |
| `dpd60_streak_l3m` | Consecutive months 60+ days past due, ending at the observation month, capped at 3 | Delinquency | Months back from T while dpd >= 2; an unreported month, a null status or the start of the loan's history ends the streak; at most 3. Computed on the panel; stored as tinyint |
| `current_streak_l6m` | Consecutive months current, ending at the observation month, capped at 6 | Delinquency | Months back from T while dpd == 0; an unreported month, a null status or the start of the loan's history ends the streak; at most 6. Computed on the panel; stored as tinyint |
| `dpd30_streak_l6m` | Consecutive months 30+ days past due, ending at the observation month, capped at 6 | Delinquency | Months back from T while dpd >= 1; an unreported month, a null status or the start of the loan's history ends the streak; at most 6. Computed on the panel; stored as tinyint |
| `dpd60_streak_l6m` | Consecutive months 60+ days past due, ending at the observation month, capped at 6 | Delinquency | Months back from T while dpd >= 2; an unreported month, a null status or the start of the loan's history ends the streak; at most 6. Computed on the panel; stored as tinyint |
| `current_streak_l12m` | Consecutive months current, ending at the observation month, capped at 12 | Delinquency | Months back from T while dpd == 0; an unreported month, a null status or the start of the loan's history ends the streak; at most 12. Computed on the panel; stored as tinyint |
| `dpd30_streak_l12m` | Consecutive months 30+ days past due, ending at the observation month, capped at 12 | Delinquency | Months back from T while dpd >= 1; an unreported month, a null status or the start of the loan's history ends the streak; at most 12. Computed on the panel; stored as tinyint |
| `dpd60_streak_l12m` | Consecutive months 60+ days past due, ending at the observation month, capped at 12 | Delinquency | Months back from T while dpd >= 2; an unreported month, a null status or the start of the loan's history ends the streak; at most 12. Computed on the panel; stored as tinyint |
| `n_roll_forward_l3m` | Number of times the delinquency status worsened in the last 3 months | Delinquency | Moves where dpd went up (e.g. 0 to 1, 1 to 2). Counted over the n-1 month-to-month moves inside months T-(n-1)..T; a move needs both months reported. Computed on the panel; stored as tinyint |
| `n_cure_l3m` | Number of times the loan cured from delinquency to current in the last 3 months | Delinquency | Moves from dpd >= 1 to dpd == 0. Counted over the n-1 month-to-month moves inside months T-(n-1)..T; a move needs both months reported. Computed on the panel; stored as tinyint |
| `n_roll_forward_l6m` | Number of times the delinquency status worsened in the last 6 months | Delinquency | Moves where dpd went up (e.g. 0 to 1, 1 to 2). Counted over the n-1 month-to-month moves inside months T-(n-1)..T; a move needs both months reported. Computed on the panel; stored as tinyint |
| `n_cure_l6m` | Number of times the loan cured from delinquency to current in the last 6 months | Delinquency | Moves from dpd >= 1 to dpd == 0. Counted over the n-1 month-to-month moves inside months T-(n-1)..T; a move needs both months reported. Computed on the panel; stored as tinyint |
| `n_roll_forward_l12m` | Number of times the delinquency status worsened in the last 12 months | Delinquency | Moves where dpd went up (e.g. 0 to 1, 1 to 2). Counted over the n-1 month-to-month moves inside months T-(n-1)..T; a move needs both months reported. Computed on the panel; stored as tinyint |
| `n_cure_l12m` | Number of times the loan cured from delinquency to current in the last 12 months | Delinquency | Moves from dpd >= 1 to dpd == 0. Counted over the n-1 month-to-month moves inside months T-(n-1)..T; a move needs both months reported. Computed on the panel; stored as tinyint |

## Balance & amortisation

| feature_abbreviation | feature_meaning | feature_group | logic |
|---|---|---|---|
| `current_actual_upb` | Current unpaid principal balance | Balance & amortisation | Raw performance field for the observation month |
| `original_upb` | Loan amount at origination | Balance & amortisation | Raw Freddie Mac origination field, joined by loan_identifier; constant over the loan's life |
| `n_curtail_l3m` | Number of months with extra principal paid (curtailment) in the last 3 months | Balance & amortisation | Moves where the principal repaid (balance last month - balance this month) exceeded 1.5 x the scheduled principal, the level-payment amortisation of last month's balance at last month's rate over last month's remaining term. Only judged when both months are current (a catch-up is not a curtailment), both balances are positive (a payoff is not) and last month's loan age is 7 or more: Freddie Mac rounds the balance to the nearest $1,000 up to age 6. Counted over the n-1 month-to-month moves inside months T-(n-1)..T; a move needs both months reported. Computed on the panel; stored as tinyint |
| `n_curtail_l6m` | Number of months with extra principal paid (curtailment) in the last 6 months | Balance & amortisation | Moves where the principal repaid (balance last month - balance this month) exceeded 1.5 x the scheduled principal, the level-payment amortisation of last month's balance at last month's rate over last month's remaining term. Only judged when both months are current (a catch-up is not a curtailment), both balances are positive (a payoff is not) and last month's loan age is 7 or more: Freddie Mac rounds the balance to the nearest $1,000 up to age 6. Counted over the n-1 month-to-month moves inside months T-(n-1)..T; a move needs both months reported. Computed on the panel; stored as tinyint |
| `n_curtail_l12m` | Number of months with extra principal paid (curtailment) in the last 12 months | Balance & amortisation | Moves where the principal repaid (balance last month - balance this month) exceeded 1.5 x the scheduled principal, the level-payment amortisation of last month's balance at last month's rate over last month's remaining term. Only judged when both months are current (a catch-up is not a curtailment), both balances are positive (a payoff is not) and last month's loan age is 7 or more: Freddie Mac rounds the balance to the nearest $1,000 up to age 6. Counted over the n-1 month-to-month moves inside months T-(n-1)..T; a move needs both months reported. Computed on the panel; stored as tinyint |
| `upb_pct_of_original` | Share of the original loan amount still outstanding (%); 100 = nothing repaid, 0 = paid off | Balance & amortisation | current_actual_upb / original_upb * 100, null when original_upb is null or 0. Computed row by row after joining the window features onto the observations; stored as double |
| `upb_pct_of_original_roc_l3m` | Average monthly change in upb_pct_of_original over the last 3 months, in percentage points per month (negative = paying down; a sharp drop = curtailment) | Balance & amortisation | Mean of the 2 month-over-month changes inside months T-2..T (T vs T-1, T-1 vs T-2, ...), each change being (current_actual_upb in the month - current_actual_upb in the month before) / original_upb * 100. A change needs both months reported, so a gap drops the change on either side of it; a loan younger than the horizon averages the changes it has. A change also needs the earlier month's loan age to be 7 or more, because Freddie Mac rounds the balance to the nearest $1,000 up to age 6. Null when there is no change or when original_upb is null or 0. The mean balance change is computed on the panel and divided by original_upb after the join; stored as double |
| `upb_pct_of_original_roc_l6m` | Average monthly change in upb_pct_of_original over the last 6 months, in percentage points per month (negative = paying down; a sharp drop = curtailment) | Balance & amortisation | Mean of the 5 month-over-month changes inside months T-5..T (T vs T-1, T-1 vs T-2, ...), each change being (current_actual_upb in the month - current_actual_upb in the month before) / original_upb * 100. A change needs both months reported, so a gap drops the change on either side of it; a loan younger than the horizon averages the changes it has. A change also needs the earlier month's loan age to be 7 or more, because Freddie Mac rounds the balance to the nearest $1,000 up to age 6. Null when there is no change or when original_upb is null or 0. The mean balance change is computed on the panel and divided by original_upb after the join; stored as double |
| `upb_pct_of_original_roc_l12m` | Average monthly change in upb_pct_of_original over the last 12 months, in percentage points per month (negative = paying down; a sharp drop = curtailment) | Balance & amortisation | Mean of the 11 month-over-month changes inside months T-11..T (T vs T-1, T-1 vs T-2, ...), each change being (current_actual_upb in the month - current_actual_upb in the month before) / original_upb * 100. A change needs both months reported, so a gap drops the change on either side of it; a loan younger than the horizon averages the changes it has. A change also needs the earlier month's loan age to be 7 or more, because Freddie Mac rounds the balance to the nearest $1,000 up to age 6. Null when there is no change or when original_upb is null or 0. The mean balance change is computed on the panel and divided by original_upb after the join; stored as double |

## Collateral & equity

| feature_abbreviation | feature_meaning | feature_group | logic |
|---|---|---|---|
| `estimated_loan_to_value` | Estimated current LTV: current UPB / Freddie Mac's estimate of today's property value (%) | Collateral & equity | Raw performance field for the observation month; sentinel 999 set to null at ingest |
| `original_loan_to_value` | LTV at origination: loan amount / property value (%) | Collateral & equity | Raw Freddie Mac origination field, joined by loan_identifier; constant over the loan's life; sentinel 999 set to null |
| `original_combined_loan_to_value` | Combined LTV at origination, including secondary liens (%) | Collateral & equity | Raw Freddie Mac origination field, joined by loan_identifier; constant over the loan's life; sentinel 999 set to null |
| `mortgage_insurance_percentage` | Mortgage insurance coverage (%); 0 = no MI | Collateral & equity | Raw Freddie Mac origination field, joined by loan_identifier; constant over the loan's life; sentinel 999 set to null |
| `property_valuation_method` | How the property was valued (1 ACE waiver, 2 full appraisal, 3 other, 4 ACE + PDR, 9 n/a) | Collateral & equity | Raw Freddie Mac origination field, joined by loan_identifier; constant over the loan's life |
| `ltv_ratio_to_original` | Estimated current LTV relative to LTV at origination; below 1 = equity gained, above 1 = equity lost | Collateral & equity | estimated_loan_to_value / original_loan_to_value, null when original_loan_to_value is null or 0. Computed row by row after joining the window features onto the observations; stored as double |
| `negative_equity_flag` | 1 if the loan balance exceeds the estimated property value (estimated LTV above 100) | Collateral & equity | estimated_loan_to_value > 100, null when estimated_loan_to_value is null. Computed row by row after joining the window features onto the observations; stored as double |
| `estimated_property_value` | Property value implied by the estimated LTV, in dollars | Collateral & equity | current_actual_upb / estimated_loan_to_value * 100, null when either is null or 0 (a zero balance is a paid-off loan). ELTV is reported as a whole percent, so this carries up to 0.5 / ELTV rounding error: about 1% at ELTV 45, 5% at 10. Computed row by row after joining the window features onto the observations; stored as double |
| `property_value_change` | House-price change since origination as a fraction; 0.10 = the property is worth 10% more | Collateral & equity | estimated_property_value / original property value - 1, where original property value = original_upb / original_loan_to_value * 100; null when any input is null or 0. Both LTVs are whole percents, so the rounding error of both values applies. Computed row by row after joining the window features onto the observations; stored as double |
| `property_value_roc_l3m` | Average monthly change in the estimated property value over the last 3 months, in percentage points of the original property value per month (positive = appreciating) | Collateral & equity | Mean of the 2 month-over-month changes in estimated_property_value inside months T-2..T, divided by the original property value (original_upb / original_loan_to_value * 100) and * 100. A change needs both months to have a value, so an unreported month, a missing ELTV or a zero balance drops the change on either side of it. A change also needs the earlier month's loan age to be 7 or more, because Freddie Mac rounds the balance to the nearest $1,000 up to age 6. Null when there is no change or no original property value. The mean change is computed on the panel and divided after the join; stored as double. Month-to-month ELTV rounding makes the 3-month version noisy |
| `property_value_roc_l6m` | Average monthly change in the estimated property value over the last 6 months, in percentage points of the original property value per month (positive = appreciating) | Collateral & equity | Mean of the 5 month-over-month changes in estimated_property_value inside months T-5..T, divided by the original property value (original_upb / original_loan_to_value * 100) and * 100. A change needs both months to have a value, so an unreported month, a missing ELTV or a zero balance drops the change on either side of it. A change also needs the earlier month's loan age to be 7 or more, because Freddie Mac rounds the balance to the nearest $1,000 up to age 6. Null when there is no change or no original property value. The mean change is computed on the panel and divided after the join; stored as double. Month-to-month ELTV rounding makes the 3-month version noisy |
| `property_value_roc_l12m` | Average monthly change in the estimated property value over the last 12 months, in percentage points of the original property value per month (positive = appreciating) | Collateral & equity | Mean of the 11 month-over-month changes in estimated_property_value inside months T-11..T, divided by the original property value (original_upb / original_loan_to_value * 100) and * 100. A change needs both months to have a value, so an unreported month, a missing ELTV or a zero balance drops the change on either side of it. A change also needs the earlier month's loan age to be 7 or more, because Freddie Mac rounds the balance to the nearest $1,000 up to age 6. Null when there is no change or no original property value. The mean change is computed on the panel and divided after the join; stored as double. Month-to-month ELTV rounding makes the 3-month version noisy |

## Pricing & rate

| feature_abbreviation | feature_meaning | feature_group | logic |
|---|---|---|---|
| `current_interest_rate` | Note rate currently in effect | Pricing & rate | Raw performance field for the observation month |
| `original_interest_rate` | Note rate at origination | Pricing & rate | Raw Freddie Mac origination field, joined by loan_identifier; constant over the loan's life |
| `current_interest_rate_roc_l3m` | Average monthly change in the note rate over the last 3 months, in percentage points per month (positive = rate stepped up) | Pricing & rate | Mean of the 2 month-over-month changes in current_interest_rate inside months T-2..T. A change needs both months reported, so a gap drops the change on either side of it. Null when there is no change. Computed on the panel; stored as double. Zero for a fixed-rate loan: in a 2021-05 to 2021-06 sample only 0.03% of loans changed rate (step-rate or ARM resets, modifications) |
| `current_interest_rate_roc_l6m` | Average monthly change in the note rate over the last 6 months, in percentage points per month (positive = rate stepped up) | Pricing & rate | Mean of the 5 month-over-month changes in current_interest_rate inside months T-5..T. A change needs both months reported, so a gap drops the change on either side of it. Null when there is no change. Computed on the panel; stored as double. Zero for a fixed-rate loan: in a 2021-05 to 2021-06 sample only 0.03% of loans changed rate (step-rate or ARM resets, modifications) |
| `current_interest_rate_roc_l12m` | Average monthly change in the note rate over the last 12 months, in percentage points per month (positive = rate stepped up) | Pricing & rate | Mean of the 11 month-over-month changes in current_interest_rate inside months T-11..T. A change needs both months reported, so a gap drops the change on either side of it. Null when there is no change. Computed on the panel; stored as double. Zero for a fixed-rate loan: in a 2021-05 to 2021-06 sample only 0.03% of loans changed rate (step-rate or ARM resets, modifications) |
| `rate_gap` | Change in note rate since origination; non-zero means a step-rate or a modification, negative = refinance incentive | Pricing & rate | current_interest_rate - original_interest_rate. Computed row by row after joining the window features onto the observations; stored as double |

## Borrower credit

| feature_abbreviation | feature_meaning | feature_group | logic |
|---|---|---|---|
| `classic_fico` | Borrower credit score at origination (300-850) | Borrower credit | Raw Freddie Mac origination field, joined by loan_identifier; constant over the loan's life; sentinel 9999 set to null |
| `original_debt_to_income_ratio` | Debt-to-income ratio at origination (%) | Borrower credit | Raw Freddie Mac origination field, joined by loan_identifier; constant over the loan's life; sentinel 999 set to null |
| `number_of_borrowers` | Number of borrowers on the note | Borrower credit | Raw Freddie Mac origination field, joined by loan_identifier; constant over the loan's life; sentinel 99 set to null |
| `first_time_homebuyer_indicator` | First-time homebuyer (Y / N / 9 = not available) | Borrower credit | Raw Freddie Mac origination field, joined by loan_identifier; constant over the loan's life |
| `ltv_x_dti` | Leverage and affordability combined | Borrower credit | estimated_loan_to_value * original_debt_to_income_ratio / 100. Computed row by row after joining the window features onto the observations; stored as double |

## Loan product & terms

| feature_abbreviation | feature_meaning | feature_group | logic |
|---|---|---|---|
| `original_loan_term` | Scheduled term in months | Loan product & terms | Raw Freddie Mac origination field, joined by loan_identifier; constant over the loan's life |
| `amortization_type` | FRM (fixed) or ARM (adjustable) | Loan product & terms | Raw Freddie Mac origination field, joined by loan_identifier; constant over the loan's life |
| `interest_only_indicator` | Interest-only loan (Y/N) | Loan product & terms | Raw Freddie Mac origination field, joined by loan_identifier; constant over the loan's life |
| `prepayment_penalty_indicator` | Prepayment penalty applies (Y/N) | Loan product & terms | Raw Freddie Mac origination field, joined by loan_identifier; constant over the loan's life |
| `loan_purpose` | P purchase, C cash-out refinance, N no-cash-out refinance, R refinance not specified | Loan product & terms | Raw Freddie Mac origination field, joined by loan_identifier; constant over the loan's life |
| `super_conforming_flag` | Balance above the standard conforming limit but within the high-cost-area limit (Y) | Loan product & terms | Raw Freddie Mac origination field, joined by loan_identifier; constant over the loan's life |
| `harp_indicator` | HARP relief refinance (Y) | Loan product & terms | Raw Freddie Mac origination field, joined by loan_identifier; constant over the loan's life |
| `special_eligibility_program` | Affordable-lending program (H HFA Advantage, F HomeOne, R Home Possible, 9 n/a) | Loan product & terms | Raw Freddie Mac origination field, joined by loan_identifier; constant over the loan's life |

## Property & geography

| feature_abbreviation | feature_meaning | feature_group | logic |
|---|---|---|---|
| `occupancy_status` | P primary residence, I investment, S second home, 9 n/a | Property & geography | Raw Freddie Mac origination field, joined by loan_identifier; constant over the loan's life |
| `property_type` | SF single-family, CO condo, PU PUD, MH manufactured, CP co-op | Property & geography | Raw Freddie Mac origination field, joined by loan_identifier; constant over the loan's life |
| `number_of_units` | Number of dwelling units (1-4) | Property & geography | Raw Freddie Mac origination field, joined by loan_identifier; constant over the loan's life; sentinel 99 set to null |
| `property_state` | US state of the property | Property & geography | Raw Freddie Mac origination field, joined by loan_identifier; constant over the loan's life |
| `postal_code` | First three digits of the property ZIP code (masked as NNN00) | Property & geography | Raw Freddie Mac origination field, joined by loan_identifier; constant over the loan's life |
| `metropolitan_statistical_area_or_metropolitan_division` | MSA / metropolitan division code of the property | Property & geography | Raw Freddie Mac origination field, joined by loan_identifier; constant over the loan's life |

## Origination & servicing parties

| feature_abbreviation | feature_meaning | feature_group | logic |
|---|---|---|---|
| `servicer_name` | Company servicing the loan | Origination & servicing parties | Raw performance field for the observation month |
| `channel` | Origination channel (R retail, B broker, C correspondent, T TPO not specified, 9 n/a) | Origination & servicing parties | Raw Freddie Mac origination field, joined by loan_identifier; constant over the loan's life |
| `seller_name` | Company that sold the loan to Freddie Mac | Origination & servicing parties | Raw Freddie Mac origination field, joined by loan_identifier; constant over the loan's life |
| `servicer_changed_l12m` | 1 if the loan's servicer changed in the last 12 months | Origination & servicing parties | More than one distinct servicer_name in T-11..T. Freddie Mac names only its larger servicers and reports the rest as 'OTHER', so a transfer between two small servicers does not show and a servicer rename does. Computed on the panel; stored as tinyint |

## Tenure & seasoning

| feature_abbreviation | feature_meaning | feature_group | logic |
|---|---|---|---|
| `loan_age` | Months since the loan's first scheduled payment, as of the observation month | Tenure & seasoning | Raw Freddie Mac performance field for the observation month |
| `remaining_months_to_legal_maturity` | Months left until the loan's legal maturity date | Tenure & seasoning | Raw performance field for the observation month |

## Not model features

Columns in the tables that identify a row, carry the target, or record provenance.

| column | meaning | logic |
|---|---|---|
| `loan_identifier` | Loan ID | Key for the loan-month join |
| `period` / `observation_date` | Observation month | `observation_date` is a copy of `period`, used for time-based splits |
| `year`, `month` | Partition keys | Year and month of `period`, for partition pruning |
| `label` | Target: 1 = bad within the performance window | Set by the label spec in `io_helpers/label.py`: next bad month (DPD threshold, credit event, workout) after the observation month and within `window_months`, or bad in the final window month under `basis='at_end'` |
| `first_payment_date`, `maturity_date`, `vintage` | Origination dates and quarter | In the observations table only; dropped from the training set by `fe_main.py` (`NOT_FEATURES`) |
