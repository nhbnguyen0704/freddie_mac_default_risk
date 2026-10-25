from pyspark.sql import Column, DataFrame, functions as F

PERFORMANCE = ['loan_identifier', 'period', 'current_actual_upb', 'current_interest_rate',
               'remaining_months_to_legal_maturity', 'estimated_loan_to_value', 'servicer_name']
ORIGINATION = ['classic_fico', 'mortgage_insurance_percentage', 'number_of_units', 'original_combined_loan_to_value',
               'original_debt_to_income_ratio', 'original_loan_term', 'number_of_borrowers',
               'metropolitan_statistical_area_or_metropolitan_division', 'occupancy_status', 'property_state',
               'property_type', 'postal_code', 'loan_purpose', 'seller_name', 'property_valuation_method',
               'original_upb', 'original_loan_to_value']
WORKOUT = {'modification_flag': ['Y', 'P'], 'payment_deferral_flag': ['C', 'P'],
           'borrower_assistance_plan': ['F', 'R', 'T']}


def workout() -> Column:
    """A workout this month or earlier, as development's workout_condition(include_prior=True)."""
    cond = F.col('current_non_interest_bearing_upb') > 0
    for col, values in WORKOUT.items():
        cond = cond | F.col(col).isin(values)
    return F.coalesce(cond, F.lit(False))

def eligible(perf: DataFrame, t: int) -> DataFrame:
    """Loan-months of month index t to score: active, not 90+ DPD, never in a workout, no disaster flag in the history read."""
    disaster = (perf.where(F.col('delinquency_due_to_disaster') == 'Y')
                .select('loan_identifier').distinct())
    now = perf.where((F.year('period') * 12 + F.month('period')) == t)
    return (now.where(F.coalesce(F.trim('zero_balance_code'), F.lit('')) == '')
            .where(~F.coalesce(F.col('dpd') >= 3, F.lit(False)))
            .where(~workout())
            .join(disaster, 'loan_identifier', 'left_anti'))

def observations(perf: DataFrame, orig: DataFrame, t: int) -> DataFrame:
    """The eligible loan-months of t with their origination attributes."""
    return eligible(perf, t).select(*PERFORMANCE).join(orig.select('loan_identifier', *ORIGINATION),
                                                       'loan_identifier', 'left')
