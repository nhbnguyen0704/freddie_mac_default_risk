import re
from pathlib import Path
from typing import Dict, List, Optional, Set
import pandas as pd

ORIGINATION_FIELDS: List[str] = ['classic_fico', 'first_payment_date', 'first_time_homebuyer_indicator',
                                 'maturity_date', 'metropolitan_statistical_area_or_metropolitan_division',
                                 'mortgage_insurance_percentage', 'number_of_units', 'occupancy_status',
                                 'original_combined_loan_to_value', 'original_debt_to_income_ratio', 'original_upb',
                                 'original_loan_to_value', 'original_interest_rate', 'channel',
                                 'prepayment_penalty_indicator', 'amortization_type', 'property_state',
                                 'property_type', 'postal_code', 'loan_identifier', 'loan_purpose',
                                 'original_loan_term', 'number_of_borrowers', 'seller_name', 'super_conforming_flag',
                                 'pre_harp_loan_sequence_number', 'special_eligibility_program', 'harp_indicator',
                                 'property_valuation_method', 'interest_only_indicator', 'vantage_score',]

PERFORMANCE_FIELDS: List[str] = ['loan_identifier', 'period', 'current_actual_upb', 'current_loan_delinquency_status',
                                 'loan_age', 'remaining_months_to_legal_maturity',
                                 'underwriting_defect_and_major_servicing_defect_settlement_date', 'modification_flag',
                                 'zero_balance_code', 'zero_balance_effective_date', 'current_interest_rate',
                                 'current_non_interest_bearing_upb', 'due_date_of_last_paid_installment',
                                 'mi_recoveries', 'net_sales_proceeds', 'non_mi_recoveries', 'total_expenses',
                                 'legal_costs', 'maintenance_and_preservation_costs', 'taxes_and_insurance',
                                 'miscellaneous', 'actual_loss', 'cumulative_modification_costs',
                                 'interest_rate_step_indicator', 'payment_deferral_flag', 'estimated_loan_to_value',
                                 'zero_balance_removal_upb', 'delinquent_accrued_interest', 'delinquency_due_to_disaster',
                                 'borrower_assistance_plan', 'current_perid_modification_costs',
                                 'current_interest_bearing_upb', 'mortgage_insurance_cancellation_indicator',
                                 'servicer_name', 'bankruptcy_cramdown_costs',]

N_ORIG = len(ORIGINATION_FIELDS)
N_PERF = len(PERFORMANCE_FIELDS)

DATE_FORMAT = '%Y%m'
ORIG_DATE_COLS = ['first_payment_date', 'maturity_date',]
PERF_DATE_COLS = ['period', 'underwriting_defect_and_major_servicing_defect_settlement_date',
                  'zero_balance_effective_date', 'due_date_of_last_paid_installment']

DELINQUENCY_SPECIAL = {"RA"}
DELINQUENCY_MISSING = {"XX"}

REO_ACQUISITION_DPD = 999

ZERO_BALANCE_LABELS: Dict[str, str] = {
    "01": "Prepaid or Matured (Voluntary Payoff)",
    "02": "Third Party Sale",
    "03": "Short Sale or Charge Off",
    "09": "REO Disposition",
    "15": "Whole Loan sales",
    "16": "Reperforming loan securitizations",
    "96": "Confirmed Underwriting Defect or Major Servicing Defect prior to credit event"
}
ZERO_BALANCE_DOMAIN = set(ZERO_BALANCE_LABELS)
CREDIT_EVENT_CODES = {'02', '03', '09'}
PREPAID_CODES = {'01'}
CENSORING_CODES = {'15', '16', '96'}

assert not (CREDIT_EVENT_CODES & PREPAID_CODES)
assert not (CREDIT_EVENT_CODES & CENSORING_CODES)
assert not (PREPAID_CODES & CENSORING_CODES)
assert CREDIT_EVENT_CODES | PREPAID_CODES | CENSORING_CODES == set(ZERO_BALANCE_LABELS)

BORROWER_ASSISTANCE_CODES: Dict[str, str] = {
    'F': "Forbearance",
    'R': 'Repayment',
    'T': 'Trial Period'
}

SPECIAL_MENTION_FLAGS: Dict[str, Set[str]] = {
    'modification_flag': {'Y'},
    'payment_deferral_flag': {'C'},
    'borrower_assistance_plan': set(BORROWER_ASSISTANCE_CODES),
}

SPECIAL_MENTION_PRIOR: Dict[str, Set[str]] = {
    'modification_flag': {'P'},
    'payment_deferral_flag': {'P'},
}

SPECIAL_MENTION_POSITIVE_UPB = 'current_non_interest_bearing_upb'

DISASTER_FLAG = 'delinquency_due_to_disaster'
DISASTER_VALUES: Set[str] = {'Y'}

def disaster_mask(df: pd.DataFrame) -> pd.Series:
    """Rows whose delinquency is attributed to a declared disaster."""
    if DISASTER_FLAG not in df.columns:
        return pd.Series(False, index=df.index)
    return df[DISASTER_FLAG].isin(DISASTER_VALUES).fillna(False)

def special_mention_mask(df: pd.DataFrame, exclude_disaster: bool = True) -> pd.Series:
    """Rows with a workout in this month (EVENT sense): modification 'Y', payment deferral 'C', a borrower
    assistance plan F/R/T, or a positive non-interest-bearing UPB; disaster accommodation is excluded by
    default. Freddie reports modification and deferral as a flow and a stock in one column: 'Y'/'C' mark the
    month the intervention happened and 'P' every month after it, for the life of the loan, and the stock is
    14-90x the flow. The 'P' codes are in SPECIAL_MENTION_PRIOR, the PRIOR sense; borrower_assistance_plan has
    no prior code, as F/R/T describe a plan active now."""
    mask = pd.Series(False, index=df.index)
    for key, value in SPECIAL_MENTION_FLAGS.items():
        if key in df.columns:
            mask |= df[key].isin(value).fillna(False)
    if SPECIAL_MENTION_POSITIVE_UPB in df.columns:
        mask |= (pd.to_numeric(df[SPECIAL_MENTION_POSITIVE_UPB], errors="coerce") > 0).fillna(False)
    if exclude_disaster:
        mask &= ~disaster_mask(df)
    return mask

QUARTER_RE = re.compile(r"(?P<year>(?:19|20)\d{2})Q(?P<q>[1-4])", re.IGNORECASE)

def quarter_from_filename(path: Path | str) -> Optional[str]:
    """The 'YYYYQn' in a file name, or None."""
    m = QUARTER_RE.search(Path(path).name)
    return f"{m.group('year')}Q{m.group('q')}" if m else None

def is_performance_file(path: Path | str) -> bool:
    n = Path(path).name.lower()
    return n.startswith('perf_')

def resolve_columns(kind: str, n_actual: int) -> List[str]:
    """The column names of an origination or performance file, checked against its column count."""
    if kind not in {'origination', 'performance'}:
        raise ValueError(f'kind must be origination or performance, got {kind!r}')
    names = list(ORIGINATION_FIELDS if kind == 'origination' else PERFORMANCE_FIELDS)
    if len(names) != n_actual:
        raise ValueError(f'expected {n_actual} columns, got {len(names)}')
    return names