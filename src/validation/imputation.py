"""
Missing-value imputation policies for the fraud dataset.

Imputation is chosen per column from evidence about *why* the value is
missing, not applied as a blanket default across every numeric column with
nulls. Each policy below documents the missingness analysis that justified
it, so the choice stays auditable instead of being an unexplained fillna().

Key principle:
- Every imputed column keeps a companion indicator column, so the fact that
  a value was imputed is never silently lost.
- Every policy carries a missing-rate ceiling. The ceiling is not a tuned
  hyperparameter -- it is the ceiling of the dataset the MCAR/MNAR analysis
  was actually run against. A new batch that blows past it should stop the
  pipeline rather than keep imputing on an assumption nobody re-checked.
"""

import logging
from dataclasses import dataclass
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImputationSpec:
    """One column's documented imputation policy."""

    column: str
    strategy: str
    indicator_column: str
    max_missing_rate: float
    validated_missing_rate: float
    rationale: str


# Cross-tabulated velocity_score missingness against is_first_transaction and
# is_fraud on the validated fraud.csv (n=7000, 15% missing in this column):
#   - missing rate when is_first_transaction=0: 14.7%, vs =1: 18.3%
#     (too small a gap for velocity to be structurally uncomputable for
#     first-time transactions)
#   - fraud rate in the missing group: 8.8%, vs 10.6% in the non-missing
#     group (missingness is not concentrated in fraud cases)
# Neither split shows the sharp separation expected of MNAR, so missingness
# is treated as MCAR here and filled with the column median.
VELOCITY_SCORE_SPEC = ImputationSpec(
    column="velocity_score",
    strategy="median",
    indicator_column="velocity_score_was_missing",
    max_missing_rate=0.30,
    validated_missing_rate=0.15,
    rationale=(
        "Missingness tested against is_first_transaction (14.7% vs 18.3%) "
        "and is_fraud (8.8% vs 10.6%) shows no sharp split, so it behaves "
        "close to MCAR. Median imputation is defensible; validated at ~15% "
        "missingness on fraud.csv. max_missing_rate (30%, roughly double "
        "the validated rate) is a drift ceiling on that assumption, not a "
        "tuned value -- re-run the MCAR analysis before raising it."
    ),
)

# Cross-tabulated customer_age missingness (12% overall) against
# is_first_transaction, device_type, store_type, is_weekend, and quartiles
# of transaction_amount / network_quality / prev_transactions on fraud.csv
# (n=7000): every split stayed within ~11%-13%, no sharp separation anywhere.
# Fraud rate in the missing group (9.4%) is slightly *lower* than in the
# non-missing group (10.4%) -- missingness is not concentrated in fraud.
# Pairwise correlation between customer_age's missingness indicator and every
# other column's missingness indicator was ~0.00-0.01 -- rows missing
# customer_age aren't disproportionately missing other fields either.
# Same MCAR profile as velocity_score; median imputation applied on the same
# basis.
CUSTOMER_AGE_SPEC = ImputationSpec(
    column="customer_age",
    strategy="median",
    indicator_column="customer_age_was_missing",
    max_missing_rate=0.25,
    validated_missing_rate=0.12,
    rationale=(
        "Missing rate stayed flat (~11%-13%) across is_first_transaction, "
        "device_type, store_type, is_weekend, and quartiles of "
        "transaction_amount/network_quality/prev_transactions -- no sharp "
        "split found. Fraud rate in the missing group (9.4%) is slightly "
        "lower than non-missing (10.4%), and missingness is ~uncorrelated "
        "with every other column's missingness (|r| <= 0.01). Treated as "
        "MCAR; validated at ~12% missingness on fraud.csv. max_missing_rate "
        "(25%, roughly double the validated rate) is a drift ceiling, not a "
        "tuned value -- re-run the MCAR analysis before raising it."
    ),
)

# Cross-tabulated distance_from_home missingness (10% overall) the same way:
# flat missing rate (~8%-11%) across every driver and quartile tested, fraud
# rate lower in the missing group (8.1% vs 10.5%), and missingness
# uncorrelated with every other column's missingness (|r| <= 0.02). Same
# MCAR profile; median imputation applied on the same basis.
DISTANCE_FROM_HOME_SPEC = ImputationSpec(
    column="distance_from_home",
    strategy="median",
    indicator_column="distance_from_home_was_missing",
    max_missing_rate=0.20,
    validated_missing_rate=0.10,
    rationale=(
        "Missing rate stayed flat (~8%-11%) across is_first_transaction, "
        "device_type, store_type, is_weekend, and quartiles of "
        "transaction_amount/network_quality/prev_transactions -- no sharp "
        "split found. Fraud rate in the missing group (8.1%) is lower than "
        "non-missing (10.5%), and missingness is ~uncorrelated with every "
        "other column's missingness (|r| <= 0.02). Treated as MCAR; "
        "validated at ~10% missingness on fraud.csv. max_missing_rate (20%, "
        "roughly double the validated rate) is a drift ceiling, not a tuned "
        "value -- re-run the MCAR analysis before raising it."
    ),
)


def _impute_median(df: pd.DataFrame, spec: ImputationSpec) -> pd.DataFrame:
    """
    Fill spec.column's nulls with its median, preserving an indicator column.

    Median (not mean) is the default here because it's outlier-robust and,
    for every column analyzed so far, mean and median were close enough that
    the choice is a robustness call rather than a correction for observed
    skew. Only median-strategy specs exist today; a different strategy value
    would need its own branch here plus its own justification in the spec's
    rationale.
    """
    if spec.strategy != "median":
        raise NotImplementedError(
            f"Unsupported imputation strategy '{spec.strategy}' for "
            f"'{spec.column}'."
        )

    result = df.copy()
    result[spec.indicator_column] = result[spec.column].isna().astype(int)
    median_value = result[spec.column].median()
    result[spec.column] = result[spec.column].fillna(median_value)

    logger.info(
        "Imputed '%s': %d rows filled with median=%.4f",
        spec.column,
        int(result[spec.indicator_column].sum()),
        median_value,
    )
    return result


def impute_velocity_score(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing velocity_score with the column median. See VELOCITY_SCORE_SPEC."""
    return _impute_median(df, VELOCITY_SCORE_SPEC)


def impute_customer_age(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing customer_age with the column median. See CUSTOMER_AGE_SPEC."""
    return _impute_median(df, CUSTOMER_AGE_SPEC)


def impute_distance_from_home(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing distance_from_home with the column median. See DISTANCE_FROM_HOME_SPEC."""
    return _impute_median(df, DISTANCE_FROM_HOME_SPEC)


def check_missing_rate_threshold(
    df: pd.DataFrame, spec: ImputationSpec
) -> tuple[bool, str]:
    """
    Guard the MCAR assumption behind an imputation policy before imputing.

    If a new batch's missing rate has drifted well past what the MCAR
    analysis was validated against, the strategy should not be trusted
    blindly. Surface it so a human re-checks the analysis instead of
    silently imputing an ever-larger share of the column.
    """
    if spec.column not in df.columns:
        return False, f"Column '{spec.column}' not found for missingness check."

    missing_rate = df[spec.column].isna().mean()
    if missing_rate > spec.max_missing_rate:
        return False, (
            f"'{spec.column}' missing rate {missing_rate:.1%} exceeds the "
            f"validated ceiling of {spec.max_missing_rate:.0%}. The MCAR "
            f"assumption behind {spec.strategy} imputation was validated at "
            f"~{spec.validated_missing_rate:.0%} missingness on fraud.csv "
            "and should not be assumed to hold at this rate -- halting "
            "instead of imputing silently."
        )

    return True, ""


# Explicit per-column registry. Each column's imputation strategy is chosen
# deliberately from evidence -- new columns must not be swept into a
# generic catch-all without their own missingness analysis and spec.
IMPUTATION_SPECS: dict[str, ImputationSpec] = {
    "velocity_score": VELOCITY_SCORE_SPEC,
    "customer_age": CUSTOMER_AGE_SPEC,
    "distance_from_home": DISTANCE_FROM_HOME_SPEC,
}

IMPUTATION_RULES: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "velocity_score": impute_velocity_score,
    "customer_age": impute_customer_age,
    "distance_from_home": impute_distance_from_home,
}


def run_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply every registered imputation policy after its missing-rate ceiling
    has been checked. Raises if any column has drifted past its ceiling.
    """
    result = df
    for column, spec in IMPUTATION_SPECS.items():
        passed, error = check_missing_rate_threshold(result, spec)
        if not passed:
            raise ValueError(error)
        result = IMPUTATION_RULES[column](result)
    return result
