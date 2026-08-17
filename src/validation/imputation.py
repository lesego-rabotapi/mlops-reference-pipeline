"""
Missing-value imputation policies for the fraud dataset.

Imputation is chosen per column from evidence about *why* the value is
missing, not applied as a blanket default across every column with nulls.
Each policy below documents the missingness analysis that justified it, so
the choice stays auditable instead of being an unexplained fillna().

Full methodology and results for every column analyzed so far live in
docs/MISSINGNESS_ANALYSIS.md. The same checks are re-runnable via
scripts/analyze_missingness.py <column> for a new column or a new batch.

Key principle:
- Every imputed column keeps a companion indicator column, so the fact that
  a value was imputed is never silently lost.
- Every policy carries a missing-rate ceiling: max_missing_rate is the
  smallest multiple of 5 that is >= 2x validated_missing_rate. It is not a
  tuned hyperparameter -- it is a drift ceiling on the dataset the
  MCAR/MNAR analysis was actually run against. A new batch that blows past
  it should stop the pipeline rather than keep imputing on an assumption
  nobody re-checked.
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
    strategy: str  # "median" (continuous) or "mode" (categorical/binary)
    indicator_column: str
    max_missing_rate: float
    validated_missing_rate: float
    rationale: str


# --- Continuous columns (median strategy) -----------------------------

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
        "missingness on fraud.csv."
    ),
)

# Cross-tabulated customer_age missingness (12% overall) against
# is_first_transaction, device_type, store_type, is_weekend, and quartiles
# of transaction_amount / network_quality / prev_transactions on fraud.csv
# (n=7000): every split stayed within ~11%-13%, no sharp separation anywhere.
# Fraud rate in the missing group (9.4%) is slightly *lower* than in the
# non-missing group (10.4%) -- missingness is not concentrated in fraud.
# Pairwise correlation between customer_age's missingness indicator and every
# other column's missingness indicator was ~0.00-0.01.
CUSTOMER_AGE_SPEC = ImputationSpec(
    column="customer_age",
    strategy="median",
    indicator_column="customer_age_was_missing",
    max_missing_rate=0.25,
    validated_missing_rate=0.12,
    rationale=(
        "Missing rate stayed flat (~11%-13%) across every categorical driver "
        "and continuous-driver quartile tested -- no sharp split found. "
        "Fraud rate in the missing group (9.4%) is slightly lower than "
        "non-missing (10.4%), and missingness is ~uncorrelated with every "
        "other column's missingness (|r| <= 0.01). Treated as MCAR; "
        "validated at ~12% missingness on fraud.csv."
    ),
)

# Cross-tabulated distance_from_home missingness (10% overall) the same way:
# flat missing rate (~8%-11%) across every driver and quartile tested, fraud
# rate lower in the missing group (8.1% vs 10.5%), and missingness
# uncorrelated with every other column's missingness (|r| <= 0.02).
DISTANCE_FROM_HOME_SPEC = ImputationSpec(
    column="distance_from_home",
    strategy="median",
    indicator_column="distance_from_home_was_missing",
    max_missing_rate=0.20,
    validated_missing_rate=0.10,
    rationale=(
        "Missing rate stayed flat (~8%-11%) across every categorical driver "
        "and continuous-driver quartile tested -- no sharp split found. "
        "Fraud rate in the missing group (8.1%) is lower than non-missing "
        "(10.5%), and missingness is ~uncorrelated with every other "
        "column's missingness (|r| <= 0.02). Treated as MCAR; validated at "
        "~10% missingness on fraud.csv."
    ),
)

# network_quality (9% missing): fraud-rate gap 9.7% vs 10.4% is not
# statistically significant (two-proportion z-test, p=0.593). No driver
# crosstab or continuous-quartile split exceeded a 2.1-point spread. Max
# missingness correlation with any other column: 0.022. See
# docs/MISSINGNESS_ANALYSIS.md for the full table.
NETWORK_QUALITY_SPEC = ImputationSpec(
    column="network_quality",
    strategy="median",
    indicator_column="network_quality_was_missing",
    max_missing_rate=0.20,
    validated_missing_rate=0.09,
    rationale=(
        "Fraud-rate gap (9.7% vs 10.4%) not significant (p=0.593). No driver "
        "or quartile split showed a spread over 2.1 points; missingness "
        "~uncorrelated with other columns (max |r|=0.022). Treated as MCAR; "
        "validated at ~9% missingness on fraud.csv."
    ),
)

# transaction_amount (8% missing): fraud-rate gap 10.4% vs 10.3%, p=0.963.
# Widest driver/quartile spread was 2.8 points (velocity_score quartile).
# Max missingness correlation with any other column: 0.037.
TRANSACTION_AMOUNT_SPEC = ImputationSpec(
    column="transaction_amount",
    strategy="median",
    indicator_column="transaction_amount_was_missing",
    max_missing_rate=0.20,
    validated_missing_rate=0.08,
    rationale=(
        "Fraud-rate gap (10.4% vs 10.3%) not significant (p=0.963). Widest "
        "driver/quartile spread was 2.8 points; missingness ~uncorrelated "
        "with other columns (max |r|=0.037). Treated as MCAR; validated at "
        "~8% missingness on fraud.csv."
    ),
)

# prev_transactions (7% missing): fraud-rate gap 11.0% vs 10.2%, p=0.586.
# Widest driver/quartile spread was 1.2 points. Max missingness correlation
# with any other column: 0.024.
PREV_TRANSACTIONS_SPEC = ImputationSpec(
    column="prev_transactions",
    strategy="median",
    indicator_column="prev_transactions_was_missing",
    max_missing_rate=0.15,
    validated_missing_rate=0.07,
    rationale=(
        "Fraud-rate gap (11.0% vs 10.2%) not significant (p=0.586). Widest "
        "driver/quartile spread was 1.2 points; missingness ~uncorrelated "
        "with other columns (max |r|=0.024). Treated as MCAR; validated at "
        "~7% missingness on fraud.csv."
    ),
)

# num_items (3% missing): fraud-rate gap 6.7% vs 10.4% is the closest to
# significant of any column tested (p=0.079, still above the 0.05
# threshold) -- worth re-checking first if this column's missingness rate
# grows. Widest driver/quartile spread was 1.3 points. Max missingness
# correlation with any other column: 0.020.
NUM_ITEMS_SPEC = ImputationSpec(
    column="num_items",
    strategy="median",
    indicator_column="num_items_was_missing",
    max_missing_rate=0.10,
    validated_missing_rate=0.03,
    rationale=(
        "Fraud-rate gap (6.7% vs 10.4%) is the closest to significant of "
        "any column tested (p=0.079) but still above the 0.05 threshold on "
        "n=210 missing rows. Widest driver/quartile spread was 1.3 points; "
        "missingness ~uncorrelated with other columns (max |r|=0.020). "
        "Treated as MCAR on current evidence; validated at ~3% missingness "
        "on fraud.csv -- re-run this column's analysis before raising its "
        "ceiling, since it's the weakest MCAR case of the twelve analyzed."
    ),
)

# --- Categorical / binary columns (mode strategy) ----------------------

# hour_of_day (5% missing, 3-level time-of-day bucket): fraud-rate gap
# 10.3% vs 10.3%, p=0.993. Widest driver/quartile spread was 1.8 points.
# Max missingness correlation with any other column: 0.013.
HOUR_OF_DAY_SPEC = ImputationSpec(
    column="hour_of_day",
    strategy="mode",
    indicator_column="hour_of_day_was_missing",
    max_missing_rate=0.10,
    validated_missing_rate=0.05,
    rationale=(
        "Fraud-rate gap (10.3% vs 10.3%) not significant (p=0.993). Widest "
        "driver/quartile spread was 1.8 points; missingness ~uncorrelated "
        "with other columns (max |r|=0.013). Treated as MCAR; validated at "
        "~5% missingness on fraud.csv. Mode (not median) since hour_of_day "
        "is a 3-level bucket, not continuous."
    ),
)

# device_type (4% missing, categorical code): fraud-rate gap 7.9% vs 10.4%,
# p=0.170. Widest driver/quartile spread was 1.4 points. Max missingness
# correlation with any other column: 0.027.
DEVICE_TYPE_SPEC = ImputationSpec(
    column="device_type",
    strategy="mode",
    indicator_column="device_type_was_missing",
    max_missing_rate=0.10,
    validated_missing_rate=0.04,
    rationale=(
        "Fraud-rate gap (7.9% vs 10.4%) not significant (p=0.170). Widest "
        "driver/quartile spread was 1.4 points; missingness ~uncorrelated "
        "with other columns (max |r|=0.027). Treated as MCAR; validated at "
        "~4% missingness on fraud.csv."
    ),
)

# is_first_transaction (3% missing, binary): fraud-rate gap 11.0% vs 10.3%,
# p=0.752. Widest driver/quartile spread was 1.4 points. Max missingness
# correlation with any other column: 0.029.
IS_FIRST_TRANSACTION_SPEC = ImputationSpec(
    column="is_first_transaction",
    strategy="mode",
    indicator_column="is_first_transaction_was_missing",
    max_missing_rate=0.10,
    validated_missing_rate=0.03,
    rationale=(
        "Fraud-rate gap (11.0% vs 10.3%) not significant (p=0.752). Widest "
        "driver/quartile spread was 1.4 points; missingness ~uncorrelated "
        "with other columns (max |r|=0.029). Treated as MCAR; validated at "
        "~3% missingness on fraud.csv."
    ),
)

# is_weekend (2% missing, binary): fraud-rate gap 7.1% vs 10.4%, p=0.214.
# Widest driver/quartile spread was 1.4 points. Max missingness correlation
# with any other column: 0.037.
IS_WEEKEND_SPEC = ImputationSpec(
    column="is_weekend",
    strategy="mode",
    indicator_column="is_weekend_was_missing",
    max_missing_rate=0.05,
    validated_missing_rate=0.02,
    rationale=(
        "Fraud-rate gap (7.1% vs 10.4%) not significant (p=0.214). Widest "
        "driver/quartile spread was 1.4 points; missingness ~uncorrelated "
        "with other columns (max |r|=0.037). Treated as MCAR; validated at "
        "~2% missingness on fraud.csv. A 2% baseline leaves little room "
        "before this rule should be re-checked -- ceiling is only 3 points "
        "above baseline in absolute terms."
    ),
)

# store_type (2% missing, categorical code): fraud-rate gap 6.4% vs 10.4%,
# p=0.128. Widest driver/quartile spread was 1.0 point. Max missingness
# correlation with any other column: 0.013.
STORE_TYPE_SPEC = ImputationSpec(
    column="store_type",
    strategy="mode",
    indicator_column="store_type_was_missing",
    max_missing_rate=0.05,
    validated_missing_rate=0.02,
    rationale=(
        "Fraud-rate gap (6.4% vs 10.4%) not significant (p=0.128). Widest "
        "driver/quartile spread was 1.0 point; missingness ~uncorrelated "
        "with other columns (max |r|=0.013). Treated as MCAR; validated at "
        "~2% missingness on fraud.csv. Same tight-ceiling caveat as "
        "is_weekend applies here."
    ),
)


def _impute(df: pd.DataFrame, spec: ImputationSpec) -> pd.DataFrame:
    """
    Fill spec.column's nulls per its strategy, preserving an indicator column.

    "median" is used for continuous columns (outlier-robust). "mode" is used
    for categorical/binary columns, where a median is meaningless -- it's
    the same MCAR-driven default SimpleImputer(strategy="most_frequent")
    applies downstream in build_features.py's categorical pipeline, made
    explicit and auditable here instead of silently happening later.
    """
    result = df.copy()
    result[spec.indicator_column] = result[spec.column].isna().astype(int)

    if spec.strategy == "median":
        fill_value = result[spec.column].median()
    elif spec.strategy == "mode":
        fill_value = result[spec.column].mode().iloc[0]
    else:
        raise NotImplementedError(
            f"Unsupported imputation strategy '{spec.strategy}' for '{spec.column}'."
        )

    result[spec.column] = result[spec.column].fillna(fill_value)

    logger.info(
        "Imputed '%s': %d rows filled with %s=%s",
        spec.column,
        int(result[spec.indicator_column].sum()),
        spec.strategy,
        fill_value,
    )
    return result


def impute_velocity_score(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing velocity_score with the column median. See VELOCITY_SCORE_SPEC."""
    return _impute(df, VELOCITY_SCORE_SPEC)


def impute_customer_age(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing customer_age with the column median. See CUSTOMER_AGE_SPEC."""
    return _impute(df, CUSTOMER_AGE_SPEC)


def impute_distance_from_home(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing distance_from_home with the column median. See DISTANCE_FROM_HOME_SPEC."""
    return _impute(df, DISTANCE_FROM_HOME_SPEC)


def impute_network_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing network_quality with the column median. See NETWORK_QUALITY_SPEC."""
    return _impute(df, NETWORK_QUALITY_SPEC)


def impute_transaction_amount(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing transaction_amount with the column median. See TRANSACTION_AMOUNT_SPEC."""
    return _impute(df, TRANSACTION_AMOUNT_SPEC)


def impute_prev_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing prev_transactions with the column median. See PREV_TRANSACTIONS_SPEC."""
    return _impute(df, PREV_TRANSACTIONS_SPEC)


def impute_num_items(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing num_items with the column median. See NUM_ITEMS_SPEC."""
    return _impute(df, NUM_ITEMS_SPEC)


def impute_hour_of_day(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing hour_of_day with the column mode. See HOUR_OF_DAY_SPEC."""
    return _impute(df, HOUR_OF_DAY_SPEC)


def impute_device_type(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing device_type with the column mode. See DEVICE_TYPE_SPEC."""
    return _impute(df, DEVICE_TYPE_SPEC)


def impute_is_first_transaction(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing is_first_transaction with the column mode. See IS_FIRST_TRANSACTION_SPEC."""
    return _impute(df, IS_FIRST_TRANSACTION_SPEC)


def impute_is_weekend(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing is_weekend with the column mode. See IS_WEEKEND_SPEC."""
    return _impute(df, IS_WEEKEND_SPEC)


def impute_store_type(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing store_type with the column mode. See STORE_TYPE_SPEC."""
    return _impute(df, STORE_TYPE_SPEC)


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
    "network_quality": NETWORK_QUALITY_SPEC,
    "transaction_amount": TRANSACTION_AMOUNT_SPEC,
    "prev_transactions": PREV_TRANSACTIONS_SPEC,
    "num_items": NUM_ITEMS_SPEC,
    "hour_of_day": HOUR_OF_DAY_SPEC,
    "device_type": DEVICE_TYPE_SPEC,
    "is_first_transaction": IS_FIRST_TRANSACTION_SPEC,
    "is_weekend": IS_WEEKEND_SPEC,
    "store_type": STORE_TYPE_SPEC,
}

IMPUTATION_RULES: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "velocity_score": impute_velocity_score,
    "customer_age": impute_customer_age,
    "distance_from_home": impute_distance_from_home,
    "network_quality": impute_network_quality,
    "transaction_amount": impute_transaction_amount,
    "prev_transactions": impute_prev_transactions,
    "num_items": impute_num_items,
    "hour_of_day": impute_hour_of_day,
    "device_type": impute_device_type,
    "is_first_transaction": impute_is_first_transaction,
    "is_weekend": impute_is_weekend,
    "store_type": impute_store_type,
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
