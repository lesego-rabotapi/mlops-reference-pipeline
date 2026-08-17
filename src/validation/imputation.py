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
    rationale=(
        "Missingness tested against is_first_transaction (14.7% vs 18.3%) "
        "and is_fraud (8.8% vs 10.6%) shows no sharp split, so it behaves "
        "close to MCAR. Median imputation is defensible; validated at ~15% "
        "missingness on fraud.csv. max_missing_rate (30%, roughly double "
        "the validated rate) is a drift ceiling on that assumption, not a "
        "tuned value -- re-run the MCAR analysis before raising it."
    ),
)


def impute_velocity_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing velocity_score with the column median.

    Median (not mean) is used because velocity_score's minimum is negative
    while most of the distribution is positive; mean and median were nearly
    identical at validation time (5.01 vs 4.99), so this is a robustness
    choice rather than a correction for skew observed so far.
    """
    spec = VELOCITY_SCORE_SPEC
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
            "~15% missingness on the original dataset and should not be "
            "assumed to hold at this rate -- halting instead of imputing "
            "silently."
        )

    return True, ""


# Explicit per-column registry. Each column's imputation strategy is chosen
# deliberately from evidence -- new columns must not be swept into a
# generic catch-all without their own missingness analysis and spec.
IMPUTATION_SPECS: dict[str, ImputationSpec] = {
    "velocity_score": VELOCITY_SCORE_SPEC,
}

IMPUTATION_RULES: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "velocity_score": impute_velocity_score,
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
