"""
Validation Rules

These are reusable validation specifications.
They define WHAT to validate, not HOW to validate.

Key principle:
- Each rule returns (bool, str): (passed, error_message)
- Rules are stateless functions
- Rules can be composed and reused across pipelines

Used by:
- validate_data.py (initial data validation)
- train_model.py (training data validation)
- monitoring (data drift detection)
"""

import pandas as pd
from pandas.api.types import (
    is_float_dtype,
    is_integer_dtype,
    is_object_dtype,
    is_string_dtype,
)
from typing import Callable, Tuple

from src.validation.imputation import IMPUTATION_SPECS, ImputationSpec, check_missing_rate_threshold


class SchemaRules:
    """Schema and datatype validation."""

    # Define the expected schema as a class constant
    # This becomes the single source of truth for schema
    #
    # All columns except is_fraud are float in the raw CSV: every one of
    # them has missing values (2%-15%), which forces pandas to upcast even
    # binary/count columns to float64 on read. is_fraud is the only column
    # with 0% missingness, so it stays integer.
    REQUIRED_SCHEMA = {
        "transaction_amount": "float",
        "hour_of_day": "float",
        "is_weekend": "float",
        "num_items": "float",
        "customer_age": "float",
        "prev_transactions": "float",
        "distance_from_home": "float",
        "device_type": "float",
        "network_quality": "float",
        "is_first_transaction": "float",
        "store_type": "float",
        "velocity_score": "float",
        "is_fraud": "integer",
    }

    @staticmethod
    def validate_required_columns(df: pd.DataFrame) -> Tuple[bool, list]:
        """
        Check that all required columns exist.

        Returns:
            (passed: bool, errors: list of missing columns)
        """
        missing = set(SchemaRules.REQUIRED_SCHEMA.keys()) - set(df.columns)

        if missing:
            return False, [f"Missing required column: {col}" for col in missing]

        return True, []

    @staticmethod
    def validate_datatypes(df: pd.DataFrame) -> Tuple[bool, list]:
        """
        Check that columns have correct datatypes.

        Returns:
            (passed: bool, errors: list of datatype mismatches)
        """
        errors = []

        for column, expected_dtype in SchemaRules.REQUIRED_SCHEMA.items():
            if column not in df.columns:
                continue  # Caught by validate_required_columns

            if not SchemaRules._dtype_matches(df[column], expected_dtype):
                errors.append(
                    f"Invalid dtype for '{column}': "
                    f"expected {expected_dtype}, got {df[column].dtype}"
                )

        return len(errors) == 0, errors

    @staticmethod
    def _dtype_matches(series: pd.Series, expected_dtype: str) -> bool:
        """
        Check logical dtypes instead of exact pandas dtype strings.

        Exact strings such as object, str, int64, or Int64 can differ across
        pandas versions and CSV parsing settings. The data contract cares about
        meaning: integer-like, float-like, or string-like.
        """
        if expected_dtype == "integer":
            return is_integer_dtype(series)

        if expected_dtype == "float":
            return is_float_dtype(series)

        if expected_dtype == "string":
            return is_string_dtype(series) or is_object_dtype(series)

        raise ValueError(f"Unsupported expected dtype: {expected_dtype}")

    @staticmethod
    def validate_no_unexpected_columns(df: pd.DataFrame) -> Tuple[bool, list]:
        """
        Check for columns that shouldn't exist.
        This helps catch data drift or schema creep.
        """
        unexpected = set(df.columns) - set(SchemaRules.REQUIRED_SCHEMA.keys())

        if unexpected:
            return False, [f"Unexpected column: {col}" for col in unexpected]

        return True, []


class FeatureRules:
    """
    Business logic and feature validation rules.

    Every column here has real missingness (2%-15% in the source dataset), so
    each check drops nulls before evaluating a range or set. Unlike a
    zero-null schema, a raw NaN here is expected, and it's validated
    separately (see DataQualityRules.target_not_null and the missing-rate
    ceiling rule).
    """

    @staticmethod
    def transaction_amount_non_negative(df: pd.DataFrame) -> Tuple[bool, str]:
        """transaction_amount cannot be negative."""
        values = df["transaction_amount"].dropna()
        invalid = values < 0
        if invalid.any():
            return False, f"Found {invalid.sum()} rows with negative transaction_amount."
        return True, ""

    @staticmethod
    def num_items_non_negative(df: pd.DataFrame) -> Tuple[bool, str]:
        """num_items cannot be negative."""
        values = df["num_items"].dropna()
        invalid = values < 0
        if invalid.any():
            return False, f"Found {invalid.sum()} rows with negative num_items."
        return True, ""

    @staticmethod
    def prev_transactions_non_negative(df: pd.DataFrame) -> Tuple[bool, str]:
        """prev_transactions cannot be negative."""
        values = df["prev_transactions"].dropna()
        invalid = values < 0
        if invalid.any():
            return False, f"Found {invalid.sum()} rows with negative prev_transactions."
        return True, ""

    @staticmethod
    def distance_from_home_non_negative(df: pd.DataFrame) -> Tuple[bool, str]:
        """distance_from_home cannot be negative."""
        values = df["distance_from_home"].dropna()
        invalid = values < 0
        if invalid.any():
            return False, f"Found {invalid.sum()} rows with negative distance_from_home."
        return True, ""

    @staticmethod
    def customer_age_in_valid_range(df: pd.DataFrame) -> Tuple[bool, str]:
        """customer_age must be between 0 and 120 (realistic human age range)."""
        values = df["customer_age"].dropna()
        invalid = (values < 0) | (values > 120)
        if invalid.any():
            return False, f"Found {invalid.sum()} rows with invalid customer_age values."
        return True, ""

    @staticmethod
    def network_quality_in_valid_range(df: pd.DataFrame) -> Tuple[bool, str]:
        """network_quality is a 0-100 quality score."""
        values = df["network_quality"].dropna()
        invalid = (values < 0) | (values > 100)
        if invalid.any():
            return False, f"Found {invalid.sum()} rows with invalid network_quality values."
        return True, ""

    @staticmethod
    def hour_of_day_valid(df: pd.DataFrame) -> Tuple[bool, str]:
        """hour_of_day is a 3-level time-of-day bucket: {1, 2, 3}."""
        valid_values = {1.0, 2.0, 3.0}
        invalid = set(df["hour_of_day"].dropna().unique()) - valid_values
        if invalid:
            return False, f"Invalid hour_of_day values found: {invalid}"
        return True, ""

    @staticmethod
    def device_type_valid(df: pd.DataFrame) -> Tuple[bool, str]:
        """device_type is a categorical code: {0, 1, 2}."""
        valid_values = {0.0, 1.0, 2.0}
        invalid = set(df["device_type"].dropna().unique()) - valid_values
        if invalid:
            return False, f"Invalid device_type values found: {invalid}"
        return True, ""

    @staticmethod
    def store_type_valid(df: pd.DataFrame) -> Tuple[bool, str]:
        """store_type is a categorical code: {0, 1}."""
        valid_values = {0.0, 1.0}
        invalid = set(df["store_type"].dropna().unique()) - valid_values
        if invalid:
            return False, f"Invalid store_type values found: {invalid}"
        return True, ""

    @staticmethod
    def is_weekend_valid(df: pd.DataFrame) -> Tuple[bool, str]:
        """is_weekend must be binary: 0 or 1."""
        valid_values = {0.0, 1.0}
        invalid = set(df["is_weekend"].dropna().unique()) - valid_values
        if invalid:
            return False, f"Invalid is_weekend values found: {invalid}"
        return True, ""

    @staticmethod
    def is_first_transaction_valid(df: pd.DataFrame) -> Tuple[bool, str]:
        """is_first_transaction must be binary: 0 or 1."""
        valid_values = {0.0, 1.0}
        invalid = set(df["is_first_transaction"].dropna().unique()) - valid_values
        if invalid:
            return False, f"Invalid is_first_transaction values found: {invalid}"
        return True, ""

    @staticmethod
    def is_fraud_binary(df: pd.DataFrame) -> Tuple[bool, str]:
        """is_fraud (target) must be binary: 0 or 1."""
        valid_values = {0, 1}
        invalid = set(df["is_fraud"].dropna().unique()) - valid_values
        if invalid:
            return False, f"Invalid is_fraud values found: {invalid}. Must be 0 or 1."
        return True, ""

    @staticmethod
    def no_duplicate_rows(df: pd.DataFrame) -> Tuple[bool, str]:
        """
        No fully duplicate rows.

        Unlike the churn dataset, this data has no id/customer-id column to
        key off of, so duplication is checked across the full row instead.
        """
        duplicated = df.duplicated()
        if duplicated.any():
            return False, f"Found {duplicated.sum()} fully duplicate rows."
        return True, ""


class DataQualityRules:
    """Data quality and completeness rules."""

    @staticmethod
    def target_not_null(df: pd.DataFrame) -> Tuple[bool, str]:
        """
        is_fraud (the label) must never be null.

        This replaces a blanket "no nulls anywhere" check, which would fail
        every batch of this dataset by design -- every feature column has
        2%-15% missingness. is_fraud is the only column verified at 0%
        missingness, so it's the one column this stage still hard-requires.
        Missingness in every other column is handled per-column via
        IMPUTATION_GUARD_RULES (see src/validation/imputation.py).
        """
        nulls = df["is_fraud"].isnull().sum()
        if nulls > 0:
            return False, f"Found {nulls} null values in target column 'is_fraud'."
        return True, ""

    @staticmethod
    def minimum_dataset_size(df: pd.DataFrame, min_rows: int = 100) -> Tuple[bool, str]:
        """Dataset must have minimum number of rows."""
        if len(df) < min_rows:
            return False, f"Dataset has {len(df)} rows, but minimum required is {min_rows}."

        return True, ""


def _make_missing_rate_guard(
    spec: ImputationSpec,
) -> Callable[[pd.DataFrame], Tuple[bool, str]]:
    """
    Build a rule-registry-shaped guard ((df) -> (bool, str)) for one
    ImputationSpec. The guard logic is identical for every column, only
    the spec differs (see src/validation/imputation.py and
    docs/MISSINGNESS_ANALYSIS.md for where each one comes from). Generated
    from IMPUTATION_SPECS instead of hand-writing one near-identical
    static method per column, so rules.py can't drift out of sync as new
    columns get an approved policy.
    """

    def guard(df: pd.DataFrame) -> Tuple[bool, str]:
        return check_missing_rate_threshold(df, spec)

    guard.__name__ = f"{spec.column}_missing_rate_within_ceiling"
    guard.__doc__ = (
        f"{spec.column}'s MCAR-based {spec.strategy} imputation was "
        f"validated at ~{spec.validated_missing_rate:.0%} missingness. If a "
        "new batch drifts well past that, halt instead of imputing on an "
        "assumption nobody re-checked."
    )
    return guard


# One ceiling-guard rule per approved imputation policy, keyed the same way
# every other rule is: "<column>_missing_rate_within_ceiling".
IMPUTATION_GUARD_RULES: dict[str, Callable[[pd.DataFrame], Tuple[bool, str]]] = {
    f"{column}_missing_rate_within_ceiling": _make_missing_rate_guard(spec)
    for column, spec in IMPUTATION_SPECS.items()
}


# Rules registry: defines all validation rules and their order
# Used by validator to run all checks systematically
VALIDATION_RULES = {
    # Schema checks (fail fast - if schema is wrong, nothing else matters)
    "required_columns": SchemaRules.validate_required_columns,
    "datatypes": SchemaRules.validate_datatypes,
    "no_unexpected_columns": SchemaRules.validate_no_unexpected_columns,

    # Data quality checks (completeness)
    "target_not_null": DataQualityRules.target_not_null,
    "minimum_size": DataQualityRules.minimum_dataset_size,

    # Imputation policy guards -- one per column in IMPUTATION_SPECS
    **IMPUTATION_GUARD_RULES,

    # Feature business logic checks
    "transaction_amount_non_negative": FeatureRules.transaction_amount_non_negative,
    "num_items_non_negative": FeatureRules.num_items_non_negative,
    "prev_transactions_non_negative": FeatureRules.prev_transactions_non_negative,
    "distance_from_home_non_negative": FeatureRules.distance_from_home_non_negative,
    "customer_age_valid": FeatureRules.customer_age_in_valid_range,
    "network_quality_valid": FeatureRules.network_quality_in_valid_range,
    "hour_of_day_valid": FeatureRules.hour_of_day_valid,
    "device_type_valid": FeatureRules.device_type_valid,
    "store_type_valid": FeatureRules.store_type_valid,
    "is_weekend_valid": FeatureRules.is_weekend_valid,
    "is_first_transaction_valid": FeatureRules.is_first_transaction_valid,
    "is_fraud_binary": FeatureRules.is_fraud_binary,
    "no_duplicate_rows": FeatureRules.no_duplicate_rows,
}
