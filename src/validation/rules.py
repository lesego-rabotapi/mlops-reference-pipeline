"""
Validation Rules (PaySim dataset)

These are reusable validation specifications.
They define WHAT to validate, not HOW to validate.

Key principle:
- Each rule returns (bool, str | list): (passed, error message(s))
- Rules are stateless functions
- Rules can be composed and reused across pipelines

Used by:
- validate_data.py (initial data validation)
"""

from typing import Tuple

import pandas as pd
from pandas.api.types import (
    is_float_dtype,
    is_integer_dtype,
    is_object_dtype,
    is_string_dtype,
)


class SchemaRules:
    """Schema and datatype validation."""

    # Single source of truth for the raw schema. PaySim has no missing
    # values, so integer columns stay integer on read (unlike the rejected
    # Dataset v1, where missingness forced float upcasting).
    REQUIRED_SCHEMA = {
        "step": "integer",
        "type": "string",
        "amount": "float",
        "nameOrig": "string",
        "oldbalanceOrg": "float",
        "newbalanceOrig": "float",
        "nameDest": "string",
        "oldbalanceDest": "float",
        "newbalanceDest": "float",
        "isFraud": "integer",
        "isFlaggedFraud": "integer",
    }

    VALID_TYPES = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}

    @staticmethod
    def validate_required_columns(df: pd.DataFrame) -> Tuple[bool, list]:
        """Check that all required columns exist."""
        missing = set(SchemaRules.REQUIRED_SCHEMA.keys()) - set(df.columns)
        if missing:
            return False, [f"Missing required column: {col}" for col in sorted(missing)]
        return True, []

    @staticmethod
    def validate_datatypes(df: pd.DataFrame) -> Tuple[bool, list]:
        """Check that columns have correct logical datatypes."""
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
        Check logical dtypes instead of exact pandas dtype strings, since
        exact strings (object, str, int64, Int64) differ across pandas
        versions and CSV parsing settings.
        """
        if expected_dtype == "integer":
            return is_integer_dtype(series)
        if expected_dtype == "float":
            # Amounts/balances can all be whole numbers; accept ints too.
            return is_float_dtype(series) or is_integer_dtype(series)
        if expected_dtype == "string":
            return is_string_dtype(series) or is_object_dtype(series)
        raise ValueError(f"Unsupported expected dtype: {expected_dtype}")

    @staticmethod
    def validate_no_unexpected_columns(df: pd.DataFrame) -> Tuple[bool, list]:
        """Check for columns that shouldn't exist (catches schema drift)."""
        unexpected = set(df.columns) - set(SchemaRules.REQUIRED_SCHEMA.keys())
        if unexpected:
            return False, [f"Unexpected column: {col}" for col in sorted(unexpected)]
        return True, []


def _non_negative(column: str):
    def rule(df: pd.DataFrame) -> Tuple[bool, str]:
        invalid = df[column] < 0
        if invalid.any():
            return False, f"Found {int(invalid.sum())} rows with negative {column}."
        return True, ""

    rule.__name__ = f"{column}_non_negative"
    rule.__doc__ = f"{column} cannot be negative."
    return rule


class FeatureRules:
    """Business-logic and value-domain rules."""

    amount_non_negative = staticmethod(_non_negative("amount"))
    oldbalanceOrg_non_negative = staticmethod(_non_negative("oldbalanceOrg"))
    newbalanceOrig_non_negative = staticmethod(_non_negative("newbalanceOrig"))
    oldbalanceDest_non_negative = staticmethod(_non_negative("oldbalanceDest"))
    newbalanceDest_non_negative = staticmethod(_non_negative("newbalanceDest"))

    @staticmethod
    def step_positive(df: pd.DataFrame) -> Tuple[bool, str]:
        """step is an hourly tick starting at 1."""
        invalid = df["step"] < 1
        if invalid.any():
            return False, f"Found {int(invalid.sum())} rows with step < 1."
        return True, ""

    @staticmethod
    def type_valid(df: pd.DataFrame) -> Tuple[bool, str]:
        """type must be one of PaySim's five transaction types."""
        invalid = set(df["type"].dropna().unique()) - SchemaRules.VALID_TYPES
        if invalid:
            return False, f"Invalid type values found: {sorted(invalid)}"
        return True, ""

    @staticmethod
    def isFraud_binary(df: pd.DataFrame) -> Tuple[bool, str]:
        """isFraud (target) must be binary: 0 or 1."""
        invalid = set(df["isFraud"].dropna().unique()) - {0, 1}
        if invalid:
            return False, f"Invalid isFraud values found: {invalid}. Must be 0 or 1."
        return True, ""

    @staticmethod
    def no_duplicate_rows(df: pd.DataFrame) -> Tuple[bool, str]:
        """No fully duplicate rows."""
        duplicated = df.duplicated()
        if duplicated.any():
            return False, f"Found {int(duplicated.sum())} fully duplicate rows."
        return True, ""


class DataQualityRules:
    """Data quality and completeness rules."""

    @staticmethod
    def no_missing_values(df: pd.DataFrame) -> Tuple[bool, list]:
        """
        PaySim is fully populated. Any null means the file is not the
        dataset this pipeline was built and validated for (corrupt download,
        wrong file, hand edit), so halt instead of imputing around it.
        """
        counts = df.isnull().sum()
        counts = counts[counts > 0]
        if len(counts):
            return False, [
                f"Found {int(n)} null values in '{col}'." for col, n in counts.items()
            ]
        return True, []

    @staticmethod
    def has_fraud_in_both_classes(df: pd.DataFrame) -> Tuple[bool, str]:
        """A dataset with only one class cannot train a classifier."""
        if df["isFraud"].nunique() < 2:
            return False, "isFraud has only one class; need both fraud and non-fraud rows."
        return True, ""

    @staticmethod
    def minimum_dataset_size(df: pd.DataFrame, min_rows: int = 100) -> Tuple[bool, str]:
        """Dataset must have minimum number of rows."""
        if len(df) < min_rows:
            return False, f"Dataset has {len(df)} rows, but minimum required is {min_rows}."
        return True, ""


# Rules registry: defines all validation rules and their order.
VALIDATION_RULES = {
    # Schema checks (fail fast - if schema is wrong, nothing else matters)
    "required_columns": SchemaRules.validate_required_columns,
    "datatypes": SchemaRules.validate_datatypes,
    "no_unexpected_columns": SchemaRules.validate_no_unexpected_columns,

    # Data quality checks
    "no_missing_values": DataQualityRules.no_missing_values,
    "minimum_size": DataQualityRules.minimum_dataset_size,
    "both_classes_present": DataQualityRules.has_fraud_in_both_classes,

    # Value-domain checks
    "amount_non_negative": FeatureRules.amount_non_negative,
    "oldbalanceOrg_non_negative": FeatureRules.oldbalanceOrg_non_negative,
    "newbalanceOrig_non_negative": FeatureRules.newbalanceOrig_non_negative,
    "oldbalanceDest_non_negative": FeatureRules.oldbalanceDest_non_negative,
    "newbalanceDest_non_negative": FeatureRules.newbalanceDest_non_negative,
    "step_positive": FeatureRules.step_positive,
    "type_valid": FeatureRules.type_valid,
    "isFraud_binary": FeatureRules.isFraud_binary,
    "no_duplicate_rows": FeatureRules.no_duplicate_rows,
}
