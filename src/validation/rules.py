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
from typing import Tuple


class SchemaRules:
    """Schema and datatype validation."""
    
    # Define the expected schema as a class constant
    # This becomes the single source of truth for schema
    REQUIRED_SCHEMA = {
        "id": "integer",
        "CustomerId": "integer",
        "Surname": "string",
        "CreditScore": "integer",
        "Geography": "string",
        "Gender": "string",
        "Age": "float",
        "Tenure": "integer",
        "Balance": "float",
        "NumOfProducts": "integer",
        "HasCrCard": "float",
        "IsActiveMember": "float",
        "EstimatedSalary": "float",
        "Exited": "integer",
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
    """Business logic and feature validation rules."""
    
    @staticmethod
    def tenure_non_negative(df: pd.DataFrame) -> Tuple[bool, str]:
        """Tenure cannot be negative."""
        if (df["Tenure"] < 0).any():
            count = (df["Tenure"] < 0).sum()
            return False, f"Found {count} rows with negative Tenure."
        return True, ""
    
    @staticmethod
    def age_in_valid_range(df: pd.DataFrame) -> Tuple[bool, str]:
        """Age must be between 0 and 120 (realistic human age range)."""
        invalid = (df["Age"] < 0) | (df["Age"] > 120)
        
        if invalid.any():
            count = invalid.sum()
            return False, f"Found {count} rows with invalid Age values."
        
        return True, ""
    
    @staticmethod
    def credit_score_valid(df: pd.DataFrame) -> Tuple[bool, str]:
        """Credit score must be within standard range (300-850)."""
        invalid = (df["CreditScore"] < 300) | (df["CreditScore"] > 850)
        
        if invalid.any():
            count = invalid.sum()
            return False, f"Found {count} rows with invalid CreditScore (not in 300-850)."
        
        return True, ""
    
    @staticmethod
    def balance_non_negative(df: pd.DataFrame) -> Tuple[bool, str]:
        """Balance cannot be negative."""
        if (df["Balance"] < 0).any():
            count = (df["Balance"] < 0).sum()
            return False, f"Found {count} rows with negative Balance."
        
        return True, ""
    
    @staticmethod
    def salary_non_negative(df: pd.DataFrame) -> Tuple[bool, str]:
        """Estimated salary cannot be negative."""
        if (df["EstimatedSalary"] < 0).any():
            count = (df["EstimatedSalary"] < 0).sum()
            return False, f"Found {count} rows with negative EstimatedSalary."
        
        return True, ""
    
    @staticmethod
    def geography_valid(df: pd.DataFrame) -> Tuple[bool, str]:
        """Geography must be one of: France, Spain, Germany."""
        valid_geographies = {"France", "Spain", "Germany"}
        invalid = set(df["Geography"].unique()) - valid_geographies
        
        if invalid:
            return False, f"Invalid Geography values found: {invalid}"
        
        return True, ""
    
    @staticmethod
    def gender_valid(df: pd.DataFrame) -> Tuple[bool, str]:
        """Gender must be either Male or Female."""
        valid_genders = {"Male", "Female"}
        invalid = set(df["Gender"].unique()) - valid_genders
        
        if invalid:
            return False, f"Invalid Gender values found: {invalid}"
        
        return True, ""
    
    @staticmethod
    def exited_binary(df: pd.DataFrame) -> Tuple[bool, str]:
        """Exited (target) must be binary: 0 or 1."""
        valid_values = {0, 1}
        invalid = set(df["Exited"].unique()) - valid_values
        
        if invalid:
            return False, f"Invalid Exited values found: {invalid}. Must be 0 or 1."
        
        return True, ""
    
    @staticmethod
    def no_duplicate_record_ids(df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Raw row identifier must be unique.

        In this dataset CustomerId is not a reliable primary key because the
        source contains repeated CustomerId values. The id column is the record
        identity used to detect duplicate raw rows.
        """
        if df["id"].duplicated().any():
            count = df["id"].duplicated().sum()
            return False, f"Found {count} duplicate id values."
        
        return True, ""
    
    @staticmethod
    def reasonable_num_products(df: pd.DataFrame) -> Tuple[bool, str]:
        """Number of products should be between 1 and 4 (realistic constraint)."""
        invalid = (df["NumOfProducts"] < 1) | (df["NumOfProducts"] > 4)
        
        if invalid.any():
            count = invalid.sum()
            return False, f"Found {count} rows with unrealistic NumOfProducts (not 1-4)."
        
        return True, ""


class DataQualityRules:
    """Data quality and completeness rules."""
    
    @staticmethod
    def no_null_values(df: pd.DataFrame) -> Tuple[bool, str]:
        """No null values allowed in any column."""
        nulls = df.isnull().sum()
        
        if nulls.sum() > 0:
            null_cols = nulls[nulls > 0]
            error_msg = ", ".join(
                [f"{col}({count})" for col, count in null_cols.items()]
            )
            return False, f"Found null values in columns: {error_msg}"
        
        return True, ""
    
    @staticmethod
    def minimum_dataset_size(df: pd.DataFrame, min_rows: int = 100) -> Tuple[bool, str]:
        """Dataset must have minimum number of rows."""
        if len(df) < min_rows:
            return False, f"Dataset has {len(df)} rows, but minimum required is {min_rows}."
        
        return True, ""


# Rules registry: defines all validation rules and their order
# Used by validator to run all checks systematically
VALIDATION_RULES = {
    # Schema checks (fail fast - if schema is wrong, nothing else matters)
    "required_columns": SchemaRules.validate_required_columns,
    "datatypes": SchemaRules.validate_datatypes,
    "no_unexpected_columns": SchemaRules.validate_no_unexpected_columns,
    
    # Data quality checks (completeness)
    "no_nulls": DataQualityRules.no_null_values,
    "minimum_size": DataQualityRules.minimum_dataset_size,
    
    # Feature business logic checks
    "tenure_non_negative": FeatureRules.tenure_non_negative,
    "age_valid": FeatureRules.age_in_valid_range,
    "credit_score_valid": FeatureRules.credit_score_valid,
    "balance_non_negative": FeatureRules.balance_non_negative,
    "salary_non_negative": FeatureRules.salary_non_negative,
    "geography_valid": FeatureRules.geography_valid,
    "gender_valid": FeatureRules.gender_valid,
    "exited_binary": FeatureRules.exited_binary,
    "no_duplicate_ids": FeatureRules.no_duplicate_record_ids,
    "reasonable_products": FeatureRules.reasonable_num_products,
}
