"""
Feature engineering configuration.
Column lists and transformation parameters live here, not in pipeline code.
Changing what gets transformed means changing config, not application logic.
"""

TARGET_COLUMN = "Exited"

# Columns to drop — identifiers have no predictive signal
COLUMNS_TO_DROP = ["id", "CustomerId", "Surname"]

NUMERIC_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "EstimatedSalary",
]

CATEGORICAL_FEATURES = [
    "Geography",
    "Gender",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
]

# Split configuration
TEST_SIZE   = 0.2
RANDOM_SEED = 42