"""
Feature engineering configuration.
Column lists and transformation parameters live here, not in pipeline code.
Changing what gets transformed means changing config, not application logic.
"""

TARGET_COLUMN = "is_fraud"

# Columns to drop before feeding the model.
# velocity_score_was_missing is a validation-stage audit artifact (see
# src/validation/imputation.py) -- current MCAR analysis found it isn't
# fraud-informative, so it's dropped by default here rather than fed to the
# model. Reversible: move it into NUMERIC_FEATURES if that evidence changes.
COLUMNS_TO_DROP = ["velocity_score_was_missing"]

NUMERIC_FEATURES = [
    "transaction_amount",
    "num_items",
    "customer_age",
    "prev_transactions",
    "distance_from_home",
    "network_quality",
    "velocity_score",
]

# hour_of_day is a 3-level time-of-day bucket (values {1,2,3}), not a raw
# 0-23 hour, so it belongs here rather than in NUMERIC_FEATURES.
CATEGORICAL_FEATURES = [
    "hour_of_day",
    "is_weekend",
    "device_type",
    "is_first_transaction",
    "store_type",
]

# Split configuration
TEST_SIZE   = 0.2
RANDOM_SEED = 42
