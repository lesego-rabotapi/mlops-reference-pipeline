"""
Feature engineering configuration (PaySim dataset).
Column lists and transformation parameters live here, not in pipeline code.
Changing what gets transformed means changing config, not application logic.
"""

TARGET_COLUMN = "isFraud"

# Time column. Used ONLY to split train/test chronologically, never as a
# model feature -- a model that keys on "step" learns the calendar, not
# fraud, and cannot generalize to future time.
TIME_COLUMN = "step"

# In PaySim, fraud occurs only in these two transaction types (0 fraud in
# CASH_IN / DEBIT / PAYMENT; see docs/DATASET_ASSESSMENT.md). The model is
# scoped to them: the feature stage filters to them, and the serving schema
# rejects any other type. Reported metrics describe this population only.
SCORED_TRANSACTION_TYPES = ["TRANSFER", "CASH_OUT"]

# Dropped before modeling:
# - nameOrig / nameDest: high-cardinality account IDs, no generalizable signal.
# - isFlaggedFraud: output of the simulator's own rule engine; leaks the label
#   logic and would not exist as an independent input in production.
# - step: time column, used for the split only (see TIME_COLUMN).
COLUMNS_TO_DROP = ["nameOrig", "nameDest", "isFlaggedFraud", "step"]

# Raw numeric inputs, followed by the two engineered balance-error features
# (added by src/features/engineering.py). In PaySim, a legitimate transaction
# moves balances consistently (old - amount = new); fraud frequently leaves
# the destination balance unchanged or the origin account zeroed out, so the
# discrepancy is the signal, not the raw balances.
NUMERIC_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "errorBalanceOrig",
    "errorBalanceDest",
]

CATEGORICAL_FEATURES = ["type"]

# Chronological split: steps <= TRAIN_STEP_CUTOFF train, later steps test.
# PaySim spans 743 hourly steps (~30 days); 600 leaves roughly the last
# 19% of time as the held-out future. A random split would leak across time.
TRAIN_STEP_CUTOFF = 600

RANDOM_SEED = 42
