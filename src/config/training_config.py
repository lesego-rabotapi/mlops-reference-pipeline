"""
Explicit configuration for the isolated training stage.

Why this file exists: model and experiment choices must be reviewable without
changing training orchestration code.
Responsible for: deterministic estimator settings and MLflow experiment naming.
Must not: load data, train models, or create MLflow runs.
"""

from src.config.feature_config import RANDOM_SEED


# HistGradientBoosting rather than RandomForest: PaySim's in-scope training
# set is ~2M rows, where a 100-tree unlimited-depth forest is slow and
# memory-hungry and a histogram booster is both faster and usually stronger
# on tabular data. class_weight="balanced" is kept for the same reason as
# before: fraud is a rare event (~0.3% of the scored types) and an
# unweighted fit optimizes as if both classes cost the same to get wrong.
MODEL_PARAMS = {
    "max_iter": 200,
    "learning_rate": 0.1,
    "random_state": RANDOM_SEED,
    "class_weight": "balanced",
}

# Probability cutoff for the hard 0/1 prediction. NOT tuned: no cost model
# exists yet (see docs/ENGINEERING_LOG.md open items). The threshold-free
# metrics (pr_auc, recall_at_1pct_alerts) are the ones to compare models on.
DECISION_THRESHOLD = 0.5

# Alert-budget metric: share of transactions an analyst team could review.
ALERT_BUDGET_FRACTION = 0.01

MLFLOW_EXPERIMENT_NAME = "fraud_training"
