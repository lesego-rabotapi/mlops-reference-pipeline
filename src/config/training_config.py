"""
Explicit configuration for the isolated training stage.

Why this file exists: model and experiment choices must be reviewable without
changing training orchestration code.
Responsible for: deterministic estimator settings and MLflow experiment naming.
Must not: load data, train models, or create MLflow runs.
"""

from src.config.feature_config import RANDOM_SEED


RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "random_state": RANDOM_SEED,
    # This is a rare-event classification problem (is_fraud), not a balanced
    # one -- even at this synthetic dataset's inflated 10.3% fraud rate, an
    # unweighted forest optimizes as if both classes cost the same to get
    # wrong. "balanced" reweights each class inversely to its frequency
    # (computed once on the full training set, not per bootstrap sample --
    # "balanced_subsample" is the per-tree variant, unnecessary at this
    # scale/imbalance level). Explicit here because the wrong default is
    # silent: the model still trains and reports a plausible-looking
    # accuracy without it.
    "class_weight": "balanced",
}
MLFLOW_EXPERIMENT_NAME = "fraud_training"
