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
}
MLFLOW_EXPERIMENT_NAME = "fraud_training"
