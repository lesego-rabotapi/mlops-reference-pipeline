"""
Training and evaluation stage for the fraud-detection pipeline.

Why this file exists: turns the processed-feature contract into a reproducible
model, evaluation evidence, and serving-compatibility manifest.
Responsible for: loading processed data, training, evaluating, and publishing
training artifacts and MLflow evidence.
Must not: validate raw data, perform feature engineering, or serve predictions.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config.feature_config import RANDOM_SEED
from src.config.paths import (
    EVALUATION_REPORT_PATH,
    MANIFEST_PATH,
    MLFLOW_TRACKING_DIR,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    TEST_FEATURES_PATH,
    TEST_TARGET_PATH,
    TRAIN_FEATURES_PATH,
    TRAIN_TARGET_PATH,
    TRAINING_ARTIFACTS_DIR,
)
from src.config.training_config import MLFLOW_EXPERIMENT_NAME, RANDOM_FOREST_PARAMS


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "roc_auc")


def assert_required_artifacts(paths: tuple[Path, ...]) -> None:
    """Fail before training if its upstream feature contract is incomplete."""
    missing_paths = [str(path) for path in paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Training inputs are incomplete. Run the feature engineering stage first. "
            f"Missing: {', '.join(missing_paths)}"
        )


def load_target(path: Path) -> pd.Series:
    """Load one persisted target column and reject ambiguous CSV contracts."""
    target_frame = pd.read_csv(path)
    if target_frame.shape[1] != 1:
        raise ValueError(
            f"Target file {path} must contain exactly one column; "
            f"found {target_frame.shape[1]}."
        )
    return target_frame.iloc[:, 0]


def load_training_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load and verify the processed training/test contract written upstream."""
    required_paths = (
        TRAIN_FEATURES_PATH,
        TEST_FEATURES_PATH,
        TRAIN_TARGET_PATH,
        TEST_TARGET_PATH,
        PREPROCESSOR_PATH,
    )
    assert_required_artifacts(required_paths)

    X_train = pd.read_csv(TRAIN_FEATURES_PATH)
    X_test = pd.read_csv(TEST_FEATURES_PATH)
    y_train = load_target(TRAIN_TARGET_PATH)
    y_test = load_target(TEST_TARGET_PATH)

    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError(
            "Feature and target row counts do not match. Re-run feature engineering "
            "rather than training on misaligned samples."
        )

    invalid_targets = set(pd.concat([y_train, y_test]).dropna().unique()) - {0, 1}
    if invalid_targets or y_train.isna().any() or y_test.isna().any():
        raise ValueError(
            "Training targets must be non-null binary integers (0 or 1); "
            f"found invalid values: {sorted(map(str, invalid_targets))}."
        )

    logger.info(
        "Loaded processed contract: train=%d rows, test=%d rows, features=%d",
        len(X_train),
        len(X_test),
        X_train.shape[1],
    )
    return X_train, X_test, y_train, y_test


def build_model() -> RandomForestClassifier:
    """Create the deterministic baseline classifier from explicit configuration."""
    return RandomForestClassifier(**RANDOM_FOREST_PARAMS)


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Compute imbalance-aware evaluation evidence on the held-out test set."""
    if y_test.nunique() < 2:
        raise ValueError("ROC-AUC requires both fraud classes in the test target.")

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }
    logger.info(
        "Evaluation complete: precision=%.4f recall=%.4f roc_auc=%.4f accuracy=%.4f",
        metrics["precision"],
        metrics["recall"],
        metrics["roc_auc"],
        metrics["accuracy"],
    )
    return metrics


def sha256_file(path: Path) -> str:
    """Calculate an artifact's immutable content fingerprint."""
    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_evaluation_report(metrics: dict[str, float]) -> None:
    """Persist local evaluation evidence for review without requiring MLflow UI."""
    with EVALUATION_REPORT_PATH.open("w", encoding="utf-8") as report_file:
        json.dump(metrics, report_file, indent=2)
    logger.info("Evaluation report saved: %s", EVALUATION_REPORT_PATH)


def save_manifest() -> None:
    """Bind the trained model to the exact preprocessor used to create its inputs."""
    manifest = {
        "model_hash": sha256_file(MODEL_PATH),
        "preprocessor_hash": sha256_file(PREPROCESSOR_PATH),
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with MANIFEST_PATH.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
    logger.info("Artifact manifest saved: %s", MANIFEST_PATH)


def configure_mlflow() -> None:
    """Use a project-local, named experiment so runs do not depend on cwd."""
    MLFLOW_TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_DIR.resolve().as_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def run_training() -> dict[str, float]:
    """Orchestrate the isolated training stage for CLI, Makefile, and CI use."""
    logger.info("=== Training and Evaluation START ===")
    X_train, X_test, y_train, y_test = load_training_data()
    TRAINING_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    configure_mlflow()

    with mlflow.start_run():
        model = build_model()
        model.fit(X_train, y_train)
        logger.info("RandomForestClassifier fitted with random_state=%d", RANDOM_SEED)

        metrics = evaluate_model(model, X_test, y_test)
        joblib.dump(model, MODEL_PATH)
        logger.info("Model saved: %s", MODEL_PATH)
        save_evaluation_report(metrics)
        save_manifest()

        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.set_tag("model_artifact_path", str(MODEL_PATH))
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")
        mlflow.log_artifact(str(EVALUATION_REPORT_PATH), artifact_path="evaluation")
        mlflow.log_artifact(str(MANIFEST_PATH), artifact_path="governance")

    logger.info("=== Training and Evaluation COMPLETE ===")
    return metrics


def main() -> None:
    """CLI entrypoint used by Makefile and CI/CD."""
    try:
        run_training()
    except Exception as exc:
        logger.exception("Training failed before completion: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
