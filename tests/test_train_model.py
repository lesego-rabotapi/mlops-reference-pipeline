"""
Contract tests for the isolated training and evaluation stage.

Why this file exists: protects reproducibility, artifact governance, and binary
inference outputs from regressions.
Responsible for: exercising training only against temporary upstream contracts.
Must not: depend on repository data or run validation/feature engineering.
"""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

import src.training.train_model as training


def configure_temporary_paths(tmp_path, monkeypatch) -> dict[str, object]:
    """Route every training input and output to an isolated test directory."""
    processed_dir = tmp_path / "processed"
    feature_artifacts_dir = tmp_path / "feature_artifacts"
    training_artifacts_dir = tmp_path / "training_artifacts"
    paths = {
        "train_features": processed_dir / "X_train.csv",
        "test_features": processed_dir / "X_test.csv",
        "train_target": processed_dir / "y_train.csv",
        "test_target": processed_dir / "y_test.csv",
        "preprocessor": feature_artifacts_dir / "preprocessor.joblib",
        "model": training_artifacts_dir / "model.joblib",
        "report": training_artifacts_dir / "evaluation_report.json",
        "manifest": training_artifacts_dir / "manifest.json",
        "mlflow": tmp_path / ".mlflow",
        "training_artifacts": training_artifacts_dir,
    }
    monkeypatch.setattr(training, "TRAIN_FEATURES_PATH", paths["train_features"])
    monkeypatch.setattr(training, "TEST_FEATURES_PATH", paths["test_features"])
    monkeypatch.setattr(training, "TRAIN_TARGET_PATH", paths["train_target"])
    monkeypatch.setattr(training, "TEST_TARGET_PATH", paths["test_target"])
    monkeypatch.setattr(training, "PREPROCESSOR_PATH", paths["preprocessor"])
    monkeypatch.setattr(training, "MODEL_PATH", paths["model"])
    monkeypatch.setattr(training, "EVALUATION_REPORT_PATH", paths["report"])
    monkeypatch.setattr(training, "MANIFEST_PATH", paths["manifest"])
    monkeypatch.setattr(training, "MLFLOW_TRACKING_DIR", paths["mlflow"])
    monkeypatch.setattr(training, "TRAINING_ARTIFACTS_DIR", paths["training_artifacts"])
    return paths


@pytest.fixture
def trained_artifacts(tmp_path, monkeypatch):
    """Publish a compact, balanced processed-data contract through training."""
    paths = configure_temporary_paths(tmp_path, monkeypatch)
    processed_dir = paths["train_features"].parent
    preprocessor_path = paths["preprocessor"]
    processed_dir.mkdir(parents=True)
    preprocessor_path.parent.mkdir(parents=True)

    X_train = pd.DataFrame(
        {
            "numeric_feature": list(range(40)),
            "encoded_feature": [value % 3 for value in range(40)],
        }
    )
    y_train = pd.DataFrame({"Exited": [0, 1] * 20})
    X_test = pd.DataFrame(
        {
            "numeric_feature": list(range(40, 60)),
            "encoded_feature": [value % 3 for value in range(40, 60)],
        }
    )
    y_test = pd.DataFrame({"Exited": [0, 1] * 10})
    X_train.to_csv(paths["train_features"], index=False)
    X_test.to_csv(paths["test_features"], index=False)
    y_train.to_csv(paths["train_target"], index=False)
    y_test.to_csv(paths["test_target"], index=False)
    joblib.dump({"test_preprocessor": True}, preprocessor_path)

    metrics = training.run_training()
    return paths, X_test, metrics


def test_training_creates_model_and_evaluation_artifacts(trained_artifacts):
    """The Stage 4 evidence files exist and expose the documented metric contract."""
    paths, _, metrics = trained_artifacts

    assert paths["model"].exists()
    assert paths["report"].exists()
    assert set(metrics) == set(training.METRIC_NAMES)

    with paths["report"].open(encoding="utf-8") as report_file:
        report = json.load(report_file)

    assert set(report) == set(training.METRIC_NAMES)


def test_training_writes_manifest_and_binary_integer_predictions(trained_artifacts):
    """The serving manifest binds both artifacts and model predictions stay binary."""
    paths, X_test, _ = trained_artifacts

    assert paths["manifest"].exists()
    with paths["manifest"].open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    assert set(manifest) == {
        "model_hash",
        "preprocessor_hash",
        "sklearn_version",
        "trained_at",
    }
    assert manifest["model_hash"] == training.sha256_file(paths["model"])
    assert manifest["preprocessor_hash"] == training.sha256_file(paths["preprocessor"])

    model = joblib.load(paths["model"])
    predictions = model.predict(X_test)
    assert np.issubdtype(predictions.dtype, np.integer)
    assert set(predictions).issubset({0, 1})


def test_training_fails_without_processed_inputs(tmp_path, monkeypatch):
    """Missing upstream data aborts before a partial model or report is published."""
    paths = configure_temporary_paths(tmp_path, monkeypatch)

    with pytest.raises(FileNotFoundError, match="Training inputs are incomplete"):
        training.run_training()

    assert not paths["training_artifacts"].exists()
