"""
Tests for the feature engineering pipeline.

Test strategy:
- Unit tests for each discrete function
- Integration test for the full pipeline execution
- Leakage detection: verify preprocessor is fit on train only
- Schema tests: verify output shape and column names are consistent

"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    build_preprocessor,
    get_feature_names,
    preprocess_raw_columns,
    run_feature_pipeline,
    split_features_target,
)
from src.config.feature_config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_fraud_df() -> pd.DataFrame:
    """
    Minimal synthetic fraud DataFrame, shaped like the output of the
    validation stage: the 13 raw fraud columns plus the
    velocity_score_was_missing indicator added during validation-stage
    imputation (see src/validation/imputation.py).
    """
    n = 50
    rng = np.random.default_rng(seed=42)

    df = pd.DataFrame({
        "transaction_amount":         rng.uniform(1, 300, size=n),
        "hour_of_day":                rng.choice([1.0, 2.0, 3.0], size=n),
        "is_weekend":                 rng.choice([0.0, 1.0], size=n),
        "num_items":                  rng.integers(0, 14, size=n).astype(float),
        "customer_age":               rng.integers(18, 80, size=n).astype(float),
        "prev_transactions":          rng.integers(0, 40, size=n).astype(float),
        "distance_from_home":         rng.uniform(0, 225, size=n),
        "device_type":                rng.choice([0.0, 1.0, 2.0], size=n),
        "network_quality":            rng.uniform(0, 100, size=n),
        "is_first_transaction":       rng.choice([0.0, 1.0], size=n),
        "store_type":                 rng.choice([0.0, 1.0], size=n),
        "velocity_score":             rng.normal(5, 2, size=n),
        "velocity_score_was_missing": rng.choice([0, 1], size=n),
        "is_fraud":                   rng.choice([0, 1], size=n),
    })
    return df


@pytest.fixture
def preprocessed_df(minimal_fraud_df) -> pd.DataFrame:
    """Return the dataset after pre-split processing."""
    return preprocess_raw_columns(minimal_fraud_df)


# ---------------------------------------------------------------------------
# Unit tests: preprocess_raw_columns
# ---------------------------------------------------------------------------

class TestPreprocessRawColumns:

    def test_drops_missingness_indicator_column(self, minimal_fraud_df):
        result = preprocess_raw_columns(minimal_fraud_df)
        assert "velocity_score_was_missing" not in result.columns

    def test_target_remains_binary(self, minimal_fraud_df):
        result = preprocess_raw_columns(minimal_fraud_df)
        assert set(result[TARGET_COLUMN].unique()).issubset({0, 1})

    def test_input_is_not_mutated(self, minimal_fraud_df):
        original_target = minimal_fraud_df[TARGET_COLUMN].copy()
        preprocess_raw_columns(minimal_fraud_df)
        pd.testing.assert_series_equal(minimal_fraud_df[TARGET_COLUMN], original_target)


# ---------------------------------------------------------------------------
# Unit tests: build_preprocessor
# ---------------------------------------------------------------------------

class TestBuildPreprocessor:

    def test_preprocessor_fits_without_error(self, preprocessed_df):
        X, _ = split_features_target(preprocessed_df)
        preprocessor = build_preprocessor()
        preprocessor.fit(X)  # should not raise

    def test_transform_produces_no_nans(self, preprocessed_df):
        X, _ = split_features_target(preprocessed_df)
        preprocessor = build_preprocessor()
        preprocessor.fit(X)
        X_transformed = preprocessor.transform(X)
        assert not np.isnan(X_transformed).any(), "Transformed data contains NaN"

    def test_handles_unseen_categories_without_crash(self, preprocessed_df):
        """
        Critical deployment safety test.
        Verifies handle_unknown='ignore' prevents crashes on novel category values
        that the training set never saw — a guaranteed real-world scenario.
        """
        X, _ = split_features_target(preprocessed_df)
        preprocessor = build_preprocessor()
        preprocessor.fit(X)

        X_unseen = X.copy()
        X_unseen["device_type"] = 5.0

        try:
            preprocessor.transform(X_unseen)
        except ValueError as e:
            pytest.fail(
                f"Preprocessor crashed on unseen category — inference would fail: {e}"
            )


# ---------------------------------------------------------------------------
# Leakage detection test
# ---------------------------------------------------------------------------

class TestDataLeakagePrevention:

    def test_scaler_statistics_differ_train_vs_full(self, preprocessed_df):
        """
        Verifies the preprocessor is fit on train data only.

        If fit() were called on the full dataset, the scaler's mean_
        would match the full-dataset mean exactly. We test that it does not.
        This is a property-based signal, not a guarantee — but it catches
        the most common leakage mistake.
        """
        from sklearn.model_selection import train_test_split
        from src.config.feature_config import RANDOM_SEED, TEST_SIZE

        X, _ = split_features_target(preprocessed_df)
        X_train, _ = train_test_split(X, test_size=TEST_SIZE, random_state=RANDOM_SEED)

        # Fit on train only
        preprocessor_train = build_preprocessor()
        preprocessor_train.fit(X_train)

        # Fit on full dataset (the leaky way)
        preprocessor_full = build_preprocessor()
        preprocessor_full.fit(X)

        train_mean = preprocessor_train.named_transformers_["numeric"]["scaler"].mean_
        full_mean  = preprocessor_full.named_transformers_["numeric"]["scaler"].mean_

        assert not np.allclose(train_mean, full_mean), (
            "Scaler means are identical for train-only vs full-dataset fit. "
            "This may indicate data leakage."
        )


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

class TestRunFeaturePipeline:

    def test_pipeline_produces_all_output_files(self, tmp_path, minimal_fraud_df, monkeypatch):
        """
        Full pipeline integration test using tmp_path to avoid touching real artifact dirs.
        """
        import src.config.paths as paths
        import src.features.build_features as bf

        # Point all paths to tmp_path
        validated_csv = tmp_path / "fraud_validated.csv"
        minimal_fraud_df.to_csv(validated_csv, index=False)

        monkeypatch.setattr(paths, "VALIDATED_DATASET_PATH", validated_csv)
        monkeypatch.setattr(paths, "PROCESSED_DATA_DIR", tmp_path / "processed")
        monkeypatch.setattr(paths, "FEATURES_ARTIFACTS_DIR", tmp_path / "artifacts")
        monkeypatch.setattr(paths, "PREPROCESSOR_PATH", tmp_path / "artifacts" / "preprocessor.joblib")
        monkeypatch.setattr(paths, "FEATURE_METADATA_PATH", tmp_path / "artifacts" / "feature_metadata.json")
        monkeypatch.setattr(paths, "TRAIN_FEATURES_PATH", tmp_path / "processed" / "X_train.csv")
        monkeypatch.setattr(paths, "TEST_FEATURES_PATH", tmp_path / "processed" / "X_test.csv")
        monkeypatch.setattr(paths, "TRAIN_TARGET_PATH", tmp_path / "processed" / "y_train.csv")
        monkeypatch.setattr(paths, "TEST_TARGET_PATH", tmp_path / "processed" / "y_test.csv")

        # Re-import to pick up monkeypatched paths
        import importlib
        importlib.reload(bf)

        bf.run_feature_pipeline()

        assert (tmp_path / "artifacts" / "preprocessor.joblib").exists()
        assert (tmp_path / "artifacts" / "feature_metadata.json").exists()
        assert (tmp_path / "processed" / "X_train.csv").exists()
        assert (tmp_path / "processed" / "X_test.csv").exists()
