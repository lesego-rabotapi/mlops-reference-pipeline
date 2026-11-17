"""
Tests for the feature stage: scoping, engineered features, the chronological
split, and the preprocessor.
"""

import json

import numpy as np
import pandas as pd
import pytest

import src.features.build_features as bf
from src.config.feature_config import (
    CATEGORICAL_FEATURES,
    COLUMNS_TO_DROP,
    NUMERIC_FEATURES,
    SCORED_TRANSACTION_TYPES,
    TARGET_COLUMN,
    TRAIN_STEP_CUTOFF,
)
from src.features.engineering import add_engineered_features


def test_engineered_features_are_zero_for_consistent_balances():
    df = pd.DataFrame(
        {
            "amount": [100.0],
            "oldbalanceOrg": [500.0],
            "newbalanceOrig": [400.0],
            "oldbalanceDest": [50.0],
            "newbalanceDest": [150.0],
        }
    )
    out = add_engineered_features(df)
    assert out.loc[0, "errorBalanceOrig"] == 0
    assert out.loc[0, "errorBalanceDest"] == 0


def test_engineered_features_flag_a_drained_origin_and_unchanged_destination():
    df = pd.DataFrame(
        {
            "amount": [500.0],
            "oldbalanceOrg": [500.0],
            "newbalanceOrig": [0.0],  # drained, but consistent with amount
            "oldbalanceDest": [50.0],
            "newbalanceDest": [50.0],  # destination never credited
        }
    )
    out = add_engineered_features(df)
    assert out.loc[0, "errorBalanceOrig"] == 0
    assert out.loc[0, "errorBalanceDest"] == 500.0


def test_engineered_features_do_not_mutate_input():
    df = pd.DataFrame(
        {c: [1.0] for c in
         ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]}
    )
    add_engineered_features(df)
    assert "errorBalanceOrig" not in df.columns


def test_preprocess_scopes_to_scored_types_and_adds_features(paysim_df):
    out = bf.preprocess_raw_columns(paysim_df)
    assert set(out["type"]) <= set(SCORED_TRANSACTION_TYPES)
    assert {"errorBalanceOrig", "errorBalanceDest"} <= set(out.columns)
    # step is still present: the split needs it
    assert "step" in out.columns


def test_all_fraud_in_dataset_is_in_scored_types(paysim_df):
    """Guards the scoping assumption: filtering must not discard fraud."""
    fraud_types = set(paysim_df.loc[paysim_df["isFraud"] == 1, "type"])
    assert fraud_types <= set(SCORED_TRANSACTION_TYPES)


def _split(df):
    X, y = bf.split_features_target(bf.preprocess_raw_columns(df))
    return X, y, *bf.split_by_time(X, y)


def test_time_split_is_strictly_chronological(paysim_df):
    X, _, X_train, X_test, y_train, y_test = _split(paysim_df)
    steps = paysim_df.loc[X.index, "step"]
    assert steps.loc[X_train.index].max() <= TRAIN_STEP_CUTOFF
    assert steps.loc[X_test.index].min() > TRAIN_STEP_CUTOFF
    assert set(X_train.index).isdisjoint(X_test.index)
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) == len(X_train) and len(y_test) == len(X_test)


def test_time_split_drops_non_feature_columns(paysim_df):
    _, _, X_train, X_test, _, _ = _split(paysim_df)
    for frame in (X_train, X_test):
        assert not set(COLUMNS_TO_DROP) & set(frame.columns)
        assert TARGET_COLUMN not in frame.columns
        assert set(frame.columns) == set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)


def test_time_split_raises_when_a_side_has_no_fraud(paysim_df):
    X, y = bf.split_features_target(bf.preprocess_raw_columns(paysim_df))
    y = y.copy()
    y[X["step"] > TRAIN_STEP_CUTOFF] = 0
    with pytest.raises(ValueError, match="only one class"):
        bf.split_by_time(X, y)


def test_time_split_raises_on_empty_side(paysim_df):
    X, y = bf.split_features_target(bf.preprocess_raw_columns(paysim_df))
    with pytest.raises(ValueError, match="empty side"):
        bf.split_by_time(X, y, cutoff=10_000)


def test_preprocessor_fits_on_train_only_and_handles_unseen_category(paysim_df):
    _, _, X_train, X_test, _, _ = _split(paysim_df)
    pre = bf.build_preprocessor().fit(X_train)

    out = pre.transform(X_test)
    names = bf.get_feature_names(pre)
    assert out.shape == (len(X_test), len(names))
    assert not np.isnan(out).any()

    # fitted on training statistics, not test: train means ~0, test need not be
    train_out = pre.transform(X_train)
    assert abs(train_out[:, 0].mean()) < 1e-9

    unseen = X_test.head(1).copy()
    unseen["type"] = "NEW_TYPE"
    assert not np.isnan(pre.transform(unseen)).any()  # handle_unknown="ignore"


def test_full_feature_pipeline_writes_artifacts(paysim_df, tmp_path, monkeypatch):
    paths = {
        "VALIDATED_DATASET_PATH": tmp_path / "validated.csv",
        "FEATURES_ARTIFACTS_DIR": tmp_path / "features",
        "PREPROCESSOR_PATH": tmp_path / "features" / "preprocessor.joblib",
        "FEATURE_METADATA_PATH": tmp_path / "features" / "feature_metadata.json",
        "PROCESSED_DATA_DIR": tmp_path / "processed",
        "TRAIN_FEATURES_PATH": tmp_path / "processed" / "X_train.csv",
        "TEST_FEATURES_PATH": tmp_path / "processed" / "X_test.csv",
        "TRAIN_TARGET_PATH": tmp_path / "processed" / "y_train.csv",
        "TEST_TARGET_PATH": tmp_path / "processed" / "y_test.csv",
    }
    for name, value in paths.items():
        monkeypatch.setattr(bf, name, value)
    paysim_df.to_csv(paths["VALIDATED_DATASET_PATH"], index=False)

    bf.run_feature_pipeline()

    assert paths["PREPROCESSOR_PATH"].exists()
    metadata = json.loads(paths["FEATURE_METADATA_PATH"].read_text())
    assert metadata["train_step_cutoff"] == TRAIN_STEP_CUTOFF
    X_train = pd.read_csv(paths["TRAIN_FEATURES_PATH"])
    y_train = pd.read_csv(paths["TRAIN_TARGET_PATH"])
    assert len(X_train) == len(y_train)
    assert X_train.shape[1] == metadata["n_features_out"]


def test_engineered_features_are_rounded_to_cents():
    """Float residuals on large balances must not leak in as tiny nonzero values."""
    df = pd.DataFrame(
        {
            "amount": [0.1],
            "oldbalanceOrg": [9_999_999.3],
            "newbalanceOrig": [9_999_999.2],
            "oldbalanceDest": [12_345_678.1],
            "newbalanceDest": [12_345_678.2],
        }
    )
    out = add_engineered_features(df)
    assert out.loc[0, "errorBalanceOrig"] == 0.0
    assert out.loc[0, "errorBalanceDest"] == 0.0
