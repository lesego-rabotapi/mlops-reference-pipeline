"""
Feature Engineering Pipeline

Loads the validated dataset, splits it into train/test before any fitting
happens (splitting after would leak test data into the transformation
statistics), builds and fits a ColumnTransformer on the training data
only, then transforms both sets with it. Saves the fitted preprocessor
for reuse at inference time, plus the processed feature matrices and
targets.

A few design choices worth knowing: fit() only ever runs on the training
split, since fitting on the full dataset would leak test data into the
transformation statistics. OneHotEncoder uses handle_unknown="ignore"
because inference will see category values training never did, and
without that flag it just raises. Numeric imputation uses the median
because it's robust to outliers in production data. Artifacts are saved
with joblib rather than pickle, since sklearn objects serialize cleanly
with it and pickle carries more version risk.
"""

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config.feature_config import (
    CATEGORICAL_FEATURES,
    COLUMNS_TO_DROP,
    NUMERIC_FEATURES,
    RANDOM_SEED,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.config.paths import (
    FEATURES_ARTIFACTS_DIR,
    FEATURE_METADATA_PATH,
    PREPROCESSOR_PATH,
    PROCESSED_DATA_DIR,
    TEST_FEATURES_PATH,
    TEST_TARGET_PATH,
    TRAIN_FEATURES_PATH,
    TRAIN_TARGET_PATH,
    VALIDATED_DATASET_PATH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


def load_validated_data(path: Path) -> pd.DataFrame:
    """
    Load the validated dataset.
    Raises if the file is missing — fail fast rather than silently proceeding.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Validated dataset not found at {path}. "
            "Run the validation stage first."
        )
    df = pd.read_csv(path)
    logger.info("Loaded validated dataset: %d rows, %d columns", *df.shape)
    return df


def preprocess_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply lightweight pre-split fixes that aren't really transformations,
    just dropping identifier columns for now. These are deterministic, with
    no learned statistics involved, so it's safe to do this before the
    split.
    """
    df = df.copy()

    df = df.drop(columns=COLUMNS_TO_DROP, errors="ignore")

    logger.info("Pre-split preprocessing complete.")
    return df


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate feature matrix from target vector."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """
    Build the sklearn ColumnTransformer.

    Numeric columns get median imputation (robust to outliers, unlike the
    mean) followed by StandardScaler, since distance-based models need
    zero-mean, unit-variance input.

    Categorical columns get most-frequent imputation, then a
    OneHotEncoder with handle_unknown="ignore". That flag isn't optional:
    production inference requests will contain category values training
    never saw, and without it the pipeline just raises a ValueError and
    your API returns a 500.

    Both SimpleImputer steps are defense-in-depth, not the primary way
    nulls get handled. Validation (src/validation/imputation.py) already
    fills every column with an evidence-backed, per-column MCAR policy
    before this file's input even gets written, so in normal operation
    these imputers see zero nulls and do nothing. They stay because this
    file reads that validated file directly without rerunning Validation's
    checks, and they're the only thing standing between this pipeline and
    a stale file, a hand-edited one, or a new feature column that shows up
    without a matching ImputationSpec. See docs/ENGINEERING_LOG.md, Entry
    8, for the full reasoning.

    remainder="drop" is written out explicitly: columns not listed here
    are meant to be excluded, not forgotten.
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """
    Get the feature names after fitting.
    One-hot encoding expands categorical columns into several, so we need
    the full output name list to rebuild the DataFrame and for the
    inference layer to use later.
    """
    return list(preprocessor.get_feature_names_out())


def save_artifacts(
    preprocessor: ColumnTransformer,
    feature_names: list[str],
) -> None:
    """
    Save the fitted preprocessor and feature metadata.

    Both are deployment dependencies: the inference service has to load
    the exact same preprocessor that was fitted during training. Get this
    out of sync and you don't get a crash, you get silently wrong
    predictions.
    """
    FEATURES_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    logger.info("Preprocessor saved → %s", PREPROCESSOR_PATH)

    metadata = {
        "feature_names": feature_names,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "n_features_out": len(feature_names),
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
    }
    with open(FEATURE_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Feature metadata saved → %s", FEATURE_METADATA_PATH)


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Save the processed feature matrices and targets as CSVs."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(TRAIN_FEATURES_PATH, index=False)
    X_test.to_csv(TEST_FEATURES_PATH, index=False)
    y_train.to_csv(TRAIN_TARGET_PATH, index=False)
    y_test.to_csv(TEST_TARGET_PATH, index=False)

    logger.info(
        "Processed data saved → %s (train: %d rows, test: %d rows)",
        PROCESSED_DATA_DIR,
        len(X_train),
        len(X_test),
    )


def run_feature_pipeline() -> None:
    """
    Orchestrates the full feature engineering stage.
    Entry point for CLI, Makefile, and CI/CD invocation.
    """
    logger.info("=== Feature Engineering Pipeline START ===")

    # 1. Load
    df = load_validated_data(VALIDATED_DATASET_PATH)

    # 2. Pre-split deterministic preprocessing
    df = preprocess_raw_columns(df)

    # 3. Split features from target
    X, y = split_features_target(df)

    # 4. Train/test split — MUST happen before any fit() call
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    logger.info(
        "Split complete — Train: %d rows | Test: %d rows", len(X_train), len(X_test)
    )

    # 5. Build and fit preprocessor on TRAINING DATA ONLY
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    logger.info("Preprocessor fitted on training data.")

    # 6. Transform both sets using the fitted preprocessor
    feature_names = get_feature_names(preprocessor)

    X_train_processed = pd.DataFrame(
        preprocessor.transform(X_train), columns=feature_names
    )
    X_test_processed = pd.DataFrame(
        preprocessor.transform(X_test), columns=feature_names
    )

    # 7. Persist artifacts
    save_artifacts(preprocessor, feature_names)

    # 8. Save processed data
    save_processed_data(X_train_processed, X_test_processed, y_train, y_test)

    logger.info("=== Feature Engineering Pipeline COMPLETE ===")


if __name__ == "__main__":
    run_feature_pipeline()
