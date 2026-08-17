"""
Centralised path configuration for the pipeline.
All stages import from here — nothing is hardcoded.
"""

import os
from pathlib import Path

# Project root is two levels up from src/config/
# Why this module exists: stages need one reproducible filesystem contract.
# Responsible for: resolving project, data, experiment, and artifact paths.
# Must not: create directories, read data, or execute pipeline stages.
PIPELINE_ROOT = Path(
    os.environ.get("PIPELINE_ROOT", Path(__file__).resolve().parents[2])
)
PROJECT_ROOT = PIPELINE_ROOT

# Data directories
RAW_DATA_DIR       = PROJECT_ROOT / "data" / "raw"
VALIDATED_DATA_DIR = PROJECT_ROOT / "data" / "validated"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Artifact directories
VALIDATION_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "validation"
FEATURES_ARTIFACTS_DIR   = PROJECT_ROOT / "artifacts" / "features"
TRAINING_ARTIFACTS_DIR   = PROJECT_ROOT / "artifacts" / "training"

# Specific files
RAW_DATASET_PATH         = RAW_DATA_DIR / "fraud_raw.csv"
VALIDATED_DATASET_PATH   = VALIDATED_DATA_DIR / "fraud_validated.csv"
PREPROCESSOR_PATH        = FEATURES_ARTIFACTS_DIR / "preprocessor.joblib"
FEATURE_METADATA_PATH    = FEATURES_ARTIFACTS_DIR / "feature_metadata.json"

TRAIN_FEATURES_PATH      = PROCESSED_DATA_DIR / "X_train.csv"
TEST_FEATURES_PATH       = PROCESSED_DATA_DIR / "X_test.csv"
TRAIN_TARGET_PATH        = PROCESSED_DATA_DIR / "y_train.csv"
TEST_TARGET_PATH         = PROCESSED_DATA_DIR / "y_test.csv"

MODEL_PATH               = TRAINING_ARTIFACTS_DIR / "model.joblib"
EVALUATION_REPORT_PATH   = TRAINING_ARTIFACTS_DIR / "evaluation_report.json"
MANIFEST_PATH            = TRAINING_ARTIFACTS_DIR / "manifest.json"
MLFLOW_TRACKING_DIR      = PIPELINE_ROOT / ".mlflow"
