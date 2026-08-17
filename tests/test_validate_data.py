import json

import numpy as np
import pandas as pd

from src.validation.imputation import IMPUTATION_SPECS
from src.validation.validate_data import (
    run_core_validation,
    run_validation,
)


def make_valid_raw_df(
    row_count: int = 120, missing_rate_overrides: dict | None = None
) -> pd.DataFrame:
    """
    Create a raw fraud dataset that matches the validation contract.

    row_count=120 divides evenly into the small cycles below (2s and 3s) so
    every column ends up fully populated with in-range values before
    missingness is seeded. Every column with an approved imputation policy
    (IMPUTATION_SPECS) gets nulls seeded at its own validated_missing_rate
    by default -- pass missing_rate_overrides={"column": rate} to push a
    specific column's rate for ceiling-guard tests.
    """
    overrides = missing_rate_overrides or {}
    df = pd.DataFrame(
        {
            "transaction_amount": np.linspace(10.0, 200.0, row_count),
            "hour_of_day": ([1.0, 2.0, 3.0] * (row_count // 3))[:row_count],
            "is_weekend": ([0.0, 1.0] * (row_count // 2))[:row_count],
            "num_items": [float(i % 10) for i in range(row_count)],
            "customer_age": np.linspace(18.0, 80.0, row_count),
            "prev_transactions": [float(i % 20) for i in range(row_count)],
            "distance_from_home": np.linspace(0.0, 100.0, row_count),
            "device_type": ([0.0, 1.0, 2.0] * (row_count // 3))[:row_count],
            "network_quality": np.linspace(1.0, 99.0, row_count),
            "is_first_transaction": ([0.0, 1.0] * (row_count // 2))[:row_count],
            "store_type": ([0.0, 1.0] * (row_count // 2))[:row_count],
            "velocity_score": np.linspace(1.0, 10.0, row_count),
            "is_fraud": ([0, 1] * (row_count // 2))[:row_count],
        }
    )

    for column, spec in IMPUTATION_SPECS.items():
        rate = overrides.get(column, spec.validated_missing_rate)
        n_missing = int(row_count * rate)
        if n_missing > 0:
            df.loc[: n_missing - 1, column] = np.nan

    return df


def test_core_validation_passes_for_valid_raw_dataset():
    df = make_valid_raw_df()

    results = run_core_validation(df)

    assert all(result.passed for result in results)


def test_core_validation_fails_cleanly_when_required_column_is_missing():
    df = make_valid_raw_df().drop(columns=["network_quality"])

    results = run_core_validation(df)

    failed_rules = {result.rule_name for result in results if not result.passed}
    assert "required_columns" in failed_rules


def test_core_validation_fails_on_duplicate_rows():
    df = make_valid_raw_df()
    df.loc[1] = df.loc[0]

    results = run_core_validation(df)

    duplicate_result = next(
        result for result in results if result.rule_name == "no_duplicate_rows"
    )
    assert not duplicate_result.passed
    assert "duplicate row" in duplicate_result.errors[0]


def test_core_validation_fails_when_a_column_missing_rate_exceeds_its_ceiling():
    for column, spec in IMPUTATION_SPECS.items():
        df = make_valid_raw_df(
            missing_rate_overrides={column: min(spec.max_missing_rate + 0.20, 0.95)}
        )

        results = run_core_validation(df)

        ceiling_result = next(
            result
            for result in results
            if result.rule_name == f"{column}_missing_rate_within_ceiling"
        )
        assert not ceiling_result.passed, f"{column} ceiling guard should have failed"
        assert "exceeds the validated ceiling" in ceiling_result.errors[0]


def test_run_validation_writes_report_and_validated_dataset_on_success(tmp_path):
    input_path = tmp_path / "raw.csv"
    output_path = tmp_path / "validated" / "validated.csv"
    report_dir = tmp_path / "reports"
    make_valid_raw_df().to_csv(input_path, index=False)

    report = run_validation(
        input_path=input_path,
        validated_output_path=output_path,
        report_output_dir=report_dir,
        include_great_expectations=False,
    )

    report_files = list(report_dir.glob("validation_report_*.json"))
    assert report.passed
    assert output_path.exists()
    assert len(report_files) == 1

    with report_files[0].open(encoding="utf-8") as report_file:
        report_payload = json.load(report_file)

    assert report_payload["passed"] is True
    assert report_payload["row_count"] == 120


def test_run_validation_imputes_every_approved_column(tmp_path):
    """
    Every column in IMPUTATION_SPECS should come out fully populated with
    its indicator column present -- proof the full missingness-handling
    scope (all 12 flagged columns) runs end to end through the actual
    validation pipeline, not just the standalone imputation functions.
    """
    input_path = tmp_path / "raw.csv"
    output_path = tmp_path / "validated" / "validated.csv"
    report_dir = tmp_path / "reports"

    make_valid_raw_df().to_csv(input_path, index=False)

    report = run_validation(
        input_path=input_path,
        validated_output_path=output_path,
        report_output_dir=report_dir,
        include_great_expectations=False,
    )

    assert report.passed
    validated_df = pd.read_csv(output_path)

    for column, spec in IMPUTATION_SPECS.items():
        assert spec.indicator_column in validated_df.columns, column
        assert not validated_df[column].isna().any(), column

    assert not validated_df["is_fraud"].isna().any()


def test_run_validation_does_not_write_validated_dataset_on_failure(tmp_path):
    input_path = tmp_path / "raw.csv"
    output_path = tmp_path / "validated" / "validated.csv"
    report_dir = tmp_path / "reports"

    df = make_valid_raw_df()
    df.loc[0, "customer_age"] = 200.0
    df.to_csv(input_path, index=False)

    report = run_validation(
        input_path=input_path,
        validated_output_path=output_path,
        report_output_dir=report_dir,
        include_great_expectations=False,
    )

    assert not report.passed
    assert not output_path.exists()
    assert len(list(report_dir.glob("validation_report_*.json"))) == 1
