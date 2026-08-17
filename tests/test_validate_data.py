import json

import numpy as np
import pandas as pd

from src.validation.validate_data import (
    run_core_validation,
    run_validation,
)


def make_valid_raw_df(row_count: int = 120, velocity_missing_rate: float = 0.15) -> pd.DataFrame:
    """
    Create a raw fraud dataset that matches the validation contract.

    row_count=120 divides evenly into the small cycles below (2s and 3s) so
    every column ends up fully populated with in-range values; only
    velocity_score is seeded with nulls, at a caller-controlled rate, since
    that's the one column with an approved missing-rate ceiling to test.
    """
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

    n_missing = int(row_count * velocity_missing_rate)
    df.loc[: n_missing - 1, "velocity_score"] = np.nan

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


def test_core_validation_fails_when_velocity_score_missing_rate_exceeds_ceiling():
    df = make_valid_raw_df(velocity_missing_rate=0.45)

    results = run_core_validation(df)

    ceiling_result = next(
        result
        for result in results
        if result.rule_name == "velocity_score_missing_rate_within_ceiling"
    )
    assert not ceiling_result.passed
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


def test_run_validation_imputes_velocity_score_but_leaves_other_columns_null(tmp_path):
    """
    velocity_score should come out fully populated with an indicator column;
    a column with no approved imputation policy (customer_age here) should
    be saved with its nulls intact -- proof nothing beyond the approved
    scope was silently imputed.
    """
    input_path = tmp_path / "raw.csv"
    output_path = tmp_path / "validated" / "validated.csv"
    report_dir = tmp_path / "reports"

    df = make_valid_raw_df()
    df.loc[0, "customer_age"] = np.nan
    df.to_csv(input_path, index=False)

    report = run_validation(
        input_path=input_path,
        validated_output_path=output_path,
        report_output_dir=report_dir,
        include_great_expectations=False,
    )

    assert report.passed
    validated_df = pd.read_csv(output_path)

    assert "velocity_score_was_missing" in validated_df.columns
    assert not validated_df["velocity_score"].isna().any()
    assert validated_df["customer_age"].isna().sum() == 1


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
