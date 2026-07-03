import json

import pandas as pd

from src.validation.validate_data import (
    run_core_validation,
    run_validation,
)


def make_valid_raw_df(row_count: int = 120) -> pd.DataFrame:
    """Create a raw dataset that matches the validation contract."""
    return pd.DataFrame(
        {
            "id": range(1, row_count + 1),
            "CustomerId": range(100000, 100000 + row_count),
            "Surname": [f"Customer{i}" for i in range(row_count)],
            "CreditScore": [650] * row_count,
            "Geography": ["France"] * row_count,
            "Gender": ["Female"] * row_count,
            "Age": [42.0] * row_count,
            "Tenure": [5] * row_count,
            "Balance": [1200.0] * row_count,
            "NumOfProducts": [2] * row_count,
            "HasCrCard": [1.0] * row_count,
            "IsActiveMember": [1.0] * row_count,
            "EstimatedSalary": [85000.0] * row_count,
            "Exited": [0] * row_count,
        }
    )


def test_core_validation_passes_for_valid_raw_dataset():
    df = make_valid_raw_df()

    results = run_core_validation(df)

    assert all(result.passed for result in results)


def test_core_validation_fails_cleanly_when_required_column_is_missing():
    df = make_valid_raw_df().drop(columns=["CustomerId"])

    results = run_core_validation(df)

    failed_rules = {result.rule_name for result in results if not result.passed}
    assert "required_columns" in failed_rules


def test_core_validation_fails_on_duplicate_record_ids():
    df = make_valid_raw_df()
    df.loc[1, "id"] = df.loc[0, "id"]

    results = run_core_validation(df)

    duplicate_result = next(
        result for result in results if result.rule_name == "no_duplicate_ids"
    )
    assert not duplicate_result.passed
    assert "duplicate id" in duplicate_result.errors[0]


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


def test_run_validation_does_not_write_validated_dataset_on_failure(tmp_path):
    input_path = tmp_path / "raw.csv"
    output_path = tmp_path / "validated" / "validated.csv"
    report_dir = tmp_path / "reports"

    df = make_valid_raw_df()
    df.loc[0, "CreditScore"] = 100
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
