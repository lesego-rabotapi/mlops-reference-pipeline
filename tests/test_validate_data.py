import json

import numpy as np
import pandas as pd

from src.validation.validate_data import (
    run_core_validation,
    run_great_expectations,
    run_validation,
)


def failed_rules(df: pd.DataFrame) -> dict[str, list[str]]:
    return {r.rule_name: r.errors for r in run_core_validation(df) if not r.passed}


def test_core_validation_passes_for_valid_dataset(paysim_df):
    assert failed_rules(paysim_df) == {}


def test_missing_required_column_fails_schema_and_stops(paysim_df):
    results = run_core_validation(paysim_df.drop(columns=["amount"]))
    failed = [r for r in results if not r.passed]
    assert [r.rule_name for r in failed] == ["required_columns"]
    assert "amount" in failed[0].errors[0]
    # schema failure stops the run: no downstream rule results
    assert len(results) == 3


def test_unexpected_column_fails(paysim_df):
    paysim_df["surprise"] = 1
    assert "no_unexpected_columns" in failed_rules(paysim_df)


def test_wrong_dtype_fails(paysim_df):
    paysim_df["isFraud"] = paysim_df["isFraud"].astype(float) + 0.5
    assert "datatypes" in failed_rules(paysim_df)


def test_any_null_fails_instead_of_being_imputed(paysim_df):
    paysim_df.loc[3, "amount"] = np.nan
    assert "no_missing_values" in failed_rules(paysim_df)


def test_negative_amount_and_balance_fail(paysim_df):
    paysim_df.loc[0, "amount"] = -1.0
    paysim_df.loc[1, "oldbalanceOrg"] = -5.0
    failed = failed_rules(paysim_df)
    assert "amount_non_negative" in failed
    assert "oldbalanceOrg_non_negative" in failed


def test_invalid_type_fails(paysim_df):
    paysim_df.loc[0, "type"] = "REFUND"
    assert "type_valid" in failed_rules(paysim_df)


def test_invalid_target_value_fails(paysim_df):
    paysim_df.loc[0, "isFraud"] = 2
    assert "isFraud_binary" in failed_rules(paysim_df)


def test_single_class_target_fails(paysim_df):
    paysim_df["isFraud"] = 0
    assert "both_classes_present" in failed_rules(paysim_df)


def test_step_below_one_fails(paysim_df):
    paysim_df.loc[0, "step"] = 0
    assert "step_positive" in failed_rules(paysim_df)


def test_duplicate_rows_fail(paysim_df):
    dup = pd.concat([paysim_df, paysim_df.iloc[[0]]], ignore_index=True)
    assert "no_duplicate_rows" in failed_rules(dup)


def test_tiny_dataset_fails_minimum_size(paysim_df):
    assert "minimum_size" in failed_rules(paysim_df.head(10))


def test_great_expectations_passes_for_valid_dataset(paysim_df):
    result = run_great_expectations(paysim_df)
    assert result.passed, result.errors


def test_great_expectations_catches_invalid_type(paysim_df):
    paysim_df.loc[0, "type"] = "REFUND"
    assert not run_great_expectations(paysim_df).passed


def test_run_validation_writes_validated_data_only_on_success(paysim_df, tmp_path):
    raw = tmp_path / "raw.csv"
    out = tmp_path / "validated" / "v.csv"
    reports = tmp_path / "reports"
    paysim_df.to_csv(raw, index=False)

    report = run_validation(raw, out, reports)
    assert report.passed and out.exists()
    # no imputation: validated file is identical in content to the input
    pd.testing.assert_frame_equal(pd.read_csv(out), paysim_df)
    assert len(list(reports.glob("validation_report_*.json"))) == 1


def test_run_validation_halts_and_writes_report_on_failure(paysim_df, tmp_path):
    paysim_df.loc[0, "type"] = "REFUND"
    raw = tmp_path / "raw.csv"
    out = tmp_path / "validated" / "v.csv"
    reports = tmp_path / "reports"
    paysim_df.to_csv(raw, index=False)

    report = run_validation(raw, out, reports)
    assert not report.passed
    assert not out.exists()
    saved = json.loads(next(reports.glob("validation_report_*.json")).read_text())
    assert saved["passed"] is False
