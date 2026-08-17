import numpy as np
import pandas as pd
import pytest

from src.validation.imputation import (
    VELOCITY_SCORE_SPEC,
    check_missing_rate_threshold,
    impute_velocity_score,
    run_imputation,
)


def make_velocity_df(missing_rate: float = 0.15, row_count: int = 200) -> pd.DataFrame:
    values = pd.Series(np.linspace(1.0, 10.0, row_count))
    n_missing = int(row_count * missing_rate)
    df = pd.DataFrame({"velocity_score": values})
    df.loc[: n_missing - 1, "velocity_score"] = np.nan
    return df


def test_impute_velocity_score_adds_indicator_column():
    df = make_velocity_df()

    result = impute_velocity_score(df)

    assert "velocity_score_was_missing" in result.columns
    assert result["velocity_score_was_missing"].sum() == df["velocity_score"].isna().sum()


def test_impute_velocity_score_fills_with_median_and_keeps_present_values():
    df = make_velocity_df()
    median_value = df["velocity_score"].median()

    result = impute_velocity_score(df)

    assert not result["velocity_score"].isna().any()
    imputed_mask = result["velocity_score_was_missing"] == 1
    assert (result.loc[imputed_mask, "velocity_score"] == median_value).all()
    present_mask = ~imputed_mask
    pd.testing.assert_series_equal(
        result.loc[present_mask, "velocity_score"],
        df.loc[present_mask, "velocity_score"],
    )


def test_check_missing_rate_threshold_passes_at_validated_rate():
    df = make_velocity_df(missing_rate=0.15)

    passed, error = check_missing_rate_threshold(df, VELOCITY_SCORE_SPEC)

    assert passed
    assert error == ""


def test_check_missing_rate_threshold_fails_when_missingness_drifts_above_ceiling():
    df = make_velocity_df(missing_rate=0.45)

    passed, error = check_missing_rate_threshold(df, VELOCITY_SCORE_SPEC)

    assert not passed
    assert "exceeds the validated ceiling" in error


def test_check_missing_rate_threshold_fails_when_column_missing():
    df = pd.DataFrame({"other_column": [1, 2, 3]})

    passed, error = check_missing_rate_threshold(df, VELOCITY_SCORE_SPEC)

    assert not passed
    assert "not found" in error


def test_run_imputation_applies_registered_policy():
    df = make_velocity_df()

    result = run_imputation(df)

    assert not result["velocity_score"].isna().any()
    assert "velocity_score_was_missing" in result.columns


def test_run_imputation_raises_when_missingness_exceeds_ceiling():
    df = make_velocity_df(missing_rate=0.45)

    with pytest.raises(ValueError, match="exceeds the validated ceiling"):
        run_imputation(df)
