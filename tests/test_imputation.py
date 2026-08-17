import numpy as np
import pandas as pd
import pytest

from src.validation.imputation import (
    CUSTOMER_AGE_SPEC,
    DISTANCE_FROM_HOME_SPEC,
    VELOCITY_SCORE_SPEC,
    check_missing_rate_threshold,
    impute_customer_age,
    impute_distance_from_home,
    impute_velocity_score,
    run_imputation,
)


def make_single_column_df(
    column: str, missing_rate: float = 0.15, row_count: int = 200
) -> pd.DataFrame:
    values = pd.Series(np.linspace(1.0, 10.0, row_count))
    n_missing = int(row_count * missing_rate)
    df = pd.DataFrame({column: values})
    df.loc[: n_missing - 1, column] = np.nan
    return df


def make_full_df(
    velocity_missing_rate: float = 0.15,
    customer_age_missing_rate: float = 0.12,
    distance_from_home_missing_rate: float = 0.10,
    row_count: int = 200,
) -> pd.DataFrame:
    """A df carrying all three columns run_imputation's registry expects."""
    df = pd.DataFrame(
        {
            "velocity_score": np.linspace(1.0, 10.0, row_count),
            "customer_age": np.linspace(18.0, 80.0, row_count),
            "distance_from_home": np.linspace(0.0, 100.0, row_count),
        }
    )
    df.loc[: int(row_count * velocity_missing_rate) - 1, "velocity_score"] = np.nan
    df.loc[: int(row_count * customer_age_missing_rate) - 1, "customer_age"] = np.nan
    df.loc[
        : int(row_count * distance_from_home_missing_rate) - 1, "distance_from_home"
    ] = np.nan
    return df


# ---------------------------------------------------------------------------
# Per-column imputation, parametrized across the three approved policies
# ---------------------------------------------------------------------------

IMPUTE_CASES = [
    ("velocity_score", "velocity_score_was_missing", impute_velocity_score),
    ("customer_age", "customer_age_was_missing", impute_customer_age),
    ("distance_from_home", "distance_from_home_was_missing", impute_distance_from_home),
]


@pytest.mark.parametrize("column,indicator,impute_fn", IMPUTE_CASES)
def test_impute_adds_indicator_column(column, indicator, impute_fn):
    df = make_single_column_df(column)

    result = impute_fn(df)

    assert indicator in result.columns
    assert result[indicator].sum() == df[column].isna().sum()


@pytest.mark.parametrize("column,indicator,impute_fn", IMPUTE_CASES)
def test_impute_fills_with_median_and_keeps_present_values(column, indicator, impute_fn):
    df = make_single_column_df(column)
    median_value = df[column].median()

    result = impute_fn(df)

    assert not result[column].isna().any()
    imputed_mask = result[indicator] == 1
    assert (result.loc[imputed_mask, column] == median_value).all()
    present_mask = ~imputed_mask
    pd.testing.assert_series_equal(
        result.loc[present_mask, column],
        df.loc[present_mask, column],
    )


# ---------------------------------------------------------------------------
# Missing-rate ceiling guard, parametrized across the three specs
# ---------------------------------------------------------------------------

SPEC_CASES = [
    ("velocity_score", VELOCITY_SCORE_SPEC),
    ("customer_age", CUSTOMER_AGE_SPEC),
    ("distance_from_home", DISTANCE_FROM_HOME_SPEC),
]


@pytest.mark.parametrize("column,spec", SPEC_CASES)
def test_check_missing_rate_threshold_passes_at_validated_rate(column, spec):
    df = make_single_column_df(column, missing_rate=spec.validated_missing_rate)

    passed, error = check_missing_rate_threshold(df, spec)

    assert passed
    assert error == ""


@pytest.mark.parametrize("column,spec", SPEC_CASES)
def test_check_missing_rate_threshold_fails_when_missingness_drifts_above_ceiling(
    column, spec
):
    df = make_single_column_df(column, missing_rate=spec.max_missing_rate + 0.15)

    passed, error = check_missing_rate_threshold(df, spec)

    assert not passed
    assert "exceeds the validated ceiling" in error


def test_check_missing_rate_threshold_fails_when_column_missing():
    df = pd.DataFrame({"other_column": [1, 2, 3]})

    passed, error = check_missing_rate_threshold(df, VELOCITY_SCORE_SPEC)

    assert not passed
    assert "not found" in error


# ---------------------------------------------------------------------------
# run_imputation: applies every registered policy together
# ---------------------------------------------------------------------------

def test_run_imputation_applies_every_registered_policy():
    df = make_full_df()

    result = run_imputation(df)

    for column, indicator, _ in IMPUTE_CASES:
        assert not result[column].isna().any()
        assert indicator in result.columns


def test_run_imputation_raises_when_any_column_exceeds_its_ceiling():
    df = make_full_df(customer_age_missing_rate=0.40)

    with pytest.raises(ValueError, match="exceeds the validated ceiling"):
        run_imputation(df)
