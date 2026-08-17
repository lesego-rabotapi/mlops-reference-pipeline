import numpy as np
import pandas as pd
import pytest

from src.validation.imputation import (
    IMPUTATION_RULES,
    IMPUTATION_SPECS,
    ImputationSpec,
    check_missing_rate_threshold,
    run_imputation,
)


def make_column_values(spec: ImputationSpec, row_count: int) -> np.ndarray:
    """
    A plausible value series for a spec's strategy: a spread of continuous
    values for "median" columns, a small repeating set of category-like
    values (so .mode() is well-defined) for "mode" columns.
    """
    if spec.strategy == "median":
        return np.linspace(1.0, 10.0, row_count)
    return np.array([float(i % 3) for i in range(row_count)])


def make_single_column_df(
    column: str, missing_rate: float, row_count: int = 200
) -> pd.DataFrame:
    spec = IMPUTATION_SPECS[column]
    values = make_column_values(spec, row_count)
    df = pd.DataFrame({column: values})
    n_missing = int(row_count * missing_rate)
    df.loc[: n_missing - 1, column] = np.nan
    return df


def make_full_df(row_count: int = 200, missing_rate_overrides: dict | None = None) -> pd.DataFrame:
    """A df carrying every column run_imputation's registry expects."""
    overrides = missing_rate_overrides or {}
    df = pd.DataFrame(
        {
            column: make_column_values(spec, row_count)
            for column, spec in IMPUTATION_SPECS.items()
        }
    )
    for column, spec in IMPUTATION_SPECS.items():
        rate = overrides.get(column, spec.validated_missing_rate)
        n_missing = int(row_count * rate)
        if n_missing > 0:
            df.loc[: n_missing - 1, column] = np.nan
    return df


SPEC_CASES = list(IMPUTATION_SPECS.items())


# ---------------------------------------------------------------------------
# Per-column imputation, parametrized across every approved policy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("column,spec", SPEC_CASES)
def test_impute_adds_indicator_column(column, spec):
    df = make_single_column_df(column, missing_rate=spec.validated_missing_rate)

    result = IMPUTATION_RULES[column](df)

    assert spec.indicator_column in result.columns
    assert result[spec.indicator_column].sum() == df[column].isna().sum()


@pytest.mark.parametrize("column,spec", SPEC_CASES)
def test_impute_fills_nulls_and_keeps_present_values(column, spec):
    df = make_single_column_df(column, missing_rate=spec.validated_missing_rate)
    fill_value = df[column].median() if spec.strategy == "median" else df[column].mode().iloc[0]

    result = IMPUTATION_RULES[column](df)

    assert not result[column].isna().any()
    imputed_mask = result[spec.indicator_column] == 1
    assert (result.loc[imputed_mask, column] == fill_value).all()
    present_mask = ~imputed_mask
    pd.testing.assert_series_equal(
        result.loc[present_mask, column],
        df.loc[present_mask, column],
    )


# ---------------------------------------------------------------------------
# Missing-rate ceiling guard, parametrized across every spec
# ---------------------------------------------------------------------------

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
    df = make_single_column_df(column, missing_rate=min(spec.max_missing_rate + 0.15, 0.95))

    passed, error = check_missing_rate_threshold(df, spec)

    assert not passed
    assert "exceeds the validated ceiling" in error


def test_check_missing_rate_threshold_fails_when_column_missing():
    velocity_spec = IMPUTATION_SPECS["velocity_score"]
    df = pd.DataFrame({"other_column": [1, 2, 3]})

    passed, error = check_missing_rate_threshold(df, velocity_spec)

    assert not passed
    assert "not found" in error


# ---------------------------------------------------------------------------
# run_imputation: applies every registered policy together
# ---------------------------------------------------------------------------

def test_run_imputation_applies_every_registered_policy():
    df = make_full_df()

    result = run_imputation(df)

    for column, spec in IMPUTATION_SPECS.items():
        assert not result[column].isna().any()
        assert spec.indicator_column in result.columns


def test_run_imputation_raises_when_any_column_exceeds_its_ceiling():
    df = make_full_df(missing_rate_overrides={"customer_age": 0.90})

    with pytest.raises(ValueError, match="exceeds the validated ceiling"):
        run_imputation(df)
