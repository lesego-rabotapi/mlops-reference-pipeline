"""
Repeatable MCAR/MNAR missingness analysis for the fraud dataset.

This is the same methodology used to justify every imputation policy in
src/validation/imputation.py, packaged as a script so a new column (or a new
batch of data) can be re-checked instead of pattern-matching an old verdict
onto it. See docs/MISSINGNESS_ANALYSIS.md for how to read the output and the
full set of prior results.

For a given column, this reports:
1. Overall missing rate.
2. Fraud-rate split (missing vs present) with a two-proportion z-test --
   a low p-value is evidence the missingness is concentrated in fraud/
   non-fraud cases (MNAR-leaning), not evidence of MCAR by itself.
3. Missing-rate split against every other categorical-ish column
   (values with 2/3 the row count is a poor granularity is skipped) --
   a wide spread suggests the column's absence depends on another field
   (MNAR-leaning).
4. Missing-rate split against quartiles of every continuous column.
5. How correlated the column's missingness indicator is with every other
   column's missingness indicator -- a high correlation suggests a shared,
   possibly batch-level, missingness mechanism worth investigating further
   rather than treating as independent MCAR noise.

None of this alone *proves* MCAR -- it's evidence-gathering, not a formal
test (a real Little's MCAR test would need the full joint distribution).
Treat "no signal found" as "no evidence against MCAR", and use judgment
about the domain (e.g. would this field plausibly be uncomputable for a
specific subgroup?) alongside the numbers.

Usage:
    python scripts/analyze_missingness.py <column> [<column> ...]
    python scripts/analyze_missingness.py --all
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.paths import RAW_DATASET_PATH  # noqa: E402

TARGET_COLUMN = "is_fraud"

# Columns treated as discrete "groups" for the crosstab check -- low
# cardinality, so a groupby is informative. Continuous columns are checked
# via quartile binning instead (see CONTINUOUS_DRIVERS).
CATEGORICAL_DRIVERS = [
    "is_first_transaction",
    "device_type",
    "store_type",
    "is_weekend",
    "hour_of_day",
]

CONTINUOUS_DRIVERS = [
    "transaction_amount",
    "network_quality",
    "prev_transactions",
    "customer_age",
    "distance_from_home",
    "velocity_score",
]

ALL_COLUMNS_WITH_MISSINGNESS = [
    "velocity_score",
    "customer_age",
    "distance_from_home",
    "network_quality",
    "transaction_amount",
    "prev_transactions",
    "hour_of_day",
    "device_type",
    "num_items",
    "is_first_transaction",
    "is_weekend",
    "store_type",
]


def fraud_rate_split(df: pd.DataFrame, column: str) -> None:
    missing_mask = df[column].isna()
    n_missing, n_present = int(missing_mask.sum()), int((~missing_mask).sum())
    if n_missing == 0:
        print("  (no missing rows -- skipping fraud-rate split)")
        return

    x_missing = df.loc[missing_mask, TARGET_COLUMN].sum()
    x_present = df.loc[~missing_mask, TARGET_COLUMN].sum()
    p_missing, p_present = x_missing / n_missing, x_present / n_present

    p_pool = (x_missing + x_present) / (n_missing + n_present)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_missing + 1 / n_present))
    z = (p_missing - p_present) / se if se > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    print(
        f"  fraud rate | missing (n={n_missing})={p_missing:.1%}  "
        f"present (n={n_present})={p_present:.1%}  p={p_value:.3f}"
    )
    if p_value < 0.05:
        print("  ** statistically significant gap -- possible MNAR signal, investigate **")


def categorical_driver_splits(df: pd.DataFrame, column: str) -> None:
    for driver in CATEGORICAL_DRIVERS:
        if driver == column:
            continue
        rates = df.groupby(driver)[column].apply(lambda s: s.isna().mean())
        spread = rates.max() - rates.min()
        flag = "  ** wide spread **" if spread > 0.05 else ""
        print(f"  by {driver}: spread={spread:.1%}{flag}")


def continuous_driver_splits(df: pd.DataFrame, column: str) -> None:
    for driver in CONTINUOUS_DRIVERS:
        if driver == column:
            continue
        try:
            bins = pd.qcut(df[driver], 4, duplicates="drop")
        except ValueError:
            continue
        rates = df.groupby(bins, observed=True)[column].apply(lambda s: s.isna().mean())
        spread = rates.max() - rates.min()
        flag = "  ** wide spread **" if spread > 0.05 else ""
        print(f"  by {driver} quartile: spread={spread:.1%}{flag}")


def missingness_correlation(df: pd.DataFrame, column: str) -> None:
    other_columns = [c for c in ALL_COLUMNS_WITH_MISSINGNESS if c in df.columns and c != column]
    miss = df[[column] + other_columns].isna()
    corr = miss.corr()[column].drop(column)
    max_abs_corr = corr.abs().max() if len(corr) else 0.0
    flag = "  ** notably correlated **" if max_abs_corr > 0.1 else ""
    print(f"  max |correlation| with other columns' missingness: {max_abs_corr:.3f}{flag}")


def analyze_column(df: pd.DataFrame, column: str) -> None:
    missing_rate = df[column].isna().mean()
    print(f"===== {column} =====")
    print(f"overall missing rate: {missing_rate:.1%}")
    fraud_rate_split(df, column)
    categorical_driver_splits(df, column)
    continuous_driver_splits(df, column)
    missingness_correlation(df, column)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("columns", nargs="*", help="Column(s) to analyze")
    parser.add_argument(
        "--all", action="store_true", help="Analyze every column with missingness"
    )
    args = parser.parse_args()

    df = pd.read_csv(RAW_DATASET_PATH)

    columns = ALL_COLUMNS_WITH_MISSINGNESS if args.all else args.columns
    if not columns:
        parser.error("Pass one or more column names, or --all")

    for column in columns:
        if column not in df.columns:
            print(f"===== {column} =====\n  column not found in {RAW_DATASET_PATH}\n")
            continue
        analyze_column(df, column)


if __name__ == "__main__":
    main()
