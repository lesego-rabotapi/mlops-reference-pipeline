"""
Predictive-signal assessment: is this dataset even learnable?

Schema/type/business-rule/missingness validation (src/validation/) answers
"is this data structurally sound?" That's a necessary but separate question
from "does this data contain a relationship a supervised model could learn?"
A dataset can pass every validation check -- complete, correctly typed, no
duplicates, in-range values -- and still be unsuitable for training, if the
label is statistically independent of every feature. This script is that
second, separate check. It intentionally does NOT belong in
src/validation/rules.py or the Great Expectations suite -- that boundary is
deliberate; see docs/DATASET_ASSESSMENT.md.

Four independent checks, each catching a different kind of signal a naive
check would miss:
1. Linear correlation (Pearson) -- catches monotonic linear relationships.
2. Mutual information -- catches nonlinear/non-monotonic relationships
   correlation is blind to (e.g. U-shaped).
3. Decile / group fraud-rate spread per feature -- catches a feature with a
   real "high-risk zone" even if the overall correlation washes out.
4. Row-order gap analysis on the label -- catches generation artifacts
   (clustering, periodicity, drift) that would masquerade as "no feature
   signal" when the real issue is how the file was assembled.

Usage:
    python scripts/assess_dataset_signal.py
    python scripts/assess_dataset_signal.py --output artifacts/data_quality/signal_analysis.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.paths import RAW_DATASET_PATH  # noqa: E402

TARGET_COLUMN = "is_fraud"

# Signal is judged "present" for a feature if any check clears these
# thresholds. These are read-the-numbers-yourself heuristics, not a formal
# significance test on their own -- see docs/DATASET_ASSESSMENT.md for how
# the checks are combined into an overall verdict.
CORRELATION_FLAG_THRESHOLD = 0.10
MUTUAL_INFO_FLAG_THRESHOLD = 0.01
DECILE_SPREAD_FLAG_THRESHOLD = 0.08  # 8 percentage points


def correlation_check(df: pd.DataFrame, numeric_cols: list[str]) -> dict:
    corr = df[numeric_cols + [TARGET_COLUMN]].corr()[TARGET_COLUMN].drop(TARGET_COLUMN)
    return {
        col: {
            "correlation": round(float(value), 4),
            "flagged": bool(abs(value) >= CORRELATION_FLAG_THRESHOLD),
        }
        for col, value in corr.items()
    }


def mutual_information_check(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    X = SimpleImputer(strategy="median").fit_transform(df[feature_cols])
    y = df[TARGET_COLUMN].values
    mi = mutual_info_classif(X, y, random_state=42)
    return {
        col: {"mutual_info": round(float(value), 4), "flagged": bool(value >= MUTUAL_INFO_FLAG_THRESHOLD)}
        for col, value in zip(feature_cols, mi)
    }


def decile_spread_check(df: pd.DataFrame, numeric_cols: list[str]) -> dict:
    results = {}
    for col in numeric_cols:
        subset = df[[col, TARGET_COLUMN]].dropna()
        try:
            bins = pd.qcut(subset[col], 10, duplicates="drop")
        except ValueError:
            continue
        rates = subset.groupby(bins, observed=True)[TARGET_COLUMN].mean()
        spread = float(rates.max() - rates.min())
        results[col] = {
            "min_rate": round(float(rates.min()), 4),
            "max_rate": round(float(rates.max()), 4),
            "spread": round(spread, 4),
            "flagged": bool(spread >= DECILE_SPREAD_FLAG_THRESHOLD),
        }
    return results


def categorical_group_check(df: pd.DataFrame, categorical_cols: list[str]) -> dict:
    results = {}
    for col in categorical_cols:
        rates = df.groupby(col, observed=True)[TARGET_COLUMN].mean()
        spread = float(rates.max() - rates.min())
        results[col] = {
            "rates_by_group": {str(k): round(float(v), 4) for k, v in rates.items()},
            "spread": round(spread, 4),
            "flagged": bool(spread >= DECILE_SPREAD_FLAG_THRESHOLD),
        }
    return results


def row_order_check(df: pd.DataFrame) -> dict:
    fraud_idx = df.index[df[TARGET_COLUMN] == 1].to_numpy()
    gaps = pd.Series(fraud_idx).diff().dropna()
    overall_rate = df[TARGET_COLUMN].mean()
    expected_mean_gap = 1 / overall_rate if overall_rate > 0 else float("inf")
    n = len(df)
    half = n // 2
    return {
        "overall_fraud_rate": round(float(overall_rate), 4),
        "fraud_rate_first_half": round(float(df[TARGET_COLUMN][:half].mean()), 4),
        "fraud_rate_second_half": round(float(df[TARGET_COLUMN][half:].mean()), 4),
        "gap_mean": round(float(gaps.mean()), 2),
        "gap_std": round(float(gaps.std()), 2),
        "expected_gap_mean_if_iid_bernoulli": round(expected_mean_gap, 2),
        "consistent_with_iid_bernoulli": bool(abs(gaps.mean() - expected_mean_gap) < gaps.std()),
    }


def run_assessment(df: pd.DataFrame) -> dict:
    numeric_cols = [c for c in df.columns if c != TARGET_COLUMN and df[c].nunique() > 10]
    categorical_cols = [
        c for c in df.columns if c != TARGET_COLUMN and c not in numeric_cols
    ]
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]

    correlation = correlation_check(df, numeric_cols)
    mutual_info = mutual_information_check(df, feature_cols)
    decile_spread = decile_spread_check(df, numeric_cols)
    categorical_spread = categorical_group_check(df, categorical_cols)
    row_order = row_order_check(df)

    any_flagged = (
        any(v["flagged"] for v in correlation.values())
        or any(v["flagged"] for v in mutual_info.values())
        or any(v["flagged"] for v in decile_spread.values())
        or any(v["flagged"] for v in categorical_spread.values())
    )

    return {
        "row_count": len(df),
        "correlation": correlation,
        "mutual_information": mutual_info,
        "decile_fraud_rate_spread": decile_spread,
        "categorical_group_fraud_rate_spread": categorical_spread,
        "row_order_analysis": row_order,
        "verdict": "SIGNAL_DETECTED" if any_flagged else "NO_SIGNAL_DETECTED",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/data_quality/signal_analysis.json"),
        help="Where to write the JSON assessment (default: artifacts/data_quality/signal_analysis.json)",
    )
    args = parser.parse_args()

    df = pd.read_csv(RAW_DATASET_PATH)
    assessment = run_assessment(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(assessment, f, indent=2)

    print(f"Verdict: {assessment['verdict']}")
    print(f"Written to: {args.output}")
    if assessment["verdict"] == "NO_SIGNAL_DETECTED":
        print(
            "\nNo feature cleared the correlation, mutual-information, or "
            "group-spread thresholds. See docs/DATASET_ASSESSMENT.md before "
            "training a model against this dataset."
        )


if __name__ == "__main__":
    main()
