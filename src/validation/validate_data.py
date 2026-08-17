import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from src.config.paths import (
    RAW_DATASET_PATH,
    VALIDATED_DATASET_PATH,
    VALIDATION_ARTIFACTS_DIR,
)
from src.validation.imputation import impute_velocity_score
from src.validation.rules import VALIDATION_RULES


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SCHEMA_RULE_NAMES = [
    "required_columns",
    "datatypes",
    "no_unexpected_columns",
]


@dataclass(frozen=True)
class ValidationResult:
    
    """Single validation check result that can be stored in reports."""

    rule_name: str
    passed: bool
    errors: list[str]
    category: str = "core"


@dataclass(frozen=True)
class ValidationReport:
    """Audit-friendly summary of one validation run."""

    timestamp_utc: str
    source_path: str
    row_count: int
    column_count: int
    passed: bool
    results: list[ValidationResult]


def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load the raw dataset into a DataFrame.

    The validation stage is the first trusted boundary. If the raw input is
    missing, fail before downstream stages can accidentally use stale outputs.
    """
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {path}")

    df = pd.read_csv(path)
    logger.info("Loaded raw dataset: %d rows, %d columns", *df.shape)
    return df


def normalize_errors(errors: str | list[str]) -> list[str]:
    """Convert rule error output into a consistent list of messages."""
    if not errors:
        return []

    if isinstance(errors, list):
        return [str(error) for error in errors if error]

    return [str(errors)]


def run_rule(
    rule_name: str,
    rule: Callable[[pd.DataFrame], tuple[bool, str | list[str]]],
    df: pd.DataFrame,
) -> ValidationResult:
    """
    Execute one validation rule and capture failures as reportable evidence.

    Validation should not crash without a report when a rule encounters an
    unexpected shape. A crash is converted into a failed rule result so the
    operator can see what happened.
    """
    try:
        passed, errors = rule(df)
        return ValidationResult(
            rule_name=rule_name,
            passed=bool(passed),
            errors=normalize_errors(errors),
        )
    except Exception as exc:
        return ValidationResult(
            rule_name=rule_name,
            passed=False,
            errors=[f"Rule raised {type(exc).__name__}: {exc}"],
        )


def run_core_validation(df: pd.DataFrame) -> list[ValidationResult]:
    """
    Run reusable validation rules from the shared registry.

    Schema rules run first because all later checks assume the expected columns
    and logical dtypes exist. If schema fails, stop there to avoid noisy or
    misleading downstream rule failures.
    """
    results: list[ValidationResult] = []

    for rule_name in SCHEMA_RULE_NAMES:
        results.append(run_rule(rule_name, VALIDATION_RULES[rule_name], df))

    if any(not result.passed for result in results):
        return results

    for rule_name, rule in VALIDATION_RULES.items():
        if rule_name in SCHEMA_RULE_NAMES:
            continue
        results.append(run_rule(rule_name, rule, df))

    return results


def run_great_expectations(df: pd.DataFrame) -> ValidationResult:
    """
    Run a compact Great Expectations suite for governance evidence.

    Great Expectations is a project standard, so missing or incompatible GX
    should fail validation instead of being silently skipped.
    """
    try:
        import great_expectations as gx
    except ImportError as exc:
        return ValidationResult(
            rule_name="great_expectations_suite",
            passed=False,
            errors=[f"Great Expectations is not installed: {exc}"],
            category="great_expectations",
        )

    if not hasattr(gx, "from_pandas"):
        return ValidationResult(
            rule_name="great_expectations_suite",
            passed=False,
            errors=[
                "Installed Great Expectations version does not expose "
                "gx.from_pandas. Migrate this stage to a Data Context, "
                "Expectation Suite, and Checkpoint before treating GX as passed."
            ],
            category="great_expectations",
        )

    gx_df = gx.from_pandas(df)
    expectations = [
        ("is_fraud_exists", lambda: gx_df.expect_column_to_exist("is_fraud")),
        (
            "is_fraud_not_null",
            lambda: gx_df.expect_column_values_to_not_be_null("is_fraud"),
        ),
        (
            "customer_age_range",
            lambda: gx_df.expect_column_values_to_be_between(
                "customer_age",
                min_value=0,
                max_value=120,
                mostly=1.0,
            ),
        ),
        (
            "network_quality_range",
            lambda: gx_df.expect_column_values_to_be_between(
                "network_quality",
                min_value=0,
                max_value=100,
                mostly=1.0,
            ),
        ),
        (
            "device_type_allowed_values",
            lambda: gx_df.expect_column_values_to_be_in_set(
                "device_type",
                [0.0, 1.0, 2.0],
                mostly=1.0,
            ),
        ),
        (
            "store_type_allowed_values",
            lambda: gx_df.expect_column_values_to_be_in_set(
                "store_type",
                [0.0, 1.0],
                mostly=1.0,
            ),
        ),
        (
            "is_fraud_binary",
            lambda: gx_df.expect_column_values_to_be_in_set("is_fraud", [0, 1]),
        ),
    ]

    errors = []
    for expectation_name, expectation in expectations:
        result = expectation()
        if not getattr(result, "success", False):
            errors.append(f"Great Expectations check failed: {expectation_name}")

    return ValidationResult(
        rule_name="great_expectations_suite",
        passed=not errors,
        errors=errors,
        category="great_expectations",
    )


def build_validation_report(
    df: pd.DataFrame,
    source_path: Path,
    results: list[ValidationResult],
) -> ValidationReport:
    """Create the persisted validation report payload."""
    passed = all(result.passed for result in results)
    return ValidationReport(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        source_path=str(source_path),
        row_count=len(df),
        column_count=len(df.columns),
        passed=passed,
        results=results,
    )


def save_validation_report(
    report: ValidationReport,
    output_dir: Path = VALIDATION_ARTIFACTS_DIR,
) -> Path:
    """Persist validation evidence for governance and debugging."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"validation_report_{timestamp}.json"

    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(asdict(report), report_file, indent=2)

    logger.info("Validation report saved: %s", report_path)
    return report_path


def save_validated_dataset(
    df: pd.DataFrame,
    output_path: Path = VALIDATED_DATASET_PATH,
) -> None:
    """
    Save the trusted dataset for downstream pipeline stages.

    This only runs after every validation check passes, so the output path
    becomes the contract boundary consumed by feature engineering and training.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Validated dataset saved: %s", output_path)


def run_validation(
    input_path: Path = RAW_DATASET_PATH,
    validated_output_path: Path = VALIDATED_DATASET_PATH,
    report_output_dir: Path = VALIDATION_ARTIFACTS_DIR,
    include_great_expectations: bool = True,
) -> ValidationReport:
    """
    Execute the full validation stage.

    Reports are always saved. Validated data is only written when all checks
    pass, which prevents stale or invalid data from becoming trusted input.
    velocity_score is imputed (median fill + a velocity_score_was_missing
    indicator) only after every check -- including its own missing-rate
    ceiling guard -- has passed. Other missing columns are saved as-is; no
    MCAR/MNAR analysis has been done for them yet.
    """
    logger.info("=== Data Validation START ===")
    df = load_dataset(input_path)

    results = run_core_validation(df)
    if include_great_expectations:
        results.append(run_great_expectations(df))

    report = build_validation_report(df, input_path, results)
    save_validation_report(report, report_output_dir)

    if not report.passed:
        logger.error("Validation failed. Validated dataset will not be written.")
        for result in report.results:
            if result.passed:
                continue
            for error in result.errors:
                logger.error("%s: %s", result.rule_name, error)
        return report

    df = impute_velocity_score(df)
    save_validated_dataset(df, validated_output_path)
    logger.info("=== Data Validation COMPLETE ===")
    return report


def main() -> None:
    """CLI entrypoint used by Makefile and CI/CD."""
    try:
        report = run_validation()
    except Exception as exc:
        logger.exception("Validation crashed before completion: %s", exc)
        raise SystemExit(1) from exc

    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
