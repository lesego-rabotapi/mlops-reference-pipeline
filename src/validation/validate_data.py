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
from src.validation.imputation import run_imputation
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


def build_great_expectations_suite(gx):
    """
    Build the ExpectationSuite of governance-evidence checks for fraud.csv.

    This mirrors (not replaces) the core rules in src/validation/rules.py --
    GX exists as a second, industry-standard evidence trail, so the checks
    are intentionally the same facts checked a different way, not new rules.
    """
    suite = gx.ExpectationSuite(name="fraud_validation_suite")
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="is_fraud"))
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="is_fraud")
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="customer_age", min_value=0, max_value=120, mostly=1.0
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="network_quality", min_value=0, max_value=100, mostly=1.0
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="device_type", value_set=[0.0, 1.0, 2.0], mostly=1.0
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="store_type", value_set=[0.0, 1.0], mostly=1.0
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(column="is_fraud", value_set=[0, 1])
    )
    return suite


def run_great_expectations(df: pd.DataFrame) -> ValidationResult:
    """
    Run a compact Great Expectations suite for governance evidence.

    Great Expectations is a project standard, so missing or incompatible GX
    should fail validation instead of being silently skipped. Uses the GX
    1.x Data Context / ExpectationSuite / Batch API -- the pre-1.0
    gx.from_pandas() one-liner this stage used to call was removed upstream.
    A fresh ephemeral context is created per call (no state persisted to
    disk), matching this stage's stateless, per-run validation model.
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

    try:
        context = gx.get_context(mode="ephemeral")
        data_source = context.data_sources.add_pandas("pandas")
        data_asset = data_source.add_dataframe_asset(name="fraud_dataset")
        batch_definition = data_asset.add_batch_definition_whole_dataframe(
            "fraud_batch"
        )
        batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

        suite = build_great_expectations_suite(gx)
        suite_result = batch.validate(suite)
    except Exception as exc:
        return ValidationResult(
            rule_name="great_expectations_suite",
            passed=False,
            errors=[f"Great Expectations suite raised {type(exc).__name__}: {exc}"],
            category="great_expectations",
        )

    errors = [
        f"Great Expectations check failed: {result.expectation_config.type} "
        f"on '{result.expectation_config.kwargs.get('column')}'"
        for result in suite_result.results
        if not result.success
    ]

    return ValidationResult(
        rule_name="great_expectations_suite",
        passed=suite_result.success,
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
    Every column with an approved imputation policy (see IMPUTATION_SPECS in
    src/validation/imputation.py and docs/MISSINGNESS_ANALYSIS.md for the
    MCAR evidence behind each) is imputed only after every check --
    including its own missing-rate ceiling guard -- has passed. is_fraud
    (the target) is never imputed; it's required non-null by the
    target_not_null rule instead.
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

    df = run_imputation(df)
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
