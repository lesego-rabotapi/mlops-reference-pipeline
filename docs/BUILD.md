<!--
Why this file exists: documents how to build and verify the local-first MLOps pipeline in a way a reviewer can reproduce.
Responsible for: explaining build order, engineering rationale, produced evidence, and production-minded failure prevention.
Must not: introduce cloud-only implementation steps, replace architecture docs, or pretend unfinished serving/observability work is complete.
-->

# Build Documentation

## Purpose

This document explains how to build the project locally and how to think like an
engineer while doing it. The goal is not only to run commands. The goal is to
understand what each stage proves, what failure it prevents, and what evidence a
reviewer can inspect afterward.

This project is a local-first, cloud-portable MLOps reference pipeline. It is
designed for portfolio work and real-world problem solving under a realistic
constraint: no live AWS account is required. The engineering standard is still
production-minded: explicit contracts, reproducible commands, persisted
artifacts, tests, and clear promotion boundaries.

## Build Philosophy

The pipeline is built as a sequence of trusted boundaries:

```text
raw data
  -> validation report + validated data
  -> processed features + fitted preprocessor
  -> trained model + metrics + artifact manifest + MLflow run
  -> future serving, container, CI, and observability layers
```

Each stage owns one responsibility. Validation should not train models. Feature
engineering should not serve predictions. Training should not repair bad raw
data. This separation is what makes the project testable, debuggable, and
portable to a future cloud environment.

## Current Build State

Implemented locally:

- Data validation through `src/validation/validate_data.py`
- Feature engineering through `src/features/build_features.py`
- Training and evaluation through `src/training/train_model.py`
- Local artifact contracts under `artifacts/`
- Processed data contracts under `data/processed/`
- MLflow tracking under `.mlflow/`
- Stage-specific and end-to-end test coverage through `pytest`
- Makefile commands for repeatable local execution

Planned next stages:

- FastAPI inference service
- Docker image for the inference service
- docker-compose stack for FastAPI, Prometheus, and Grafana
- GitHub Actions workflow with install, tests, Docker build, and Trivy scan
- Governance and monitoring documentation

This distinction matters for portfolio credibility. A reviewer should be able to
tell which work is finished, which work is intentionally scoped, and which work
is next.

## Prerequisites

Required for the implemented pipeline stages:

- Python 3.12
- Git
- Project dependencies from `requirements.txt`

Required for later serving and observability stages:

- Docker Desktop
- Docker Compose

The Python pipeline should run without AWS credentials, cloud configuration, or
machine-specific paths.

## Environment Build

From the repository root, create the local Python environment:

```powershell
.\scripts\bootstrap_env.ps1 -Python python
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the project package in editable mode when working across modules:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

Verify the environment:

```powershell
.\.venv\Scripts\python.exe scripts\check_environment.py
.\.venv\Scripts\python.exe -m pytest tests
```

Production rationale: the environment is part of the build artifact story. If a
fresh clone cannot install dependencies and run tests, the project is not yet
reproducible.

## Pipeline Build Order

Run the pipeline in this order:

```powershell
make validate
make features
make train
```

Or run the composed pipeline target:

```powershell
make pipeline
```

The order is intentional. Training depends on processed feature files and the
fitted preprocessor. Feature engineering depends on validated data. Validation
depends on raw data. Running a later stage before its upstream contract exists
should fail loudly.

## Stage 1: Validation

Command:

```powershell
make validate
```

Primary code:

- `src/validation/validate_data.py`
- `src/validation/rules.py`

Inputs:

- `data/raw/customer_churn_raw.csv`

Outputs:

- `data/validated/customer_churn_validated.csv`
- `artifacts/validation/validation_report_<timestamp>.json`

What this proves:

- Required columns exist
- Unexpected columns are rejected
- Data types and value ranges match the contract
- Invalid raw data does not become trusted pipeline input

Production failure prevented:

- Training on malformed or silently corrupted raw data
- Debugging model behavior when the real issue is a broken data contract
- Accidentally promoting stale validated data after a failed validation run

Reviewer evidence:

- Validation report JSON
- Validated dataset file
- Passing validation tests

## Stage 2: Feature Engineering

Command:

```powershell
make features
```

Primary code:

- `src/features/build_features.py`
- `src/config/feature_config.py`

Inputs:

- `data/validated/customer_churn_validated.csv`

Outputs:

- `data/processed/X_train.csv`
- `data/processed/X_test.csv`
- `data/processed/y_train.csv`
- `data/processed/y_test.csv`
- `artifacts/features/preprocessor.joblib`
- `artifacts/features/feature_metadata.json`

What this proves:

- The train/test split happens before fitting transformations
- The fitted preprocessor is persisted for inference reuse
- Feature names and preprocessing metadata are inspectable
- Unknown categorical values can be handled at inference time

Production failure prevented:

- Data leakage from fitting transformations on test data
- Training-serving skew from rebuilding preprocessing logic separately
- Runtime failures when production traffic contains unseen categories

Reviewer evidence:

- Processed train/test CSVs
- Preprocessor artifact
- Feature metadata
- Leakage-focused feature tests

## Stage 3: Training And Evaluation

Command:

```powershell
make train
```

Primary code:

- `src/training/train_model.py`
- `src/config/training_config.py`

Inputs:

- `data/processed/X_train.csv`
- `data/processed/X_test.csv`
- `data/processed/y_train.csv`
- `data/processed/y_test.csv`
- `artifacts/features/preprocessor.joblib`

Outputs:

- `artifacts/training/model.joblib`
- `artifacts/training/evaluation_report.json`
- `artifacts/training/manifest.json`
- `.mlflow/` local experiment tracking data

What this proves:

- Training consumes the feature contract instead of raw data
- The model is trained with explicit configuration
- Evaluation uses multiple metrics, not accuracy alone
- The model and preprocessor are bound together by artifact hashes
- MLflow records parameters, metrics, and artifacts for run comparison

Production failure prevented:

- Silent training on missing or stale processed data
- Overstating quality with accuracy on an imbalanced classification problem
- Serving a model with the wrong preprocessor
- Losing the evidence needed to compare or roll back model versions

Reviewer evidence:

- Saved model artifact
- Evaluation report with `accuracy`, `precision`, `recall`, `f1`, and `roc_auc`
- Manifest with model hash, preprocessor hash, sklearn version, and timestamp
- MLflow run containing parameters, metrics, and artifacts
- Passing training tests

## Full Verification

Run all tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests
```

Run targeted tests:

```powershell
make test-validation
make test-features
make test-training
```

What the test suite should protect:

- Validation rules fail bad input before downstream stages run
- Feature engineering avoids train/test leakage
- Training publishes expected artifacts and metrics
- Missing upstream training inputs fail before partial artifacts are written
- Model predictions stay binary and integer-like

In production terms, tests are not decoration. They are the cheapest way to
prove that pipeline boundaries still hold after code changes.

## Artifact Governance

The important local artifacts are:

| Artifact | Owner Stage | Why It Matters |
| --- | --- | --- |
| `artifacts/validation/*.json` | Validation | Audit trail for raw data acceptance or rejection |
| `data/validated/customer_churn_validated.csv` | Validation | Trusted dataset boundary for downstream work |
| `artifacts/features/preprocessor.joblib` | Feature engineering | Exact transformation object required by training and serving |
| `artifacts/features/feature_metadata.json` | Feature engineering | Feature contract evidence for review and inference alignment |
| `artifacts/training/model.joblib` | Training | Deployable model artifact |
| `artifacts/training/evaluation_report.json` | Training | Local model quality evidence |
| `artifacts/training/manifest.json` | Training | Hash-based guard against model/preprocessor mismatch |
| `.mlflow/` | Training | Experiment history and run comparison evidence |

The production-minded habit is to ask: "What file proves this stage ran
correctly?" If there is no artifact, report, metric, or log, the stage is hard
to operate.

## Path And Portability Contract

Project paths are centralized in `src/config/paths.py`. The `PIPELINE_ROOT`
environment variable can override the repository root for tests or alternative
runtime layouts.

This prevents hidden dependencies on the current working directory. A command
should behave the same when run manually, from CI, or from a future container as
long as the pipeline root is configured correctly.

## MLflow Inspection

After training, inspect local experiment runs with:

```powershell
.\.venv\Scripts\mlflow.exe ui --backend-store-uri .\.mlflow
```

Then open:

```text
http://127.0.0.1:5000
```

Use MLflow to compare:

- Random forest parameters
- Accuracy, precision, recall, weighted F1, and ROC-AUC
- Model artifact paths
- Evaluation and governance artifacts

The production lesson is that a model run is not just code output. It is a
decision record.

## Build Failure Guide

Common failure modes:

| Symptom | Likely Cause | Correct Fix |
| --- | --- | --- |
| `make features` cannot find validated data | Validation has not run or failed | Run `make validate` and inspect the validation report |
| `make train` reports missing processed files | Feature stage has not published its contract | Run `make features` after validation succeeds |
| ROC-AUC fails | Test target contains only one class | Revisit split strategy or test fixture data |
| Imports fail from tests or Makefile | Project package is not installed or root is wrong | Run editable install and verify `PIPELINE_ROOT` |
| MLflow UI shows no runs | Wrong backend store path | Start MLflow with `--backend-store-uri .\.mlflow` from repo root |

The engineering instinct is to fix the earliest broken contract, not patch the
downstream symptom.

## Portfolio Reading Guide

When presenting this project, lead with the engineering story:

1. The pipeline is intentionally local-first because AWS access is unavailable.
2. The design is cloud-portable because each local component maps to a future
   AWS equivalent in `docs/ARCHITECTURE.md`.
3. Each stage has a narrow responsibility and publishes evidence.
4. Tests protect operational contracts, not just function outputs.
5. Artifact hashes prevent training-serving skew.
6. MLflow provides experiment traceability.
7. Future Docker, CI, and observability work extends the same contracts rather
   than changing the pipeline design.

This is the signal a production-minded reviewer cares about: not "I trained a
model," but "I built a system where data, artifacts, metrics, and runtime
behavior can be trusted."

## Next Build Targets

The next implementation work should follow this order:

1. Build the FastAPI inference service that loads `model.joblib`,
   `preprocessor.joblib`, and `manifest.json`.
2. Add startup checks that recompute artifact hashes and fail if they do not
   match the manifest.
3. Add `/health`, `/predict`, and `/metrics` endpoints.
4. Add Docker packaging for the inference service.
5. Add docker-compose with FastAPI, Prometheus, and Grafana.
6. Add GitHub Actions CI with tests, Docker build, and Trivy scan.

That order keeps the project honest: serving depends on trained artifacts,
Docker depends on a working service, observability depends on emitted metrics,
and CI proves the build works away from the author machine.
