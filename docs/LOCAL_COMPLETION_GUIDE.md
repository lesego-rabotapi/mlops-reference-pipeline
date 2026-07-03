# Local Completion Guide

## Purpose

This guide replaces cloud-specific implementation steps with local-first
delivery steps. The goal is to finish the pipeline without AWS access while
preserving production-style MLOps architecture.

## Completion Roadmap

### 1. Environment Reproducibility

Required outcome:

- Python 3.12 environment can be created from documented commands
- Dependencies install from `requirements.txt`
- Environment check script passes
- Tests can run locally

Evidence:

- `scripts/check_environment.py` passes
- `pytest tests` passes

### 2. Validation

Required outcome:

- Raw data is validated before downstream stages run
- Invalid data fails the command
- Validation report is written to `artifacts/validation`
- Validated data is written to `data/validated`

Evidence:

- Validation tests pass
- Validation report JSON exists
- `data/validated/customer_churn_validated.csv` exists after a successful run

### 3. Feature Engineering

Required outcome:

- Validated data is transformed into train/test feature matrices
- Preprocessor is fitted only on training data
- Preprocessor artifact is saved for inference reuse
- Feature metadata is saved

Evidence:

- Feature tests pass
- `artifacts/features/preprocessor.joblib` exists
- Processed train/test files exist

### 4. Training And Evaluation

Required outcome:

- Training code consumes processed features
- Model is trained reproducibly
- Metrics are generated and persisted
- Model artifact is saved

Evidence:

- Training artifact exists
- Evaluation report exists
- Metrics are logged locally and through MLflow

### 5. Experiment Tracking

Required outcome:

- MLflow records parameters, metrics, and artifacts
- A reviewer can compare runs locally

Evidence:

- `.mlflow/` or configured MLflow tracking directory contains runs
- README explains how to inspect runs

### 6. Local Inference API

Required outcome:

- FastAPI loads the saved model and preprocessor
- `/health` confirms service readiness
- `/predict` returns predictions for valid input
- Invalid input returns a useful error

Evidence:

- API tests pass
- Local prediction example is documented

### 7. Containerization

Required outcome:

- Docker image can run the inference service locally
- Container uses persisted model and preprocessing artifacts

Evidence:

- Dockerfile exists
- Local Docker run command is documented

### 8. CI/CD Readiness

Required outcome:

- CI workflow runs setup, tests, and pipeline checks
- No cloud credentials are required

Evidence:

- GitHub Actions workflow exists
- Workflow runs validation and tests

### 9. Governance And Operations

Required outcome:

- Governance document explains data, model, artifact, approval, rollback, and
  retention practices
- Monitoring document explains local metrics and future cloud mapping

Evidence:

- `docs/GOVERNANCE.md` exists
- `docs/MONITORING.md` exists

## Definition Of Done

The project is complete when the full local pipeline can be reproduced from a
fresh clone without AWS credentials.

Minimum command flow:

```powershell
.\scripts\bootstrap_env.ps1 -Python python
.\.venv\Scripts\python.exe scripts\check_environment.py
.\.venv\Scripts\python.exe -m pytest tests
make validate
make features
make train
```

FastAPI and Docker commands should be added once those stages exist.
