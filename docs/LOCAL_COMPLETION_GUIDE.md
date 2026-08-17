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
- `data/validated/fraud_validated.csv` exists after a successful run

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

- FastAPI loads the saved model and preprocessor at startup
- Startup asserts that model and preprocessor artifacts are compatible using
  a saved manifest hash (prevents silent training-serving skew)
- `/health` confirms service readiness
- `/predict` returns predictions for valid input
- `/metrics` exposes Prometheus-format metrics
- Invalid input returns a structured error response

Evidence:

- API tests pass
- Local prediction example is documented

### 7. Containerization

Required outcome:

- Dockerfile builds the inference service image
- Container loads model and preprocessor artifacts from a mounted or copied
  artifact path
- No cloud credentials are required to build or run the container
- docker-compose brings up FastAPI, Prometheus, and Grafana together

Evidence:

- `Dockerfile` exists
- `docker-compose.yml` exists with FastAPI, Prometheus, and Grafana services
- `docker compose up` starts the full local stack
- Prometheus scrapes the FastAPI metrics endpoint
- Grafana dashboard shows request rate and latency

### 8. CI/CD Readiness

Required outcome:

- GitHub Actions workflow runs install, environment check, and all tests
- Trivy scans the Docker image for vulnerabilities as a CI step
- No cloud credentials are required

Evidence:

- `.github/workflows/ci.yml` exists
- Workflow passes on a clean clone
- Trivy scan step runs and reports results

### 9. Governance And Operations

Required outcome:

- Governance document explains data, model, artifact, approval, rollback, and
  retention practices
- Architecture document maps every local component to its AWS equivalent with
  reasoning
- Monitoring document explains local Prometheus metrics and future cloud mapping

Evidence:

- `docs/GOVERNANCE.md` exists
- `docs/ARCHITECTURE.md` exists
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

Once serving and observability stages are complete, the full flow is:

```powershell
# Run the pipeline
make validate
make features
make train

# Start the inference service locally
make serve

# Start the full observability stack
docker compose up

# Run the CI workflow locally (requires act or GitHub Actions)
# Or push to GitHub and let the workflow run automatically
```

The project is fully done when a reviewer can clone the repository, run the
pipeline commands, start the inference API, send a prediction request, and open
the Grafana dashboard — all without AWS credentials.
