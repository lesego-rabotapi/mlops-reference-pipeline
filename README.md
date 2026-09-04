# MLOps Reference Pipeline

This repository is a production-oriented, local-first MLOps reference pipeline
focused on reproducibility, governance, observability, deployment readiness, and
clean engineering practices.

The project is intentionally positioned as an Infrastructure-Oriented Machine
Learning Engineering portfolio project. The goal is not to maximize model
complexity. The goal is to demonstrate that each stage of the ML lifecycle can
be built, tested, documented, and operated as part of a reliable system.

## Current Scope

AWS account access is unavailable, so direct AWS implementation is out of scope
for this version of the project. The pipeline is completed with production-style
engineering practices using local and portable components:

- Local artifact storage instead of S3
- Docker for portable runtime packaging instead of EC2 or Lambda
- FastAPI inference service containerized with Docker
- Prometheus and Grafana for local observability instead of CloudWatch
- GitHub Actions for CI instead of CodePipeline
- Trivy for container security scanning in CI
- Cloud-portable architecture documentation instead of live Terraform infrastructure

This keeps the project honest and complete: it demonstrates the same MLOps
engineering concepts without depending on unavailable cloud access.

## Tooling

| Layer | Tool | Role in this project |
| --- | --- | --- |
| Pipeline | Python + scikit-learn | Validation, feature engineering, training |
| Experiment tracking | MLflow | Parameter, metric, and artifact logging |
| Serving | FastAPI + Uvicorn | Inference API with health and predict endpoints |
| Containers | Docker | Portable runtime for the inference service |
| CI | GitHub Actions | Automated install, test, and security scan |
| Security scanning | Trivy | Container image vulnerability scanning in CI |
| Metrics | Prometheus | Scrapes metrics from the FastAPI service |
| Dashboards | Grafana | Visualises Prometheus metrics locally |
| Orchestration | docker-compose | Brings up FastAPI, Prometheus, and Grafana together |

## Pipeline Stages

```text
Raw Data
  -> Validation            (src/validation/)
  -> Validated Dataset
  -> Feature Engineering   (src/features/)
  -> Processed Features + Preprocessor Artifact
  -> Training              (src/training/)
  -> Model Artifact + Evaluation Report + MLflow Run
  -> Inference API         (src/serving/)
  -> Docker Image
  -> CI (GitHub Actions + Trivy)
  -> Local Observability   (Prometheus + Grafana via docker-compose)
```

Implemented stages:

- Data validation with reusable rule registry and JSON audit reports,
  including per-column MCAR/MNAR missingness analysis and evidence-backed
  imputation policies -- see [docs/MISSINGNESS_ANALYSIS.md](docs/MISSINGNESS_ANALYSIS.md)
- Feature engineering with scikit-learn pipelines and artifact persistence
- Feature engineering tests including leakage detection
- Training and evaluation with MLflow tracking, imbalance-aware metrics,
  artifact manifests, and training tests
- Dataset predictive-signal assessment, separate from structural
  validation -- see [docs/DATASET_ASSESSMENT.md](docs/DATASET_ASSESSMENT.md)
- FastAPI inference service with a startup manifest-hash check against
  training-serving skew, `/health`, `/predict`, and Prometheus-format
  `/metrics`
- Docker image and a docker-compose stack (FastAPI + Prometheus +
  Grafana) for local observability, both verified end to end
- GitHub Actions CI: install, test, build the image, and scan it with
  Trivy

**Current status:** the raw dataset (`fraud_raw.csv`) passes every
structural validation check but was assessed and rejected for supervised
learning -- statistical evidence points to `is_fraud` being generated
independently of the other features (see
[docs/DATASET_ASSESSMENT.md](docs/DATASET_ASSESSMENT.md)). A PaySim
alternative was evaluated and rejected as an unnecessary pivot; the
project stays on this dataset by decision, not by default, with the
deliverable framed around a pipeline that correctly identifies its input
can't support a supervised model, rather than a working classifier. See
[docs/ENGINEERING_LOG.md](docs/ENGINEERING_LOG.md), Entries 7 and 9.

See [docs/ENGINEERING_LOG.md](docs/ENGINEERING_LOG.md) for the reasoning
behind these decisions -- not just what was built, but why, what
alternatives were rejected, and what went wrong along the way.

## Mentorship And Engineering Standards

This project is also used as an engineering apprenticeship exercise. Future
implementation, review, and design work should follow the standards in:

- [docs/PROJECT_SCOPE.md](docs/PROJECT_SCOPE.md)
- [docs/LOCAL_COMPLETION_GUIDE.md](docs/LOCAL_COMPLETION_GUIDE.md)
- [docs/BUILD.md](docs/BUILD.md)
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — component decisions and the AWS service mapping
- [docs/GOVERNANCE.md](docs/GOVERNANCE.md) — data, model, artifact, approval, rollback, and retention practices
- [docs/MONITORING.md](docs/MONITORING.md) — what's actually monitored locally, verified working, and how it maps to CloudWatch

These documents define how decisions should be explained, how code should be
reviewed, how the project should be completed locally, and how every
implementation should connect back to production MLOps concerns.

## Inspecting MLflow Runs

Every `make train` run is logged to a project-local MLflow tracking
directory (`.mlflow/`). To browse runs, compare metrics across them, or
pull an older run's artifacts:

```bash
mlflow ui --backend-store-uri file://$(pwd)/.mlflow
```

Then open `http://localhost:5000`. Each run's page shows its parameters,
metrics, and logged artifacts (model, evaluation report, manifest) — the
same three files `artifacts/training/` holds for the most recent run, but
kept for every run rather than overwritten by the next one. See
[docs/GOVERNANCE.md](docs/GOVERNANCE.md) for how this doubles as the
project's de facto model history in the absence of a formal registry.

## Local Commands

```bash
make install
make validate
make features
make train
make test
make serve          # start FastAPI locally
docker compose up   # start FastAPI + Prometheus + Grafana
```
