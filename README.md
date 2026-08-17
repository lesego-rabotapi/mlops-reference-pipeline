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
- Training and evaluation with MLflow tracking, metrics, artifact manifests,
  and training tests

Planned stages: FastAPI serving, Docker, GitHub Actions CI with Trivy, and
Prometheus/Grafana observability.

## Mentorship And Engineering Standards

This project is also used as an engineering apprenticeship exercise. Future
implementation, review, and design work should follow the mentorship operating
plan in:

- [docs/MENTORSHIP_OPERATING_PLAN.md](docs/MENTORSHIP_OPERATING_PLAN.md)
- [docs/PROJECT_SCOPE.md](docs/PROJECT_SCOPE.md)
- [docs/LOCAL_COMPLETION_GUIDE.md](docs/LOCAL_COMPLETION_GUIDE.md)
- [docs/BUILD.md](docs/BUILD.md)
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)

These documents define how decisions should be explained, how code should be
reviewed, how the project should be completed locally, and how every
implementation should connect back to production MLOps concerns.

## Local Commands

```bash
make install
make validate
make features
make train
make test

# Once serving and observability stages are complete:
make serve          # start FastAPI locally
docker compose up   # start FastAPI + Prometheus + Grafana
```
