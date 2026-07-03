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
for this version of the project. The pipeline will still be completed with
production-style engineering practices using local and portable components:

- Local artifact storage instead of S3
- Local logs and generated reports instead of CloudWatch
- Local or containerized FastAPI serving instead of EC2 or Lambda
- CI-ready commands instead of cloud deployment jobs
- Cloud-portable design documentation instead of Terraform-provisioned AWS
  infrastructure

This keeps the project honest and complete: it demonstrates the same MLOps
engineering concepts without depending on unavailable cloud access.

## Current Pipeline Stages

```text
Raw Data
  -> Validation
  -> Validated Dataset
  -> Feature Engineering
  -> Processed Features
  -> Training
  -> Experiment Tracking
  -> Local Serving
  -> Monitoring
```

Implemented stages currently include:

- Data validation structure
- Reusable validation rules
- Feature engineering with scikit-learn pipelines
- Preprocessing artifact persistence
- Feature engineering tests

Planned maturity areas include training, MLflow tracking, FastAPI serving,
Dockerization, CI/CD, local monitoring, governance documentation, and
cloud-portable deployment notes.

## Mentorship And Engineering Standards

This project is also used as an engineering apprenticeship exercise. Future
implementation, review, and design work should follow the mentorship operating
plan in:

- [docs/MENTORSHIP_OPERATING_PLAN.md](docs/MENTORSHIP_OPERATING_PLAN.md)
- [docs/PROJECT_SCOPE.md](docs/PROJECT_SCOPE.md)
- [docs/LOCAL_COMPLETION_GUIDE.md](docs/LOCAL_COMPLETION_GUIDE.md)
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)

These documents define how decisions should be explained, how code should be
reviewed, how the project should be completed locally, and how every
implementation should connect back to production MLOps concerns.

## Local Commands

```bash
make install
make validate
make features
make test
```
