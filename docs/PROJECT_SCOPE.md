# Project Scope

## Scope Change

Direct AWS implementation is no longer part of this project because AWS account
access is unavailable. The project will be completed as a local-first,
cloud-portable MLOps reference pipeline.

This is a deliberate engineering constraint, not a downgrade in standards. The
pipeline should still demonstrate production MLOps thinking through reliable
interfaces, reproducible commands, persisted artifacts, testable stages, and
clear operational documentation.

## Objective

Complete an end-to-end MLOps pipeline that can be run locally and later adapted
to a cloud provider.

The completed project should demonstrate:

- Data validation and data contracts
- Feature engineering with reusable preprocessing artifacts
- Model training and evaluation
- Experiment tracking with MLflow
- Model and preprocessing artifact persistence
- FastAPI inference serving with health, predict, and metrics endpoints
- Docker-based local runtime for the inference service
- docker-compose orchestration of FastAPI, Prometheus, and Grafana
- GitHub Actions CI running install, tests, and Trivy container security scans
- Local observability through Prometheus metrics and Grafana dashboards
- Governance documentation covering versioning, approval, rollback, and
  retention decisions
- Cloud-portable architecture documentation mapping local components to AWS

## Tooling Decisions

Tools are included only when they solve a real problem at this project's scale.

Docker is included because the inference service needs a portable, reproducible
runtime boundary. FastAPI alone is not enough — a reviewer must be able to run
the service without a local Python environment.

GitHub Actions is included because the Makefile already structures the pipeline
as CI-friendly commands. Automating those commands in a workflow is low cost and
high signal.

Trivy is included because container security scanning in CI is standard practice
and adds one workflow step for meaningful security coverage.

Prometheus and Grafana are included because the FastAPI service will expose a
metrics endpoint, and a local observability stack demonstrates the same pattern
used in production without requiring cloud infrastructure.

Kubernetes, Helm, Argo CD, Kustomize, Alertmanager, and OpenTelemetry are
intentionally excluded. They solve orchestration, packaging, and distributed
tracing problems that do not exist at this scale. Adding them would demonstrate
tool installation, not engineering judgment.

Terraform is excluded from the codebase because there is no live cloud
infrastructure to provision. AWS service mappings are documented in
docs/ARCHITECTURE.md instead.

## Out Of Scope

The following are intentionally out of scope for the current version:

- AWS S3 bucket creation
- AWS IAM role or policy implementation
- AWS EC2, Lambda, ECS, or SageMaker deployment
- CloudWatch dashboards or alarms
- Terraform-managed AWS infrastructure
- Live cloud-hosted inference endpoints
- Kubernetes and related orchestration tooling (Helm, Argo CD, Kustomize)
- Distributed tracing (OpenTelemetry)
- Multi-service alerting (Alertmanager)

## Cloud-Portable Mapping

The project explains how local components map to cloud services in a future
implementation. A detailed mapping with reasoning is in docs/ARCHITECTURE.md.

| Local / Portable Component | Future Cloud Equivalent |
| --- | --- |
| `data/` and `artifacts/` directories | S3 with versioning and lifecycle policies |
| Validation reports | S3-backed data quality artifacts |
| MLflow local tracking | SageMaker Experiments or managed MLflow |
| FastAPI + Docker | ECS Fargate or SageMaker Endpoint |
| Docker image | ECR with vulnerability scanning |
| GitHub Actions CI | GitHub Actions with OIDC to assume an IAM role |
| Prometheus + Grafana | CloudWatch Metrics and Dashboards |
| Application logs | CloudWatch Logs |
| Makefile pipeline commands | Step Functions state machine |

## Success Criteria

The project is complete when a reviewer can:

- Create the Python environment from documented steps
- Run validation, feature engineering, training, and evaluation locally
- Inspect persisted reports, metrics, and artifacts
- Start the inference API locally or in Docker
- Send a prediction request and receive a valid response
- Run tests through one documented command
- Understand how the design would migrate to cloud infrastructure later

## Portfolio Positioning

The portfolio signal is now:

> Infrastructure-oriented MLOps pipeline designed to be cloud-portable, completed
> locally under realistic access constraints.

This still demonstrates the core skills expected from junior Cloud, DevOps, and
MLOps roles: reproducibility, automation, artifact management, API serving,
testing, observability, and operational reasoning.
