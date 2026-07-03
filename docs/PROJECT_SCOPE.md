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
- FastAPI inference serving
- Docker-based local runtime
- CI-ready test and pipeline commands
- Local observability through logs, reports, and metrics artifacts
- Governance documentation covering versioning, approval, rollback, and
  retention decisions

## Out Of Scope

The following are intentionally out of scope for the current version:

- AWS S3 bucket creation
- AWS IAM role or policy implementation
- AWS EC2, Lambda, ECS, or SageMaker deployment
- CloudWatch dashboards or alarms
- Terraform-managed AWS infrastructure
- Live cloud-hosted inference endpoints

## Cloud-Portable Mapping

The project should still explain how local components would map to cloud
services in a future implementation.

| Local / Portable Component | Future Cloud Equivalent |
| --- | --- |
| `data/` and `artifacts/` directories | Object storage such as S3 |
| Validation reports | Data quality artifacts in object storage |
| MLflow local tracking | Managed or remote MLflow tracking server |
| FastAPI local service | Container service, VM, or serverless endpoint |
| Application logs | Cloud logging service |
| Metrics artifacts | Monitoring and dashboard service |
| Makefile / scripts | CI/CD workflow steps |
| Docker image | Deployable runtime artifact |

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
