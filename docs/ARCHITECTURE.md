# Architecture

## Design Philosophy

This project is local-first and cloud-portable. Every component is chosen
because it solves a real problem at the current scale, not because it resembles
a production system. The architecture is honest about what it is: a single-node
ML pipeline with one model, one dataset, and one inference service.

The cloud mapping section documents how each local component translates to AWS.
This is not aspirational documentation — it is the reasoning that would drive
actual infrastructure decisions if cloud access were available.

## Local Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                        Pipeline                             │
│                                                             │
│  data/raw/  →  Validation  →  data/validated/              │
│                    ↓                                        │
│              artifacts/validation/  (JSON report)           │
│                                                             │
│  data/validated/  →  Feature Engineering  →  data/processed/│
│                           ↓                                 │
│                   artifacts/features/  (preprocessor.joblib)│
│                                                             │
│  data/processed/  →  Training  →  artifacts/training/      │
│                         ↓                                   │
│                   model.joblib + evaluation_report.json     │
│                   manifest.json (artifact hashes)           │
│                         ↓                                   │
│                   MLflow run (local tracking)               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     Inference Stack                         │
│                                                             │
│  Docker Image                                               │
│    └── FastAPI (port 8000)                                  │
│          ├── /health    (readiness check)                   │
│          ├── /predict   (model inference)                   │
│          └── /metrics   (Prometheus format)                 │
│                                                             │
│  docker-compose                                             │
│    ├── fastapi     (inference service)                      │
│    ├── prometheus  (scrapes /metrics every 15s, port 9090)  │
│    └── grafana     (dashboards, port 3000)                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                          CI                                 │
│                                                             │
│  GitHub Actions (.github/workflows/ci.yml)                  │
│    ├── Python setup + pip install                           │
│    ├── check_environment.py                                 │
│    ├── pytest tests/                                        │
│    ├── docker build                                         │
│    └── trivy image scan                                     │
└─────────────────────────────────────────────────────────────┘
```

## Component Decisions

### FastAPI over Flask

FastAPI generates an OpenAPI schema automatically, enforces request/response
types through Pydantic, and is async-native. For an inference service where
input validation matters (wrong feature shapes cause silent prediction errors),
the automatic schema enforcement is not cosmetic — it is a correctness boundary.
Flask requires manual validation logic to achieve the same safety.

### Docker over bare process

Without Docker, running the inference service requires a matching Python
environment, the right working directory, and all artifact paths to resolve
correctly on the host machine. Docker makes the image the deployment unit. A
reviewer can run the service without knowing anything about the project's
internal structure. This is the same contract a production deployment expects.

### docker-compose for local observability

Prometheus and Grafana require configuration to scrape and visualize metrics.
docker-compose manages the three-service stack (FastAPI, Prometheus, Grafana)
as a single unit with a documented startup command. Without it, a reviewer would
need to start three processes manually and configure Prometheus datasource in
Grafana by hand.

### GitHub Actions over local scripts only

The Makefile runs the pipeline locally. GitHub Actions proves it runs on a clean
machine with no local state. This distinction matters for portfolio credibility:
a CI badge means the pipeline is reproducible, not just that it worked once on
the author's laptop.

### Trivy over no scanning

Trivy is one CI step. It scans the Docker image for known CVEs in base image
layers and installed packages. This is standard practice in any engineering team
that ships containers. The cost is one workflow line. The signal — that the
developer knows containers have a security surface — is real.

### Prometheus + Grafana over log-only observability

The FastAPI `/metrics` endpoint exposes request count, latency histograms, and
error rates in Prometheus format. Prometheus scrapes this on a schedule.
Grafana queries Prometheus and renders time-series dashboards. This is the same
observability pattern used in production Kubernetes clusters — the only
difference is that here it runs locally via docker-compose. A reviewer can open
Grafana and see live inference traffic without any cloud infrastructure.

### MLflow over manual metrics files

MLflow logs parameters, metrics, and artifact paths per run. A reviewer can run
`mlflow ui` and compare experiments. Without MLflow, comparing two training runs
means diffing JSON files manually. MLflow also provides the experiment metadata
that governance documentation references when explaining model approval and
rollback decisions.

## Artifact Manifest and Training-Serving Skew

The inference service asserts at startup that the loaded model and preprocessor
are compatible. It does this by comparing the SHA-256 hashes of both artifacts
against a manifest file (`artifacts/training/manifest.json`) generated at
training time.

If the manifest check fails, the service refuses to start. This prevents the
most common failure mode in ML serving: a model retrained with a new preprocessor
where the old preprocessor artifact was not replaced, or vice versa. The failure
is loud and immediate rather than silent and gradual.

## AWS Service Mapping

| Local Component | AWS Equivalent | Why this service |
| --- | --- | --- |
| `data/` and `artifacts/` directories | S3 with versioning | Object durability, versioning for rollback, lifecycle policies for retention |
| Validation JSON reports | S3 artifacts | Same bucket as data, different prefix, auditable by timestamp |
| MLflow local tracking | SageMaker Experiments | Managed experiment tracking without a self-hosted tracking server |
| FastAPI + Docker | ECS Fargate | Serverless container runtime; no EC2 instances to manage |
| FastAPI + Docker (alternative) | SageMaker Endpoint | Preferred for models with SageMaker-native training; handles autoscaling |
| Docker image | ECR | Private registry; Fargate and SageMaker pull directly from ECR |
| GitHub Actions CI | GitHub Actions with OIDC | OIDC lets CI assume an IAM role with short-lived credentials; no stored secrets |
| docker build step | CodeBuild | Managed build environment; integrates with ECR push |
| Trivy in CI | ECR image scanning + Inspector | ECR scans on push; Inspector provides continuous CVE tracking |
| Prometheus + Grafana | CloudWatch Metrics + Dashboards | Same scraping model; CloudWatch replaces Prometheus as the metrics backend |
| Application logs (stdout) | CloudWatch Logs | ECS Fargate routes container stdout to CloudWatch automatically |
| MLflow artifact store | S3 | MLflow supports S3 as artifact backend with no code changes |
| docker-compose local stack | ECS Service + ALB | ALB routes to the ECS task; health checks replace the docker-compose healthcheck |
| Makefile pipeline commands | Step Functions | State machine adds retry logic, parallel branches, and per-step audit logging |

## Excluded Tools and Why

**Kubernetes, Helm, Argo CD, Kustomize** — These solve container orchestration
across multiple nodes and environments. This project runs one container on one
machine. Adding Kubernetes to deploy a single inference pod would be
demonstrating tool installation, not engineering judgment. These belong in a
dedicated K8s project that uses this pipeline's Docker image as its deployment
artifact.

**Terraform** — Terraform manages live cloud infrastructure. There is no live
infrastructure to manage. The AWS mapping table above documents what Terraform
modules would provision, which is the same information a Terraform plan would
encode — without pretending to provision resources that do not exist.

**Alertmanager** — Alertmanager routes Prometheus alerts to notification
channels. Without a continuously running service in an environment where someone
is on-call, Alertmanager configuration is not testable or meaningful.

**OpenTelemetry** — OpenTelemetry solves distributed tracing across multiple
services. This project has one service. Prometheus metrics cover the observable
surface that matters at this scale.
