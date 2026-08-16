# Deployment Strategy

## Current Strategy

Direct AWS deployment is out of scope because AWS account access is unavailable.
The project uses a local-first deployment strategy with production-grade tooling:

1. Run the full pipeline locally (validate, features, train).
2. Persist validation, feature, training, and evaluation artifacts.
3. Serve predictions through a FastAPI inference service.
4. Package the inference service with Docker.
5. Bring up the full observability stack with docker-compose.
6. Run CI through GitHub Actions including Trivy security scanning.
7. Document how the same design maps to AWS infrastructure.

## Local Runtime Target

The primary deployment target is a local FastAPI service containerized with Docker.

Endpoints:

- `/health` — readiness check, confirms model and preprocessor are loaded
- `/predict` — accepts a JSON feature payload, returns a prediction and probability
- `/metrics` — Prometheus-format metrics for scraping

The inference service loads at startup:

- The trained model artifact (`artifacts/training/model.joblib`)
- The fitted preprocessor artifact (`artifacts/features/preprocessor.joblib`)
- The artifact manifest (`artifacts/training/manifest.json`)

On startup, the service recomputes artifact hashes and asserts they match the
saved manifest. A hash mismatch means the model and preprocessor are out of
sync, which would cause silent wrong predictions. The service refuses to start
rather than serve corrupted output.

## Docker Target

Docker is the portable runtime boundary for the inference service.

The Dockerfile should:

- Use a slim Python base image
- Install dependencies from `requirements.txt`
- Copy source code and artifacts into the image
- Start the FastAPI service with Uvicorn on port 8000

Build and run:

```bash
docker build -t mlops-pipeline:latest .
docker run -p 8000:8000 mlops-pipeline:latest
```

## Observability Stack

docker-compose brings up the full local observability stack:

- **FastAPI** — inference service exposing `/metrics`
- **Prometheus** — scrapes the FastAPI metrics endpoint every 15 seconds
- **Grafana** — visualises Prometheus metrics with a pre-configured dashboard

```bash
docker compose up
```

Grafana is available at `http://localhost:3000`. Prometheus at
`http://localhost:9090`.

## CI/CD Target

GitHub Actions runs the CI workflow on every push and pull request.

Required CI steps:

1. Check out the repository
2. Set up Python 3.12
3. Install dependencies
4. Run environment check
5. Run all tests (`pytest tests/`)
6. Build the Docker image
7. Run Trivy to scan the image for vulnerabilities

Trivy runs as a non-blocking scan by default (reports vulnerabilities without
failing the build). This can be tightened to fail on HIGH or CRITICAL severity
findings once the image baseline is established.

No cloud credentials are required for any CI step.

## Future Cloud Mapping

A detailed mapping with reasoning is in `docs/ARCHITECTURE.md`. Summary:

| Current Component | AWS Equivalent | Notes |
| --- | --- | --- |
| `artifacts/` directory | S3 with versioning | Lifecycle policies for retention |
| FastAPI + Docker | ECS Fargate or SageMaker Endpoint | Fargate for persistent service; Endpoint for managed scaling |
| Docker image | ECR | Built in CI, pushed on merge to main |
| GitHub Actions CI | GitHub Actions with OIDC | OIDC lets CI assume an IAM role without stored credentials |
| Prometheus + Grafana | CloudWatch Metrics + Dashboards | Same scraping model, managed backend |
| MLflow local | SageMaker Experiments | Drop-in for experiment tracking at scale |
| Makefile pipeline | Step Functions | Adds retry logic, parallel execution, and audit trail per step |

## Production Considerations

Before any real deployment, the project would also need:

- Secrets management (AWS Secrets Manager or Parameter Store)
- Authentication and authorization on the inference endpoint
- Artifact versioning and promotion policy
- Rollback procedure tied to artifact versions
- Alerting rules on model latency and error rate
- Resource cost controls and autoscaling limits
- Data retention and deletion policy
