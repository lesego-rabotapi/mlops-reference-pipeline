# Environment Setup

This project requires Python 3.12 and Docker. Both must be available to run the
full pipeline and observability stack locally.

The environment is part of the project's reproducibility story. A future
engineer should be able to clone the repository, satisfy these prerequisites,
and run the pipeline without cloud credentials or machine-specific configuration.

## Prerequisites

- Python 3.12
- Docker Desktop (for the inference service and observability stack)
- Git

Docker is required from Stage 7 onward. The pipeline stages (validation,
feature engineering, training) run in the Python environment only.

## Python Environment Setup

From the repository root:

```powershell
.\scripts\bootstrap_env.ps1 -Python python
```

If `python` is not available on PATH, install Python 3.12 and rerun the command.

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Verify the environment:

```powershell
.\.venv\Scripts\python.exe scripts\check_environment.py
.\.venv\Scripts\python.exe -m pytest tests
```

## Docker Setup

Docker is used for two purposes in this project:

1. Packaging the FastAPI inference service into a portable image
2. Running the local observability stack (Prometheus + Grafana) via docker-compose

Verify Docker is available:

```bash
docker --version
docker compose version
```

Both commands should return version strings. If Docker is not installed, install
Docker Desktop from https://www.docker.com/products/docker-desktop.

## Why This Matters

The validation, feature engineering, training, inference, and CI stages all
depend on the same Python dependency set. A broken environment makes pipeline
results hard to reproduce.

Docker removes the "works on my machine" problem for the inference service. A
reviewer should be able to build the Docker image and start the API without
installing project dependencies locally. This is the same contract that a
production deployment expects — the image is the deployment unit, not the
developer's laptop environment.
