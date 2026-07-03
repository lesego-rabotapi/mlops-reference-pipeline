# Python Environment

This project expects Python 3.12 and a local virtual environment named `.venv`.

The environment is part of the project's reproducibility story. A future
engineer should be able to clone the repository, create the environment, install
dependencies, and run tests without relying on machine-specific paths.

## Setup

From the repository root:

```powershell
.\scripts\bootstrap_env.ps1 -Python python
```

If `python` is not available on PATH, install Python 3.12 and rerun the command.

## Activate

```powershell
.\.venv\Scripts\Activate.ps1
```

## Verify

```powershell
.\.venv\Scripts\python.exe scripts\check_environment.py
.\.venv\Scripts\python.exe -m pytest tests
```

## Why This Matters

The validation, feature engineering, training, inference, and CI/CD stages all
depend on the same Python dependency set. A broken or machine-specific virtual
environment makes pipeline results hard to reproduce and weakens the portfolio
signal of the project.

Because direct AWS implementation is out of scope, the local Python environment
is now the main reproducibility contract for the project. It must be reliable
enough for a reviewer to run the pipeline without cloud credentials.
