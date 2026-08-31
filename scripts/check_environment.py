"""Verify that the local Python environment can run the MLOps pipeline."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


REQUIRED_MODULES = [
    "pandas",
    "sklearn",
    "great_expectations",
    "mlflow",
    "fastapi",
    "uvicorn",
    "prometheus_client",
    "pytest",
    "joblib",
]


def main() -> int:
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")

    if sys.version_info < (3, 12):
        print("ERROR: Python 3.12 or newer is required.")
        return 1

    missing_modules = []
    for module_name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            missing_modules.append(module_name)
            continue

        version = getattr(module, "__version__", "unknown")
        print(f"{module_name}: {version}")

    required_paths = [
        Path("src/validation/validate_data.py"),
        Path("src/features/build_features.py"),
        Path("src/training/train_model.py"),
        Path("src/serving/main.py"),
        Path("tests/test_validate_data.py"),
        Path("tests/test_build_features.py"),
        Path("tests/test_train_model.py"),
        Path("tests/test_serving.py"),
        Path("docs/PROJECT_SCOPE.md"),
        Path("docs/LOCAL_COMPLETION_GUIDE.md"),
    ]
    missing_paths = [path for path in required_paths if not path.exists()]

    if missing_modules:
        print("ERROR: Missing Python modules: " + ", ".join(missing_modules))
        return 1

    if missing_paths:
        print("ERROR: Missing required project files:")
        for path in missing_paths:
            print(f"  - {path}")
        return 1

    print("Project scope: local-first; AWS credentials are not required.")
    print("Environment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
