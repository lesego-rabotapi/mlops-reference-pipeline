# Why this file exists: builds a portable, reproducible runtime for the
# inference service, so a reviewer doesn't need a local Python environment.
# Responsible for: installing dependencies, packaging the fitted model and
# preprocessor, and running the FastAPI app under uvicorn.
# Must not: run training, feature engineering, or validation -- those
# produce the artifacts.joblib/.json files this image only ever copies in
# already-built.
FROM python:3.12-slim

WORKDIR /app

# python:3.12-slim's Debian packages lag behind Debian's own security
# patches. Pulling the latest openssl/libssl security fixes here rather
# than trusting the base image's snapshot -- Trivy caught this gap
# (see docs/ENGINEERING_LOG.md, Entry 12).
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

# Dependencies copied and installed before source, so an edit to src/
# doesn't bust this layer's build cache on rebuild.
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --timeout=100 --retries=5 -r requirements.txt

COPY src/ ./src/
COPY artifacts/ ./artifacts/
RUN pip install --no-cache-dir -e .

RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "src.serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
