# Why this file exists: builds a portable, reproducible runtime for the
# inference service, so a reviewer doesn't need a local Python environment.
# Responsible for: installing dependencies, packaging the fitted model and
# preprocessor, and running the FastAPI app under uvicorn.
# Must not: run training, feature engineering, or validation -- those
# produce the artifacts.joblib/.json files this image only ever copies in
# already-built.
FROM python:3.12-slim

WORKDIR /app

# Dependencies copied and installed before source, so an edit to src/
# doesn't bust this layer's build cache on rebuild.
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY artifacts/ ./artifacts/
RUN pip install --no-cache-dir -e .

RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "src.serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
