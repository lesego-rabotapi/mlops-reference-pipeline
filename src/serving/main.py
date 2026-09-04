import json
import logging
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from src.config.paths import MANIFEST_PATH, MODEL_PATH, PREPROCESSOR_PATH
from src.serving.schemas import HealthResponse, PredictionRequest, PredictionResponse
from src.training.train_model import sha256_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PREDICTIONS_TOTAL = Counter(
    "predictions_total", "Total number of predictions served"
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds", "Time spent computing a single prediction"
)


def load_and_verify_artifacts() -> tuple[object, object]:
    """
    Load the model and preprocessor, but only after checking both match
    the manifest train_model.py wrote when they were fitted together.

    A model and a preprocessor trained in different runs can each load and
    predict without error while being silently incompatible -- same
    column order assumption, different actual encoding underneath. The
    manifest hash check is what turns that into a startup crash instead of
    a wrong prediction nobody notices.
    """
    if not MANIFEST_PATH.exists():
        raise RuntimeError(
            f"No manifest found at {MANIFEST_PATH}. Run the training stage first."
        )

    manifest = json.loads(MANIFEST_PATH.read_text())

    actual_model_hash = sha256_file(MODEL_PATH)
    if actual_model_hash != manifest["model_hash"]:
        raise RuntimeError(
            f"model.joblib does not match the manifest's recorded hash. "
            f"Expected {manifest['model_hash']}, got {actual_model_hash}. "
            "This model was not the one the manifest was written for -- "
            "re-run training rather than serving a mismatched artifact."
        )

    actual_preprocessor_hash = sha256_file(PREPROCESSOR_PATH)
    if actual_preprocessor_hash != manifest["preprocessor_hash"]:
        raise RuntimeError(
            f"preprocessor.joblib does not match the manifest's recorded hash. "
            f"Expected {manifest['preprocessor_hash']}, got {actual_preprocessor_hash}. "
            "Re-run training rather than serving a mismatched artifact."
        )

    logger.info("Model and preprocessor hashes verified against manifest.")
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, preprocessor = load_and_verify_artifacts()
    app.state.model = model
    app.state.preprocessor = preprocessor
    logger.info("Inference API ready.")
    yield
    logger.info("Inference API shutting down.")


app = FastAPI(title="Fraud Detection Inference API", lifespan=lifespan)

# Keyed by remote address since this API has no auth layer to key on
# instead. "1/minute" is deliberately tight -- see docs/ENGINEERING_LOG.md
# for why.
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report whether the service is up and its artifacts are loaded."""
    model_loaded = getattr(app.state, "model", None) is not None
    return HealthResponse(status="ok" if model_loaded else "degraded", model_loaded=model_loaded)


@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("1/minute")
def predict(request: Request, payload: PredictionRequest) -> PredictionResponse:
    """
    Score one transaction.

    Pydantic already rejects a structurally invalid request (missing
    field, wrong type, out-of-range value per schemas.py's bounds) before
    this function ever runs, returning a 422 with the specific field that
    failed. What's left to handle here is a valid-shaped request the
    preprocessor or model can't actually score.
    """
    with PREDICTION_LATENCY.time():
        row = pd.DataFrame([payload.model_dump()])
        try:
            transformed = app.state.preprocessor.transform(row)
            probability = float(app.state.model.predict_proba(transformed)[0, 1])
        except Exception as exc:
            logger.exception("Prediction failed for a valid-shaped request: %s", exc)
            raise HTTPException(
                status_code=500, detail="Prediction failed. See server logs."
            ) from exc

    PREDICTIONS_TOTAL.inc()
    return PredictionResponse(
        is_fraud_prediction=int(probability >= 0.5),
        fraud_probability=probability,
    )


@app.get("/metrics")
def metrics() -> Response:
    """Expose metrics in Prometheus's text exposition format."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
