"""
Contract tests for the inference API.

Why this file exists: protects the manifest-hash compatibility check and the
request/response contract from regressions.
Responsible for: exercising the API against a real, small, fitted
preprocessor and model -- not mocks -- so a genuinely broken .transform() or
.predict_proba() call would actually fail a test.
Must not: depend on repository data or the real trained artifacts.
"""

import json

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import src.serving.main as serving
from src.config.feature_config import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.features.build_features import build_preprocessor
from src.training.train_model import build_model, sha256_file

VALID_PAYLOAD = {
    "transaction_amount": 42.50,
    "num_items": 3,
    "customer_age": 34,
    "prev_transactions": 5,
    "distance_from_home": 12.3,
    "network_quality": 80,
    "velocity_score": 0.4,
    "hour_of_day": 2,
    "is_weekend": 0,
    "device_type": 1,
    "is_first_transaction": 0,
    "store_type": 1,
}


def _synthetic_training_frame(n: int = 40) -> pd.DataFrame:
    """A tiny frame with every column build_preprocessor() expects, nothing else."""
    rows = []
    for i in range(n):
        rows.append(
            {
                "transaction_amount": float(i),
                "num_items": float(i % 5),
                "customer_age": float(20 + i % 50),
                "prev_transactions": float(i % 10),
                "distance_from_home": float(i % 20),
                "network_quality": float(i % 100),
                "velocity_score": float(i) / n,
                "hour_of_day": float(1 + i % 3),
                "is_weekend": float(i % 2),
                "device_type": float(i % 3),
                "is_first_transaction": float(i % 2),
                "store_type": float(i % 2),
            }
        )
    return pd.DataFrame(rows)


def _write_matching_artifacts(tmp_path) -> dict[str, object]:
    """
    Fit a real preprocessor and model on synthetic data shaped like
    feature_config.py's contract, then write them plus a correct manifest.

    Fitting for real (not stubbing with a dict, like train_model's own
    tests do) matters here specifically because /predict actually calls
    .transform() and .predict_proba() -- a fake artifact would hide a
    genuinely broken pipeline instead of exercising it.
    """
    assert set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES) == set(VALID_PAYLOAD)

    X = _synthetic_training_frame()
    y = pd.Series([i % 2 for i in range(len(X))])

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)

    model = build_model()
    model.fit(transformed, y)

    model_path = tmp_path / "model.joblib"
    preprocessor_path = tmp_path / "preprocessor.joblib"
    manifest_path = tmp_path / "manifest.json"

    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    manifest_path.write_text(
        json.dumps(
            {
                "model_hash": sha256_file(model_path),
                "preprocessor_hash": sha256_file(preprocessor_path),
            }
        )
    )
    return {"model": model_path, "preprocessor": preprocessor_path, "manifest": manifest_path}


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """
    The Limiter's request counts live in module-level storage on
    serving.app, not per-TestClient -- without this, whichever test
    happens to run after enough /predict calls would start seeing 429s
    caused by an earlier test's traffic, not its own.
    """
    serving.limiter.reset()
    yield


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    """A TestClient wired to a real, matching, temporary artifact set."""
    paths = _write_matching_artifacts(tmp_path)
    monkeypatch.setattr(serving, "MODEL_PATH", paths["model"])
    monkeypatch.setattr(serving, "PREPROCESSOR_PATH", paths["preprocessor"])
    monkeypatch.setattr(serving, "MANIFEST_PATH", paths["manifest"])
    with TestClient(serving.app) as client:
        yield client


def test_health_reports_model_loaded(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_a_probability(api_client):
    response = api_client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["is_fraud_prediction"] in (0, 1)
    assert 0.0 <= body["fraud_probability"] <= 1.0


def test_predict_rejects_missing_field(api_client):
    incomplete = dict(VALID_PAYLOAD)
    del incomplete["transaction_amount"]
    response = api_client.post("/predict", json=incomplete)
    assert response.status_code == 422
    assert "transaction_amount" in response.text


def test_predict_rejects_out_of_range_value(api_client):
    invalid = dict(VALID_PAYLOAD, customer_age=250)
    response = api_client.post("/predict", json=invalid)
    assert response.status_code == 422
    assert "customer_age" in response.text


def test_metrics_endpoint_is_prometheus_format(api_client):
    response = api_client.get("/metrics")
    assert response.status_code == 200
    assert "predictions_total" in response.text


def test_startup_fails_loudly_on_a_tampered_model(tmp_path, monkeypatch):
    """
    The whole point of the manifest hash check is that a mismatched model
    can't silently start serving. Tamper with the model after the manifest
    was written, and startup should refuse rather than degrade quietly.
    """
    paths = _write_matching_artifacts(tmp_path)
    joblib.dump({"not": "the real model"}, paths["model"])  # overwrite post-manifest

    monkeypatch.setattr(serving, "MODEL_PATH", paths["model"])
    monkeypatch.setattr(serving, "PREPROCESSOR_PATH", paths["preprocessor"])
    monkeypatch.setattr(serving, "MANIFEST_PATH", paths["manifest"])

    with pytest.raises(RuntimeError, match="does not match the manifest"):
        with TestClient(serving.app):
            pass


def test_predict_is_rate_limited(api_client):
    """
    /predict is capped at 1/minute per client (main.py). The 2nd request
    from the same TestClient (same key under get_remote_address) within
    the window should be rejected with 429, not served.
    """
    response = api_client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200

    response = api_client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 429
