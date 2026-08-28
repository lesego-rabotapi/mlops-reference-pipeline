from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    # Numeric features (src/config/feature_config.py: NUMERIC_FEATURES)
    transaction_amount: float = Field(..., ge=0)
    num_items: float = Field(..., ge=0)
    customer_age: float = Field(..., ge=0, le=120)
    prev_transactions: float = Field(..., ge=0)
    distance_from_home: float = Field(..., ge=0)
    network_quality: float = Field(..., ge=0, le=100)
    velocity_score: float

    # Categorical features (src/config/feature_config.py: CATEGORICAL_FEATURES)
    hour_of_day: float
    is_weekend: float
    device_type: float
    is_first_transaction: float
    store_type: float


class PredictionResponse(BaseModel):
    is_fraud_prediction: int
    fraud_probability: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
