from typing import Literal

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    One PaySim-style transaction. Raw fields only: the balance-error
    features are computed server-side by src/features/engineering.py, the
    same function training uses, so callers cannot send inconsistent values.

    `type` is restricted to the transaction types the model was trained and
    evaluated on (feature_config.SCORED_TRANSACTION_TYPES). PaySim has no
    fraud in the other types, so scoring them would be out-of-scope
    extrapolation; they get a 422 instead of a meaningless score.
    """

    type: Literal["TRANSFER", "CASH_OUT"]
    amount: float = Field(..., ge=0)
    oldbalanceOrg: float = Field(..., ge=0)
    newbalanceOrig: float = Field(..., ge=0)
    oldbalanceDest: float = Field(..., ge=0)
    newbalanceDest: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    is_fraud_prediction: int
    fraud_probability: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
