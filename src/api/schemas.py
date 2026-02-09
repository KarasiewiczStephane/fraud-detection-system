"""Pydantic schemas for the Fraud Detection API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TransactionInput(BaseModel):
    """Input schema for a single transaction."""

    transaction_id: str
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(ge=0)

    model_config = ConfigDict(extra="forbid")


class PredictionOutput(BaseModel):
    """Output schema for a single prediction."""

    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    confidence: float
    model_version: str
    explanation: Optional[Dict[str, Any]] = None


class BatchInput(BaseModel):
    """Input schema for batch predictions."""

    transactions: List[TransactionInput] = Field(max_length=100)


class BatchOutput(BaseModel):
    """Output schema for batch predictions."""

    predictions: List[PredictionOutput]


class HealthResponse(BaseModel):
    """Output schema for the health endpoint."""

    status: str
    model_version: str
