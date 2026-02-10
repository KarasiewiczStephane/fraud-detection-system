"""Prediction endpoints for the Fraud Detection API."""

from __future__ import annotations

from typing import Any, List

import numpy as np
from fastapi import APIRouter, Query

from src.api.schemas import BatchInput, BatchOutput, PredictionOutput, TransactionInput
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["predictions"])

# Feature column order expected by the model (V1-V28, Time, Amount).
_FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]


def _transaction_to_array(txn: TransactionInput) -> np.ndarray:
    """Convert a validated transaction to a 1-D feature array."""
    return np.array([getattr(txn, col) for col in _FEATURE_COLS], dtype=np.float64)


def _make_prediction(
    txn: TransactionInput,
    model: Any,
    threshold: float,
    model_version: str,
) -> PredictionOutput:
    """Run inference on a single transaction and return the result."""
    x = _transaction_to_array(txn).reshape(1, -1)
    proba = float(model.predict_proba(x)[0, 1])
    is_fraud = proba >= threshold
    confidence = proba if is_fraud else 1.0 - proba

    return PredictionOutput(
        transaction_id=txn.transaction_id,
        fraud_probability=round(proba, 6),
        is_fraud=is_fraud,
        confidence=round(confidence, 6),
        model_version=model_version,
    )


@router.post("/predict", response_model=PredictionOutput)
async def predict_single(
    transaction: TransactionInput,
    include_explanation: bool = Query(False),
) -> PredictionOutput:
    """Score a single transaction for fraud."""
    from src.api.app import get_state

    state = get_state()
    result = _make_prediction(
        transaction,
        state.model,
        state.threshold,
        state.model_version,
    )

    # Log to database
    await state.db.insert_prediction(
        transaction_id=result.transaction_id,
        features={col: getattr(transaction, col) for col in _FEATURE_COLS},
        prediction=int(result.is_fraud),
        confidence=result.confidence,
        model_version=result.model_version,
    )

    return result


@router.post("/predict/batch", response_model=BatchOutput)
async def predict_batch(batch: BatchInput) -> BatchOutput:
    """Score a batch of up to 100 transactions."""
    from src.api.app import get_state

    state = get_state()
    results: List[PredictionOutput] = []

    if not batch.transactions:
        return BatchOutput(predictions=[])

    # Vectorised inference
    X = np.vstack([_transaction_to_array(t) for t in batch.transactions])
    probas = state.model.predict_proba(X)[:, 1]

    for txn, proba in zip(batch.transactions, probas):
        proba_f = float(proba)
        is_fraud = proba_f >= state.threshold
        confidence = proba_f if is_fraud else 1.0 - proba_f

        result = PredictionOutput(
            transaction_id=txn.transaction_id,
            fraud_probability=round(proba_f, 6),
            is_fraud=is_fraud,
            confidence=round(confidence, 6),
            model_version=state.model_version,
        )
        results.append(result)

        await state.db.insert_prediction(
            transaction_id=result.transaction_id,
            features={col: getattr(txn, col) for col in _FEATURE_COLS},
            prediction=int(result.is_fraud),
            confidence=result.confidence,
            model_version=result.model_version,
        )

    return BatchOutput(predictions=results)
