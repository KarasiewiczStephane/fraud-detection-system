"""Tests for the FastAPI Fraud Detection API."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sklearn.linear_model import LogisticRegression

from src.api.app import AppState, app, get_state, set_state
from src.api.schemas import TransactionInput
from src.utils.database import DatabaseManager


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _dummy_model():
    """Train a tiny LR model on 30 features (V1-V28 + Time + Amount)."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 30))
    y = (X[:, 0] > 0).astype(int)
    m = LogisticRegression(max_iter=200)
    m.fit(X, y)
    return m


def _valid_payload(transaction_id: str = "txn_001") -> Dict[str, Any]:
    """Return a valid single-transaction JSON payload."""
    payload: Dict[str, Any] = {"transaction_id": transaction_id}
    for i in range(1, 29):
        payload[f"V{i}"] = float(i) * 0.01
    payload["Time"] = 1000.0
    payload["Amount"] = 49.99
    return payload


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest_asyncio.fixture
async def _setup_state():
    """Initialise AppState with a dummy model and in-memory DB."""
    db = DatabaseManager(":memory:")
    await db.connect()

    state = AppState(
        model=_dummy_model(),
        model_version="test_v1",
        threshold=0.5,
        db=db,
    )
    set_state(state)
    yield state
    await db.close()
    set_state(None)


@pytest_asyncio.fixture
async def client(_setup_state):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ------------------------------------------------------------------
# Health endpoint
# ------------------------------------------------------------------


class TestHealth:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_health_status_healthy(self, client):
        data = (await client.get("/health")).json()
        assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_contains_model_version(self, client):
        data = (await client.get("/health")).json()
        assert data["model_version"] == "test_v1"


# ------------------------------------------------------------------
# Input validation
# ------------------------------------------------------------------


class TestInputValidation:
    @pytest.mark.asyncio
    async def test_negative_amount_rejected(self, client):
        payload = _valid_payload()
        payload["Amount"] = -10.0
        resp = await client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_field_rejected(self, client):
        payload = _valid_payload()
        del payload["V14"]
        resp = await client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_extra_field_rejected(self, client):
        payload = _valid_payload()
        payload["extra_field"] = 999
        resp = await client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_transaction_id_rejected(self, client):
        payload = _valid_payload()
        del payload["transaction_id"]
        resp = await client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_non_numeric_v_field_rejected(self, client):
        payload = _valid_payload()
        payload["V1"] = "not_a_number"
        resp = await client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_zero_amount_accepted(self, client):
        payload = _valid_payload()
        payload["Amount"] = 0.0
        resp = await client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 200


# ------------------------------------------------------------------
# Single prediction
# ------------------------------------------------------------------


class TestSinglePrediction:
    @pytest.mark.asyncio
    async def test_returns_200(self, client):
        resp = await client.post("/api/v1/predict", json=_valid_payload())
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_response_has_required_fields(self, client):
        data = (await client.post("/api/v1/predict", json=_valid_payload())).json()
        assert "transaction_id" in data
        assert "fraud_probability" in data
        assert "is_fraud" in data
        assert "confidence" in data
        assert "model_version" in data

    @pytest.mark.asyncio
    async def test_transaction_id_matches(self, client):
        data = (await client.post("/api/v1/predict", json=_valid_payload("abc"))).json()
        assert data["transaction_id"] == "abc"

    @pytest.mark.asyncio
    async def test_fraud_probability_in_range(self, client):
        data = (await client.post("/api/v1/predict", json=_valid_payload())).json()
        assert 0.0 <= data["fraud_probability"] <= 1.0

    @pytest.mark.asyncio
    async def test_confidence_in_range(self, client):
        data = (await client.post("/api/v1/predict", json=_valid_payload())).json()
        assert 0.0 <= data["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_model_version_returned(self, client):
        data = (await client.post("/api/v1/predict", json=_valid_payload())).json()
        assert data["model_version"] == "test_v1"

    @pytest.mark.asyncio
    async def test_is_fraud_is_bool(self, client):
        data = (await client.post("/api/v1/predict", json=_valid_payload())).json()
        assert isinstance(data["is_fraud"], bool)


# ------------------------------------------------------------------
# Batch prediction
# ------------------------------------------------------------------


class TestBatchPrediction:
    @pytest.mark.asyncio
    async def test_batch_returns_200(self, client):
        txns = [_valid_payload(f"txn_{i}") for i in range(3)]
        resp = await client.post("/api/v1/predict/batch", json={"transactions": txns})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_batch_returns_correct_count(self, client):
        txns = [_valid_payload(f"txn_{i}") for i in range(5)]
        data = (
            await client.post("/api/v1/predict/batch", json={"transactions": txns})
        ).json()
        assert len(data["predictions"]) == 5

    @pytest.mark.asyncio
    async def test_batch_transaction_ids_match(self, client):
        txns = [_valid_payload(f"txn_{i}") for i in range(3)]
        data = (
            await client.post("/api/v1/predict/batch", json={"transactions": txns})
        ).json()
        ids = [p["transaction_id"] for p in data["predictions"]]
        assert ids == ["txn_0", "txn_1", "txn_2"]

    @pytest.mark.asyncio
    async def test_batch_max_100(self, client):
        txns = [_valid_payload(f"txn_{i}") for i in range(101)]
        resp = await client.post("/api/v1/predict/batch", json={"transactions": txns})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_batch_100_accepted(self, client):
        txns = [_valid_payload(f"txn_{i}") for i in range(100)]
        resp = await client.post("/api/v1/predict/batch", json={"transactions": txns})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_batch_empty_accepted(self, client):
        resp = await client.post("/api/v1/predict/batch", json={"transactions": []})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_batch_invalid_transaction_rejected(self, client):
        txns = [_valid_payload("ok"), {"transaction_id": "bad"}]  # missing fields
        resp = await client.post("/api/v1/predict/batch", json={"transactions": txns})
        assert resp.status_code == 422


# ------------------------------------------------------------------
# Timing middleware
# ------------------------------------------------------------------


class TestTimingMiddleware:
    @pytest.mark.asyncio
    async def test_response_has_timing_header(self, client):
        resp = await client.get("/health")
        assert "X-Process-Time-Ms" in resp.headers

    @pytest.mark.asyncio
    async def test_timing_header_is_numeric(self, client):
        resp = await client.get("/health")
        elapsed = float(resp.headers["X-Process-Time-Ms"])
        assert elapsed >= 0

    @pytest.mark.asyncio
    async def test_response_time_reasonable(self, client):
        resp = await client.post("/api/v1/predict", json=_valid_payload())
        elapsed = float(resp.headers["X-Process-Time-Ms"])
        assert elapsed < 1000  # under 1 second


# ------------------------------------------------------------------
# Prediction logging to SQLite
# ------------------------------------------------------------------


class TestPredictionLogging:
    @pytest.mark.asyncio
    async def test_prediction_logged(self, client):
        await client.post("/api/v1/predict", json=_valid_payload("log_test"))
        state = get_state()
        row = await state.db.get_prediction("log_test")
        assert row is not None
        assert row["transaction_id"] == "log_test"

    @pytest.mark.asyncio
    async def test_logged_prediction_value(self, client):
        await client.post("/api/v1/predict", json=_valid_payload("log_val"))
        state = get_state()
        row = await state.db.get_prediction("log_val")
        assert row["prediction"] in (0, 1)

    @pytest.mark.asyncio
    async def test_logged_confidence(self, client):
        await client.post("/api/v1/predict", json=_valid_payload("log_conf"))
        state = get_state()
        row = await state.db.get_prediction("log_conf")
        assert 0.0 <= row["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_logged_model_version(self, client):
        await client.post("/api/v1/predict", json=_valid_payload("log_mv"))
        state = get_state()
        row = await state.db.get_prediction("log_mv")
        assert row["model_version"] == "test_v1"

    @pytest.mark.asyncio
    async def test_batch_predictions_logged(self, client):
        txns = [_valid_payload(f"batch_log_{i}") for i in range(3)]
        await client.post("/api/v1/predict/batch", json={"transactions": txns})
        state = get_state()
        recent = await state.db.get_recent_predictions(limit=10)
        logged_ids = {r["transaction_id"] for r in recent}
        for i in range(3):
            assert f"batch_log_{i}" in logged_ids

    @pytest.mark.asyncio
    async def test_logged_features_json(self, client):
        await client.post("/api/v1/predict", json=_valid_payload("log_feat"))
        state = get_state()
        row = await state.db.get_prediction("log_feat")
        assert isinstance(row["features"], dict)
        assert "Amount" in row["features"]


# ------------------------------------------------------------------
# Schema unit tests
# ------------------------------------------------------------------


class TestSchemas:
    def test_transaction_input_valid(self):
        payload = _valid_payload()
        txn = TransactionInput(**payload)
        assert txn.transaction_id == "txn_001"

    def test_transaction_input_rejects_extra(self):
        payload = _valid_payload()
        payload["bad_field"] = 1
        with pytest.raises(Exception):
            TransactionInput(**payload)

    def test_transaction_input_rejects_negative_amount(self):
        payload = _valid_payload()
        payload["Amount"] = -1.0
        with pytest.raises(Exception):
            TransactionInput(**payload)
