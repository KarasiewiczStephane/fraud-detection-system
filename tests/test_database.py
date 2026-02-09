"""Tests for src/utils/database.py."""

import pytest
import pytest_asyncio

from src.utils.database import DatabaseManager


@pytest_asyncio.fixture
async def db():
    """Provide an in-memory database for each test."""
    manager = DatabaseManager(":memory:")
    await manager.connect()
    yield manager
    await manager.close()


# ------------------------------------------------------------------
# Connection lifecycle
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connect_creates_tables(db: DatabaseManager):
    cursor = await db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = {row["name"] for row in await cursor.fetchall()}
    assert "predictions" in tables
    assert "ab_test_results" in tables


@pytest.mark.asyncio
async def test_conn_property_raises_when_not_connected():
    manager = DatabaseManager(":memory:")
    with pytest.raises(RuntimeError, match="not connected"):
        _ = manager.conn


# ------------------------------------------------------------------
# Predictions CRUD
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_insert_and_get_prediction(db: DatabaseManager):
    features = {"V1": 1.0, "V2": -0.5, "Amount": 100.0}
    shap = {"V1": 0.3, "V2": -0.1}

    row_id = await db.insert_prediction(
        transaction_id="txn_001",
        features=features,
        prediction=1,
        confidence=0.95,
        model_version="xgboost_v1",
        shap_values=shap,
    )
    assert isinstance(row_id, int)
    assert row_id >= 1

    record = await db.get_prediction("txn_001")
    assert record is not None
    assert record["transaction_id"] == "txn_001"
    assert record["prediction"] == 1
    assert record["confidence"] == 0.95
    assert record["model_version"] == "xgboost_v1"
    assert record["features"] == features
    assert record["shap_values"] == shap
    assert "timestamp" in record


@pytest.mark.asyncio
async def test_get_prediction_not_found(db: DatabaseManager):
    result = await db.get_prediction("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_insert_prediction_without_shap(db: DatabaseManager):
    row_id = await db.insert_prediction(
        transaction_id="txn_002",
        features={"V1": 0.5},
        prediction=0,
        confidence=0.1,
        model_version="lr_v1",
    )
    record = await db.get_prediction("txn_002")
    assert record is not None
    assert record["shap_values"] is None


@pytest.mark.asyncio
async def test_get_recent_predictions(db: DatabaseManager):
    for i in range(5):
        await db.insert_prediction(
            transaction_id=f"txn_{i:03d}",
            features={"V1": float(i)},
            prediction=0,
            confidence=0.1,
            model_version="v1",
        )

    recent = await db.get_recent_predictions(limit=3)
    assert len(recent) == 3
    # Most recent first (highest id)
    assert recent[0]["transaction_id"] == "txn_004"


# ------------------------------------------------------------------
# A/B test results CRUD
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_insert_and_get_ab_result(db: DatabaseManager):
    row_id = await db.insert_ab_result(
        model_name="model_A",
        transaction_id="txn_100",
        prediction=1,
        actual=1,
        latency_ms=12.5,
    )
    assert isinstance(row_id, int)

    results = await db.get_ab_results(model_name="model_A")
    assert len(results) == 1
    assert results[0]["model_name"] == "model_A"
    assert results[0]["transaction_id"] == "txn_100"
    assert results[0]["prediction"] == 1
    assert results[0]["actual"] == 1
    assert results[0]["latency_ms"] == 12.5


@pytest.mark.asyncio
async def test_insert_ab_result_without_actual(db: DatabaseManager):
    await db.insert_ab_result(
        model_name="model_B",
        transaction_id="txn_101",
        prediction=0,
        actual=None,
        latency_ms=8.0,
    )
    results = await db.get_ab_results(model_name="model_B")
    assert results[0]["actual"] is None


@pytest.mark.asyncio
async def test_get_ab_results_all(db: DatabaseManager):
    await db.insert_ab_result("model_A", "txn_1", 0, 0, 5.0)
    await db.insert_ab_result("model_B", "txn_2", 1, 1, 6.0)

    all_results = await db.get_ab_results()
    assert len(all_results) == 2


@pytest.mark.asyncio
async def test_get_ab_results_filtered(db: DatabaseManager):
    await db.insert_ab_result("model_A", "txn_1", 0, 0, 5.0)
    await db.insert_ab_result("model_B", "txn_2", 1, 1, 6.0)

    a_results = await db.get_ab_results(model_name="model_A")
    assert len(a_results) == 1
    assert a_results[0]["model_name"] == "model_A"


# ------------------------------------------------------------------
# Close and reconnect
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_and_reopen():
    manager = DatabaseManager(":memory:")
    await manager.connect()
    await manager.insert_prediction("txn_x", {"V1": 1}, 0, 0.5, "v1")
    await manager.close()

    # After close, conn should raise
    with pytest.raises(RuntimeError):
        _ = manager.conn
