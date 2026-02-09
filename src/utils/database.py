"""Async SQLite database manager for prediction and A/B test storage."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

_PREDICTIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id TEXT NOT NULL,
    features TEXT NOT NULL,
    prediction INTEGER NOT NULL,
    confidence REAL NOT NULL,
    model_version TEXT NOT NULL,
    shap_values TEXT,
    timestamp TEXT NOT NULL
);
"""

_AB_TEST_SCHEMA = """
CREATE TABLE IF NOT EXISTS ab_test_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    transaction_id TEXT NOT NULL,
    prediction INTEGER NOT NULL,
    actual INTEGER,
    latency_ms REAL NOT NULL,
    timestamp TEXT NOT NULL
);
"""


class DatabaseManager:
    """Async SQLite connection manager.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Use ``":memory:"`` for an
        in-memory database (useful for testing).
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """Open the database connection and create tables if needed."""
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute(_PREDICTIONS_SCHEMA)
        await self._conn.execute(_AB_TEST_SCHEMA)
        await self._conn.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._conn

    # ------------------------------------------------------------------
    # Predictions CRUD
    # ------------------------------------------------------------------

    async def insert_prediction(
        self,
        transaction_id: str,
        features: dict[str, Any],
        prediction: int,
        confidence: float,
        model_version: str,
        shap_values: dict[str, Any] | None = None,
    ) -> int:
        """Insert a prediction record and return its row id."""
        cursor = await self.conn.execute(
            """INSERT INTO predictions
               (transaction_id, features, prediction, confidence,
                model_version, shap_values, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                transaction_id,
                json.dumps(features),
                prediction,
                confidence,
                model_version,
                json.dumps(shap_values) if shap_values else None,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        await self.conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    async def get_prediction(self, transaction_id: str) -> dict[str, Any] | None:
        """Return a single prediction by *transaction_id*."""
        cursor = await self.conn.execute(
            "SELECT * FROM predictions WHERE transaction_id = ?",
            (transaction_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    async def get_recent_predictions(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent predictions."""
        cursor = await self.conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ------------------------------------------------------------------
    # A/B test results CRUD
    # ------------------------------------------------------------------

    async def insert_ab_result(
        self,
        model_name: str,
        transaction_id: str,
        prediction: int,
        actual: int | None,
        latency_ms: float,
    ) -> int:
        """Insert an A/B test result and return its row id."""
        cursor = await self.conn.execute(
            """INSERT INTO ab_test_results
               (model_name, transaction_id, prediction, actual,
                latency_ms, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                model_name,
                transaction_id,
                prediction,
                actual,
                latency_ms,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        await self.conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    async def get_ab_results(
        self, model_name: str | None = None
    ) -> list[dict[str, Any]]:
        """Return A/B test results, optionally filtered by model."""
        if model_name:
            cursor = await self.conn.execute(
                "SELECT * FROM ab_test_results WHERE model_name = ? ORDER BY id DESC",
                (model_name,),
            )
        else:
            cursor = await self.conn.execute(
                "SELECT * FROM ab_test_results ORDER BY id DESC"
            )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_dict(row: aiosqlite.Row) -> dict[str, Any]:
        d = dict(row)
        if "features" in d and d["features"]:
            d["features"] = json.loads(d["features"])
        if "shap_values" in d and d["shap_values"]:
            d["shap_values"] = json.loads(d["shap_values"])
        return d
