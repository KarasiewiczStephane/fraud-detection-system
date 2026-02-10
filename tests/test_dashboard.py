"""Tests for the Streamlit dashboard data layer and helpers."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.dashboard.data import (
    compute_overview_metrics,
    dataframe_to_csv,
    filter_by_date_range,
    fraud_color,
    get_ab_results,
    get_ab_summary,
    get_high_confidence_alerts,
    get_recent_predictions,
    get_predictions_since,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

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


def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(_PREDICTIONS_SCHEMA)
    conn.execute(_AB_TEST_SCHEMA)
    conn.commit()
    return conn


def _insert_prediction(
    conn: sqlite3.Connection,
    txn_id: str = "txn_001",
    prediction: int = 0,
    confidence: float = 0.85,
    model_version: str = "v1",
    timestamp: str | None = None,
    features: dict | None = None,
    shap_values: dict | None = None,
) -> None:
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()
    if features is None:
        features = {"V1": 0.1, "Amount": 50.0}
    conn.execute(
        """INSERT INTO predictions
           (transaction_id, features, prediction, confidence,
            model_version, shap_values, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            txn_id,
            json.dumps(features),
            prediction,
            confidence,
            model_version,
            json.dumps(shap_values) if shap_values else None,
            timestamp,
        ),
    )
    conn.commit()


def _insert_ab(
    conn: sqlite3.Connection,
    model_name: str = "A",
    txn_id: str = "txn_001",
    prediction: int = 0,
    actual: int | None = None,
    latency_ms: float = 5.0,
    timestamp: str | None = None,
) -> None:
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO ab_test_results
           (model_name, transaction_id, prediction, actual,
            latency_ms, timestamp)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (model_name, txn_id, prediction, actual, latency_ms, timestamp),
    )
    conn.commit()


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def conn():
    return _make_db()


@pytest.fixture
def seeded_conn(conn):
    """DB with a handful of predictions and AB results."""
    now = datetime.now(timezone.utc)
    for i in range(10):
        _insert_prediction(
            conn,
            txn_id=f"txn_{i:03d}",
            prediction=1 if i < 3 else 0,
            confidence=0.95 if i < 3 else 0.2,
            timestamp=(now - timedelta(minutes=i * 5)).isoformat(),
        )
    for i in range(6):
        _insert_ab(
            conn,
            model_name="A" if i < 3 else "B",
            txn_id=f"ab_{i}",
            prediction=1 if i % 2 == 0 else 0,
            latency_ms=float(5 + i),
            timestamp=(now - timedelta(minutes=i)).isoformat(),
        )
    return conn


# ------------------------------------------------------------------
# get_recent_predictions
# ------------------------------------------------------------------


class TestGetRecentPredictions:
    def test_returns_dataframe(self, seeded_conn):
        df = get_recent_predictions(seeded_conn)
        assert isinstance(df, pd.DataFrame)

    def test_limit(self, seeded_conn):
        df = get_recent_predictions(seeded_conn, limit=3)
        assert len(df) == 3

    def test_empty_db(self, conn):
        df = get_recent_predictions(conn)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_has_expected_columns(self, seeded_conn):
        df = get_recent_predictions(seeded_conn)
        for col in ("transaction_id", "prediction", "confidence", "timestamp"):
            assert col in df.columns

    def test_features_parsed_as_dict(self, seeded_conn):
        df = get_recent_predictions(seeded_conn)
        assert isinstance(df.iloc[0]["features"], dict)

    def test_ordered_most_recent_first(self, seeded_conn):
        df = get_recent_predictions(seeded_conn)
        ids = df["id"].tolist()
        assert ids == sorted(ids, reverse=True)


# ------------------------------------------------------------------
# get_predictions_since
# ------------------------------------------------------------------


class TestGetPredictionsSince:
    def test_returns_subset(self, seeded_conn):
        since = datetime.now(timezone.utc) - timedelta(minutes=20)
        df = get_predictions_since(seeded_conn, since)
        assert len(df) <= 10

    def test_empty_when_future(self, seeded_conn):
        future = datetime.now(timezone.utc) + timedelta(days=1)
        df = get_predictions_since(seeded_conn, future)
        assert df.empty


# ------------------------------------------------------------------
# compute_overview_metrics
# ------------------------------------------------------------------


class TestOverviewMetrics:
    def test_returns_dict(self, seeded_conn):
        metrics = compute_overview_metrics(seeded_conn)
        assert isinstance(metrics, dict)

    def test_contains_keys(self, seeded_conn):
        metrics = compute_overview_metrics(seeded_conn)
        for key in (
            "total_1h",
            "total_24h",
            "total_7d",
            "total_all",
            "fraud_count",
            "fraud_rate",
        ):
            assert key in metrics

    def test_total_all(self, seeded_conn):
        metrics = compute_overview_metrics(seeded_conn)
        assert metrics["total_all"] == 10

    def test_fraud_count(self, seeded_conn):
        metrics = compute_overview_metrics(seeded_conn)
        assert metrics["fraud_count"] == 3

    def test_fraud_rate(self, seeded_conn):
        metrics = compute_overview_metrics(seeded_conn)
        assert metrics["fraud_rate"] == pytest.approx(0.3, abs=1e-6)

    def test_empty_db(self, conn):
        metrics = compute_overview_metrics(conn)
        assert metrics["total_all"] == 0
        assert metrics["fraud_rate"] == 0.0


# ------------------------------------------------------------------
# A/B results
# ------------------------------------------------------------------


class TestABResults:
    def test_get_ab_results_dataframe(self, seeded_conn):
        df = get_ab_results(seeded_conn)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 6

    def test_get_ab_results_empty(self, conn):
        df = get_ab_results(conn)
        assert df.empty

    def test_get_ab_summary(self, seeded_conn):
        summary = get_ab_summary(seeded_conn)
        assert "A" in summary
        assert "B" in summary

    def test_ab_summary_count(self, seeded_conn):
        summary = get_ab_summary(seeded_conn)
        assert summary["A"]["count"] == 3
        assert summary["B"]["count"] == 3

    def test_ab_summary_keys(self, seeded_conn):
        summary = get_ab_summary(seeded_conn)
        for model in ("A", "B"):
            assert "count" in summary[model]
            assert "fraud_rate" in summary[model]
            assert "mean_latency_ms" in summary[model]

    def test_ab_summary_empty(self, conn):
        summary = get_ab_summary(conn)
        assert summary == {}


# ------------------------------------------------------------------
# Alerts
# ------------------------------------------------------------------


class TestAlerts:
    def test_returns_dataframe(self, seeded_conn):
        df = get_high_confidence_alerts(seeded_conn, threshold=0.9)
        assert isinstance(df, pd.DataFrame)

    def test_only_fraud_predictions(self, seeded_conn):
        df = get_high_confidence_alerts(seeded_conn, threshold=0.9)
        if not df.empty:
            assert (df["prediction"] == 1).all()

    def test_confidence_above_threshold(self, seeded_conn):
        df = get_high_confidence_alerts(seeded_conn, threshold=0.9)
        if not df.empty:
            assert (df["confidence"] >= 0.9).all()

    def test_returns_3_alerts(self, seeded_conn):
        # We inserted 3 fraud predictions with confidence=0.95
        df = get_high_confidence_alerts(seeded_conn, threshold=0.9)
        assert len(df) == 3

    def test_lower_threshold_returns_more(self, seeded_conn):
        high = get_high_confidence_alerts(seeded_conn, threshold=0.95)
        low = get_high_confidence_alerts(seeded_conn, threshold=0.5)
        assert len(low) >= len(high)

    def test_empty_db(self, conn):
        df = get_high_confidence_alerts(conn)
        assert df.empty


# ------------------------------------------------------------------
# Date filtering
# ------------------------------------------------------------------


class TestDateFilter:
    def test_filters_correctly(self):
        df = pd.DataFrame(
            {
                "timestamp": [
                    "2024-01-01T10:00:00",
                    "2024-01-02T10:00:00",
                    "2024-01-03T10:00:00",
                ],
                "value": [1, 2, 3],
            }
        )
        start = datetime(2024, 1, 1, 12, 0, 0)
        end = datetime(2024, 1, 3, 0, 0, 0)
        result = filter_by_date_range(df, start, end)
        assert len(result) == 1
        assert result.iloc[0]["value"] == 2

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["timestamp", "value"])
        result = filter_by_date_range(
            df,
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
        )
        assert result.empty

    def test_all_in_range(self):
        df = pd.DataFrame(
            {
                "timestamp": [
                    "2024-06-15T12:00:00",
                    "2024-06-15T13:00:00",
                ],
                "value": [10, 20],
            }
        )
        start = datetime(2024, 6, 1)
        end = datetime(2024, 6, 30, 23, 59, 59)
        result = filter_by_date_range(df, start, end)
        assert len(result) == 2

    def test_none_in_range(self):
        df = pd.DataFrame(
            {
                "timestamp": ["2023-01-01T00:00:00"],
                "value": [1],
            }
        )
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)
        result = filter_by_date_range(df, start, end)
        assert result.empty

    def test_missing_column(self):
        df = pd.DataFrame({"other": [1, 2]})
        result = filter_by_date_range(
            df,
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
        )
        assert len(result) == 2  # unchanged


# ------------------------------------------------------------------
# Color coding
# ------------------------------------------------------------------


class TestFraudColor:
    def test_high_prob_red(self):
        # is_fraud=True, confidence=0.95 → prob=0.95 → red
        assert fraud_color(0.95, True) == "red"

    def test_low_prob_green(self):
        # is_fraud=False, confidence=0.9 → prob=0.1 → green
        assert fraud_color(0.9, False) == "green"

    def test_medium_prob_orange(self):
        # is_fraud=True, confidence=0.5 → prob=0.5 → orange
        assert fraud_color(0.5, True) == "orange"

    def test_boundary_0_3(self):
        # prob exactly 0.3 → orange
        assert fraud_color(0.3, True) == "orange"

    def test_below_0_3_green(self):
        # is_fraud=True, confidence=0.29 → prob=0.29 → green
        assert fraud_color(0.29, True) == "green"

    def test_above_0_7_red(self):
        # is_fraud=True, confidence=0.71 → prob=0.71 → red
        assert fraud_color(0.71, True) == "red"

    def test_exact_0_7_orange(self):
        # prob exactly 0.7 → orange (>0.7 required for red)
        assert fraud_color(0.7, True) == "orange"


# ------------------------------------------------------------------
# CSV export
# ------------------------------------------------------------------


class TestCSVExport:
    def test_produces_string(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        csv = dataframe_to_csv(df)
        assert isinstance(csv, str)

    def test_no_index(self):
        df = pd.DataFrame({"a": [1]})
        csv = dataframe_to_csv(df)
        lines = csv.strip().split("\n")
        assert lines[0] == "a"
        assert lines[1] == "1"

    def test_roundtrip(self):
        df = pd.DataFrame({"x": [1.5, 2.5], "y": ["foo", "bar"]})
        csv = dataframe_to_csv(df)
        df2 = pd.read_csv(pd.io.common.StringIO(csv))
        assert list(df2.columns) == ["x", "y"]
        assert len(df2) == 2

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["a", "b"])
        csv = dataframe_to_csv(df)
        assert "a,b" in csv

    def test_valid_csv_with_special_chars(self):
        df = pd.DataFrame({"col": ["hello,world", 'with "quotes"']})
        csv = dataframe_to_csv(df)
        df2 = pd.read_csv(pd.io.common.StringIO(csv))
        assert len(df2) == 2


# ------------------------------------------------------------------
# Dashboard files exist
# ------------------------------------------------------------------


class TestDashboardFiles:
    def test_app_exists(self):
        from pathlib import Path

        assert Path("src/dashboard/app.py").exists()

    def test_data_module_exists(self):
        from pathlib import Path

        assert Path("src/dashboard/data.py").exists()

    def test_overview_page_exists(self):
        from pathlib import Path

        assert Path("src/dashboard/pages/overview.py").exists()

    def test_realtime_feed_page_exists(self):
        from pathlib import Path

        assert Path("src/dashboard/pages/realtime_feed.py").exists()

    def test_performance_page_exists(self):
        from pathlib import Path

        assert Path("src/dashboard/pages/performance.py").exists()

    def test_ab_test_page_exists(self):
        from pathlib import Path

        assert Path("src/dashboard/pages/ab_test.py").exists()

    def test_feature_importance_page_exists(self):
        from pathlib import Path

        assert Path("src/dashboard/pages/feature_importance.py").exists()

    def test_alerts_page_exists(self):
        from pathlib import Path

        assert Path("src/dashboard/pages/alerts.py").exists()
