"""Synchronous data access layer for the Streamlit dashboard.

Uses plain ``sqlite3`` (not aiosqlite) so it can be called directly from
Streamlit without an event-loop.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd


# ------------------------------------------------------------------
# Connection helper
# ------------------------------------------------------------------


def get_connection(db_path: str | Path = "data/predictions.db") -> sqlite3.Connection:
    """Return a ``sqlite3.Connection`` with row-factory enabled."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


# ------------------------------------------------------------------
# Predictions
# ------------------------------------------------------------------


def get_recent_predictions(
    conn: sqlite3.Connection,
    limit: int = 50,
) -> pd.DataFrame:
    """Return the most recent predictions as a DataFrame."""
    cursor = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    )
    rows = [dict(r) for r in cursor.fetchall()]
    if not rows:
        return pd.DataFrame(
            columns=[
                "id",
                "transaction_id",
                "features",
                "prediction",
                "confidence",
                "model_version",
                "shap_values",
                "timestamp",
            ]
        )
    df = pd.DataFrame(rows)
    # Parse JSON columns
    for col in ("features", "shap_values"):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: json.loads(v) if isinstance(v, str) else v
            )
    return df


def get_predictions_since(
    conn: sqlite3.Connection,
    since: datetime,
) -> pd.DataFrame:
    """Return predictions with ``timestamp >= since``."""
    cursor = conn.execute(
        "SELECT * FROM predictions WHERE timestamp >= ? ORDER BY id DESC",
        (since.isoformat(),),
    )
    rows = [dict(r) for r in cursor.fetchall()]
    if not rows:
        return pd.DataFrame(
            columns=[
                "id",
                "transaction_id",
                "features",
                "prediction",
                "confidence",
                "model_version",
                "shap_values",
                "timestamp",
            ]
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Overview metrics
# ------------------------------------------------------------------


def compute_overview_metrics(
    conn: sqlite3.Connection,
) -> Dict[str, Any]:
    """Compute high-level metrics for the overview page."""
    now = datetime.now(timezone.utc)

    def _count_since(dt: datetime) -> int:
        cur = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE timestamp >= ?",
            (dt.isoformat(),),
        )
        return cur.fetchone()[0]

    total_1h = _count_since(now - timedelta(hours=1))
    total_24h = _count_since(now - timedelta(hours=24))
    total_7d = _count_since(now - timedelta(days=7))

    # Overall fraud rate
    cur = conn.execute(
        "SELECT COUNT(*) as total, SUM(prediction) as fraud FROM predictions"
    )
    row = cur.fetchone()
    total = row[0] or 0
    fraud = row[1] or 0
    fraud_rate = fraud / total if total > 0 else 0.0

    return {
        "total_1h": total_1h,
        "total_24h": total_24h,
        "total_7d": total_7d,
        "total_all": total,
        "fraud_count": fraud,
        "fraud_rate": round(fraud_rate, 6),
    }


# ------------------------------------------------------------------
# A/B test results
# ------------------------------------------------------------------


def get_ab_results(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return all A/B test results as a DataFrame."""
    cursor = conn.execute("SELECT * FROM ab_test_results ORDER BY id DESC")
    rows = [dict(r) for r in cursor.fetchall()]
    if not rows:
        return pd.DataFrame(
            columns=[
                "id",
                "model_name",
                "transaction_id",
                "prediction",
                "actual",
                "latency_ms",
                "timestamp",
            ]
        )
    return pd.DataFrame(rows)


def get_ab_summary(conn: sqlite3.Connection) -> Dict[str, Dict[str, Any]]:
    """Per-model summary of A/B test results."""
    df = get_ab_results(conn)
    if df.empty:
        return {}
    summary: Dict[str, Dict[str, Any]] = {}
    for model_name, group in df.groupby("model_name"):
        summary[str(model_name)] = {
            "count": len(group),
            "fraud_rate": round(float(group["prediction"].mean()), 6),
            "mean_latency_ms": round(float(group["latency_ms"].mean()), 2),
        }
    return summary


# ------------------------------------------------------------------
# Alerts
# ------------------------------------------------------------------


def get_high_confidence_alerts(
    conn: sqlite3.Connection,
    threshold: float = 0.9,
    limit: int = 200,
) -> pd.DataFrame:
    """Return predictions where ``confidence >= threshold`` AND fraud."""
    cursor = conn.execute(
        """SELECT * FROM predictions
           WHERE prediction = 1 AND confidence >= ?
           ORDER BY id DESC LIMIT ?""",
        (threshold, limit),
    )
    rows = [dict(r) for r in cursor.fetchall()]
    if not rows:
        return pd.DataFrame(
            columns=[
                "id",
                "transaction_id",
                "features",
                "prediction",
                "confidence",
                "model_version",
                "shap_values",
                "timestamp",
            ]
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Date filtering helper
# ------------------------------------------------------------------


def filter_by_date_range(
    df: pd.DataFrame,
    start: datetime,
    end: datetime,
    col: str = "timestamp",
) -> pd.DataFrame:
    """Filter a DataFrame to rows where ``col`` falls in ``[start, end]``."""
    if df.empty or col not in df.columns:
        return df
    ts = pd.to_datetime(df[col])
    mask = (ts >= pd.Timestamp(start)) & (ts <= pd.Timestamp(end))
    return df.loc[mask].reset_index(drop=True)


# ------------------------------------------------------------------
# Color coding for fraud probability
# ------------------------------------------------------------------


def fraud_color(confidence: float, is_fraud: bool) -> str:
    """Return a CSS color string based on the effective fraud probability.

    - Green  (< 0.3)
    - Yellow (0.3 – 0.7)
    - Red    (> 0.7)
    """
    # Reconstruct fraud probability from stored confidence + is_fraud flag
    prob = confidence if is_fraud else 1.0 - confidence
    if prob > 0.7:
        return "red"
    if prob >= 0.3:
        return "orange"
    return "green"


# ------------------------------------------------------------------
# CSV export
# ------------------------------------------------------------------


def dataframe_to_csv(df: pd.DataFrame) -> str:
    """Serialize a DataFrame to CSV text (no index)."""
    return df.to_csv(index=False)
