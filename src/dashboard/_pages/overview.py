"""Overview page — key metrics at a glance."""

from __future__ import annotations

import sqlite3

import streamlit as st

from src.dashboard.data import compute_overview_metrics


def render(conn: sqlite3.Connection) -> None:
    st.title("Overview")

    metrics = compute_overview_metrics(conn)

    # Big number cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transactions (1 h)", metrics["total_1h"])
    col2.metric("Transactions (24 h)", metrics["total_24h"])
    col3.metric("Transactions (7 d)", metrics["total_7d"])
    col4.metric("Total Transactions", metrics["total_all"])

    st.divider()

    col_a, col_b = st.columns(2)
    col_a.metric("Fraud Detections", metrics["fraud_count"])
    col_b.metric("Fraud Rate", f"{metrics['fraud_rate']:.4%}")
