"""Real-time prediction feed page."""

from __future__ import annotations

import sqlite3

import streamlit as st

from src.dashboard.data import fraud_color, get_recent_predictions


def render(conn: sqlite3.Connection) -> None:
    st.title("Real-time Prediction Feed")

    df = get_recent_predictions(conn, limit=50)

    if df.empty:
        st.info("No predictions recorded yet.")
        return

    # Build display table
    display = df[["transaction_id", "prediction", "confidence", "model_version", "timestamp"]].copy()
    display.rename(
        columns={
            "transaction_id": "Transaction ID",
            "prediction": "Is Fraud",
            "confidence": "Confidence",
            "model_version": "Model",
            "timestamp": "Timestamp",
        },
        inplace=True,
    )

    # Color-code rows
    def _highlight(row):
        is_fraud = bool(row["Is Fraud"])
        conf = float(row["Confidence"])
        color = fraud_color(conf, is_fraud)
        css = f"background-color: {color}; color: white"
        return [css] * len(row)

    styled = display.style.apply(_highlight, axis=1)
    st.dataframe(styled, use_container_width=True)

    # Expandable details
    st.subheader("Transaction Details")
    selected = st.selectbox(
        "Select a transaction",
        df["transaction_id"].tolist(),
    )
    if selected is not None:
        row = df[df["transaction_id"] == selected].iloc[0]
        with st.expander("Details", expanded=True):
            st.json({
                "transaction_id": row["transaction_id"],
                "prediction": int(row["prediction"]),
                "confidence": float(row["confidence"]),
                "model_version": row["model_version"],
                "timestamp": row["timestamp"],
                "features": row.get("features"),
            })
