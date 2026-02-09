"""Alert log page — high-confidence fraud detections with CSV export."""

from __future__ import annotations

import sqlite3

import streamlit as st

from src.dashboard.data import dataframe_to_csv, get_high_confidence_alerts


def render(conn: sqlite3.Connection) -> None:
    st.title("Alert Log")
    st.write("High-confidence fraud detections (probability > threshold).")

    threshold = st.slider(
        "Confidence threshold",
        min_value=0.5,
        max_value=1.0,
        value=0.9,
        step=0.05,
    )

    df = get_high_confidence_alerts(conn, threshold=threshold)

    if df.empty:
        st.info("No high-confidence fraud alerts found.")
        return

    st.metric("Alerts", len(df))

    # Display table (exclude raw JSON columns for readability)
    display_cols = [
        c for c in df.columns if c not in ("features", "shap_values")
    ]
    st.dataframe(df[display_cols], use_container_width=True)

    # CSV export
    csv = dataframe_to_csv(df[display_cols])
    st.download_button(
        "Download CSV",
        csv,
        "fraud_alerts.csv",
        "text/csv",
    )
