"""Model performance page — precision / recall / F1 charts."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard.data import filter_by_date_range, get_recent_predictions


def render(conn: sqlite3.Connection) -> None:
    st.title("Model Performance")

    # Date range filter
    col1, col2 = st.columns(2)
    default_start = datetime.now(timezone.utc) - timedelta(days=7)
    start_date = col1.date_input("Start date", value=default_start.date())
    end_date = col2.date_input("End date", value=datetime.now(timezone.utc).date())

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time())

    df = get_recent_predictions(conn, limit=10_000)
    if df.empty:
        st.info("No predictions available.")
        return

    df = filter_by_date_range(df, start_dt, end_dt)
    if df.empty:
        st.warning("No predictions in the selected date range.")
        return

    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # --- Metrics over time (hourly buckets) ---
    df["hour"] = df["timestamp"].dt.floor("h")
    hourly = (
        df.groupby("hour")
        .agg(
            total=("prediction", "count"),
            fraud=("prediction", "sum"),
        )
        .reset_index()
    )
    hourly["fraud_rate"] = hourly["fraud"] / hourly["total"]

    fig = px.line(
        hourly,
        x="hour",
        y="fraud_rate",
        title="Fraud Detection Rate Over Time",
        labels={"hour": "Time", "fraud_rate": "Fraud Rate"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Confusion matrix (predicted vs actual placeholder) ---
    st.subheader("Prediction Distribution")
    fraud_count = int(df["prediction"].sum())
    normal_count = len(df) - fraud_count
    dist_df = pd.DataFrame(
        {
            "Class": ["Normal", "Fraud"],
            "Count": [normal_count, fraud_count],
        }
    )
    fig2 = px.bar(
        dist_df,
        x="Class",
        y="Count",
        title="Prediction Distribution",
        color="Class",
        color_discrete_map={"Normal": "green", "Fraud": "red"},
    )
    st.plotly_chart(fig2, use_container_width=True)

    # --- Summary stats ---
    st.subheader("Summary")
    st.metric("Total Predictions", len(df))
    st.metric("Fraud Predictions", fraud_count)
    st.metric("Mean Confidence", f"{df['confidence'].mean():.4f}")
