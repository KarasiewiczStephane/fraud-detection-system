"""A/B test results dashboard page."""

from __future__ import annotations

import sqlite3

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.data import get_ab_results, get_ab_summary


def render(conn: sqlite3.Connection) -> None:
    st.title("A/B Test Results")

    summary = get_ab_summary(conn)
    if not summary:
        st.info("No A/B test results recorded yet.")
        return

    # Side-by-side model comparison
    cols = st.columns(len(summary))
    for col, (model_name, stats) in zip(cols, summary.items()):
        col.subheader(f"Model {model_name}")
        col.metric("Predictions", stats["count"])
        col.metric("Fraud Rate", f"{stats['fraud_rate']:.4%}")
        col.metric("Mean Latency", f"{stats['mean_latency_ms']:.1f} ms")

    st.divider()

    # Bar chart comparison
    df = get_ab_results(conn)
    if df.empty:
        return

    model_fraud = df.groupby("model_name")["prediction"].mean().reset_index()
    model_fraud.columns = ["Model", "Fraud Rate"]
    fig = px.bar(
        model_fraud,
        x="Model",
        y="Fraud Rate",
        title="Fraud Detection Rate by Model",
        color="Model",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Latency comparison
    model_latency = df.groupby("model_name")["latency_ms"].mean().reset_index()
    model_latency.columns = ["Model", "Mean Latency (ms)"]
    fig2 = px.bar(
        model_latency,
        x="Model",
        y="Mean Latency (ms)",
        title="Mean Latency by Model",
        color="Model",
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Statistical significance indicator
    st.subheader("Statistical Significance")
    if len(summary) >= 2:
        models = list(summary.keys())
        rate_a = summary[models[0]]["fraud_rate"]
        rate_b = summary[models[1]]["fraud_rate"]
        diff = abs(rate_a - rate_b)
        st.write(f"Difference in fraud rates: **{diff:.4%}**")
        total = sum(s["count"] for s in summary.values())
        if total < 1000:
            st.warning(
                f"Only {total} samples collected. "
                "Need at least 1,000 for reliable significance testing."
            )
        else:
            st.success("Sufficient samples for significance testing.")
