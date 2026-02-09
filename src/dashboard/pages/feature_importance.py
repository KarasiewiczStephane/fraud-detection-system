"""Feature importance dashboard page — SHAP summary visualization."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import plotly.express as px
import streamlit as st


def render(conn: sqlite3.Connection) -> None:
    st.title("Feature Importance")

    # Try to load pre-generated SHAP plots
    plots_dir = Path("reports")
    beeswarm = plots_dir / "shap_beeswarm.png"
    bar_plot = plots_dir / "shap_bar.png"

    if beeswarm.exists():
        st.subheader("SHAP Beeswarm Summary")
        st.image(str(beeswarm), use_container_width=True)
    else:
        st.info(
            "No beeswarm plot found. Run the explainability pipeline to "
            "generate SHAP plots in the `reports/` directory."
        )

    if bar_plot.exists():
        st.subheader("Feature Importance (Bar)")
        st.image(str(bar_plot), use_container_width=True)

    # Interactive feature selector from stored SHAP values
    st.divider()
    st.subheader("Per-prediction SHAP Values")

    cursor = conn.execute(
        "SELECT transaction_id, shap_values FROM predictions "
        "WHERE shap_values IS NOT NULL ORDER BY id DESC LIMIT 50"
    )
    rows = cursor.fetchall()

    if not rows:
        st.info("No predictions with SHAP values stored yet.")
        return

    import json

    txn_ids = [r[0] for r in rows]
    selected = st.selectbox("Select transaction", txn_ids)

    for r in rows:
        if r[0] == selected:
            sv = json.loads(r[1]) if isinstance(r[1], str) else r[1]
            if isinstance(sv, dict):
                import pandas as pd

                sv_df = pd.DataFrame(
                    list(sv.items()), columns=["Feature", "SHAP Value"]
                )
                sv_df = sv_df.sort_values("SHAP Value", key=abs, ascending=False)
                fig = px.bar(
                    sv_df.head(15),
                    x="SHAP Value",
                    y="Feature",
                    orientation="h",
                    title=f"Top SHAP Values — {selected}",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.json(sv)
            break
