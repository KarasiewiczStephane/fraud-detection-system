"""Feature importance dashboard page — SHAP summary visualization."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import plotly.express as px
import streamlit as st


_FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]


@st.cache_resource
def _load_model_importance():
    """Load the model and compute global feature importance."""
    import pandas as pd

    from src.utils.config import get_config
    from src.models.registry import ModelRegistry

    config = get_config()
    registry_path = Path(config.get("model", "registry_path", default="models"))
    registry = ModelRegistry(registry_path)
    model_name = config.get("model", "default_model", default="xgboost")

    model = None
    if registry.list_models():
        try:
            model, _ = registry.load(model_name)
        except FileNotFoundError:
            pass

    if model is None:
        from sklearn.ensemble import RandomForestClassifier

        csv_path = Path("data/sample/sample_transactions.csv")
        df = pd.read_csv(csv_path)
        X = df[_FEATURE_COLS]
        y = df["Class"]
        model = RandomForestClassifier(
            n_estimators=50, max_depth=8, class_weight="balanced", random_state=42
        )
        model.fit(X, y)

    # Use built-in feature_importances_ if available (tree models)
    if hasattr(model, "feature_importances_"):
        importance = dict(zip(_FEATURE_COLS, model.feature_importances_.tolist()))
    else:
        importance = None

    # Also try SHAP on sample data
    shap_importance = None
    try:
        csv_path = Path("data/sample/sample_transactions.csv")
        df = pd.read_csv(csv_path)
        X = df[_FEATURE_COLS].values[:200]  # subsample for speed
        from src.models.explainer import SHAPExplainer

        model_type = (
            "random_forest" if "RandomForest" in type(model).__name__ else "xgboost"
        )
        explainer = SHAPExplainer(model, model_type=model_type)
        shap_importance = explainer.global_importance(X, _FEATURE_COLS, top_n=30)
    except Exception:
        pass

    return importance, shap_importance


def render(conn: sqlite3.Connection) -> None:
    st.title("Feature Importance")

    # Try to load pre-generated SHAP plots
    plots_dir = Path("reports")
    beeswarm = plots_dir / "shap_beeswarm.png"
    bar_plot = plots_dir / "shap_bar.png"

    if beeswarm.exists():
        st.subheader("SHAP Beeswarm Summary")
        st.image(str(beeswarm), use_container_width=True)

    if bar_plot.exists():
        st.subheader("Feature Importance (Bar)")
        st.image(str(bar_plot), use_container_width=True)

    # Dynamic importance from model
    if not beeswarm.exists() and not bar_plot.exists():
        importance, shap_importance = _load_model_importance()

        if shap_importance:
            import pandas as pd

            st.subheader("Global Feature Importance (SHAP)")
            df_shap = pd.DataFrame(
                list(shap_importance.items()),
                columns=["Feature", "Mean |SHAP Value|"],
            ).sort_values("Mean |SHAP Value|", ascending=True)
            fig = px.bar(
                df_shap,
                x="Mean |SHAP Value|",
                y="Feature",
                orientation="h",
                title="Mean |SHAP Value| (top features)",
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        elif importance:
            import pandas as pd

            st.subheader("Model Feature Importance")
            df_imp = pd.DataFrame(
                list(importance.items()),
                columns=["Feature", "Importance"],
            ).sort_values("Importance", ascending=True)
            fig = px.bar(
                df_imp.tail(15),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 15 Features by Importance",
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

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

            import pandas as pd

            # Handle LocalExplanation format (contributions list)
            if isinstance(sv, dict) and "contributions" in sv:
                contributions = sv["contributions"]
                sv_df = pd.DataFrame(contributions)
                sv_df = sv_df.rename(
                    columns={"contribution": "SHAP Value", "feature": "Feature"}
                )
                sv_df["abs_shap"] = sv_df["SHAP Value"].abs()
                sv_df = sv_df.sort_values("abs_shap", ascending=False).drop(
                    columns=["abs_shap"]
                )
                fig = px.bar(
                    sv_df.head(15),
                    x="SHAP Value",
                    y="Feature",
                    orientation="h",
                    title=f"Top SHAP Values — {selected}",
                )
                st.plotly_chart(fig, use_container_width=True)
            elif isinstance(sv, dict):
                sv_df = pd.DataFrame(
                    list(sv.items()), columns=["Feature", "SHAP Value"]
                )
                sv_df["abs_shap"] = sv_df["SHAP Value"].abs()
                sv_df = sv_df.sort_values("abs_shap", ascending=False).drop(
                    columns=["abs_shap"]
                )
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
