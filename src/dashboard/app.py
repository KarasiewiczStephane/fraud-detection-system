"""Streamlit monitoring dashboard for the Fraud Detection System."""

from __future__ import annotations

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from src.dashboard.data import get_connection

# ------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ------------------------------------------------------------------

st.set_page_config(page_title="Fraud Detection Monitor", layout="wide")

# Auto-refresh every 30 seconds
st_autorefresh(interval=30_000, key="refresh")

# ------------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------------

PAGES = [
    "Overview",
    "Real-time Feed",
    "Model Performance",
    "A/B Test Results",
    "Feature Importance",
    "Alert Log",
]

page = st.sidebar.selectbox("Page", PAGES)

# ------------------------------------------------------------------
# Database connection (cached)
# ------------------------------------------------------------------


@st.cache_resource
def _db_connection():
    return get_connection()


conn = _db_connection()

# ------------------------------------------------------------------
# Route to selected page
# ------------------------------------------------------------------

if page == "Overview":
    from src.dashboard.pages.overview import render

    render(conn)
elif page == "Real-time Feed":
    from src.dashboard.pages.realtime_feed import render

    render(conn)
elif page == "Model Performance":
    from src.dashboard.pages.performance import render

    render(conn)
elif page == "A/B Test Results":
    from src.dashboard.pages.ab_test import render

    render(conn)
elif page == "Feature Importance":
    from src.dashboard.pages.feature_importance import render

    render(conn)
elif page == "Alert Log":
    from src.dashboard.pages.alerts import render

    render(conn)
