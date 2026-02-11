"""Streamlit monitoring dashboard for the Fraud Detection System."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports work when
# Streamlit is launched directly (e.g. `streamlit run src/dashboard/app.py`).
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st  # noqa: E402
from streamlit_autorefresh import st_autorefresh  # noqa: E402

from src.dashboard.data import get_connection  # noqa: E402

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
# Simulator controls (sidebar)
# ------------------------------------------------------------------

st.sidebar.divider()
st.sidebar.subheader("Simulator")

if "simulator_process" not in st.session_state:
    st.session_state.simulator_process = None


def _simulator_running() -> bool:
    proc = st.session_state.simulator_process
    return proc is not None and proc.poll() is None


if _simulator_running():
    st.sidebar.success("Simulator is running")
    if st.sidebar.button("Stop Simulator"):
        st.session_state.simulator_process.terminate()
        st.session_state.simulator_process.wait()
        st.session_state.simulator_process = None
        st.rerun()
else:
    rate = st.sidebar.slider("Transactions / sec", min_value=1, max_value=50, value=10)
    if st.sidebar.button("Start Simulator"):
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "src.streaming.run_simulator",
            ],
            cwd=_project_root,
            env={**__import__("os").environ, "STREAM_RATE": str(rate)},
        )
        st.session_state.simulator_process = proc
        st.rerun()

# ------------------------------------------------------------------
# Database connection — fresh per run to avoid cross-thread errors
# ------------------------------------------------------------------


def _get_conn() -> sqlite3.Connection:
    return get_connection()


conn = _get_conn()

# ------------------------------------------------------------------
# Route to selected page
# ------------------------------------------------------------------

if page == "Overview":
    from src.dashboard._pages.overview import render

    render(conn)
elif page == "Real-time Feed":
    from src.dashboard._pages.realtime_feed import render

    render(conn)
elif page == "Model Performance":
    from src.dashboard._pages.performance import render

    render(conn)
elif page == "A/B Test Results":
    from src.dashboard._pages.ab_test import render

    render(conn)
elif page == "Feature Importance":
    from src.dashboard._pages.feature_importance import render

    render(conn)
elif page == "Alert Log":
    from src.dashboard._pages.alerts import render

    render(conn)
