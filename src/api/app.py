"""FastAPI application for the Fraud Detection API."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

from fastapi import FastAPI, Request, Response

from src.api.routes.ab_test import router as ab_test_router
from src.api.routes.predict import router as predict_router
from src.api.schemas import HealthResponse
from src.utils.database import DatabaseManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AppState:
    """Shared application state initialised during startup."""

    model: Any = None
    model_version: str = "unknown"
    threshold: float = 0.5
    db: Optional[DatabaseManager] = None
    ab_router: Any = None


# Module-level state — set during lifespan
_state: Optional[AppState] = None


def get_state() -> AppState:
    """Return the current application state (raises if not initialised)."""
    if _state is None:
        raise RuntimeError("Application state not initialised")
    return _state


def set_state(state: AppState) -> None:
    """Override the global state (used in tests)."""
    global _state
    _state = state


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup / shutdown resources."""
    global _state

    db = DatabaseManager()
    await db.connect()

    _state = AppState(db=db)

    logger.info("API started – state initialised")
    yield

    await db.close()
    _state = None
    logger.info("API shut down")


app = FastAPI(
    title="Fraud Detection API",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Timing middleware ──────────────────────────────────────────────


@app.middleware("http")
async def timing_middleware(request: Request, call_next) -> Response:
    """Add ``X-Process-Time-Ms`` header to every response."""
    start = time.perf_counter()
    response: Response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


# ── Routers ────────────────────────────────────────────────────────

app.include_router(predict_router, prefix="/api/v1")
app.include_router(ab_test_router, prefix="/api/v1")

# ── Root endpoints ─────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Check service health and report current model version."""
    state = get_state()
    return HealthResponse(status="healthy", model_version=state.model_version)
