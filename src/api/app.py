"""FastAPI application for the Fraud Detection API."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional

from fastapi import FastAPI, Request, Response

from src.api.routes.ab_test import router as ab_test_router
from src.api.routes.predict import router as predict_router
from src.api.schemas import HealthResponse
from src.utils.config import get_config
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
    explainer: Any = None


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


def _load_or_train_model(config):
    """Load the latest model from the registry, or train a default one."""
    from pathlib import Path

    from src.models.registry import ModelRegistry

    registry_path = Path(config.get("model", "registry_path", default="models"))
    registry = ModelRegistry(registry_path)
    model_name = config.get("model", "default_model", default="xgboost")

    # Try loading from registry
    if registry.list_models():
        try:
            model, meta = registry.load(model_name)
            version = meta.get("version", "unknown")
            threshold = meta.get("threshold", 0.5)
            return model, f"{model_name}_{version}", threshold
        except FileNotFoundError:
            logger.warning(
                "Model '%s' not found in registry, training default", model_name
            )

    # No saved model — train on sample data
    logger.info("No models in registry, training default model on sample data")
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    sample_path = Path(config.get("data", "sample_path", default="data/sample"))
    csv_path = sample_path / "sample_transactions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Sample dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
    X = df[feature_cols]
    y = df["Class"]

    model = RandomForestClassifier(
        n_estimators=50, max_depth=8, class_weight="balanced", random_state=42
    )
    model.fit(X, y)
    logger.info("Trained default RandomForest on %d samples", len(df))
    return model, "rf_default", 0.5


def _create_explainer(model):
    """Create a SHAP explainer for the model."""
    try:
        from src.models.explainer import SHAPExplainer

        explainer = SHAPExplainer(model, model_type="random_forest")
        logger.info("SHAP explainer initialised")
        return explainer
    except Exception as exc:
        logger.warning("Could not create SHAP explainer: %s", exc)
        return None


def _create_ab_router(model, model_version):
    """Create an A/B router using the same model as both variants (demo)."""
    try:
        from src.streaming.ab_router import ABTestRouter

        router = ABTestRouter(
            model_a=model,
            model_b=model,
        )
        logger.info("A/B router initialised")
        return router
    except Exception as exc:
        logger.warning("Could not create A/B router: %s", exc)
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup / shutdown resources."""
    global _state

    config = get_config()
    db_path = config.get("database", "path", default="data/predictions.db")
    db = DatabaseManager(db_path)
    await db.connect()

    # Load model from registry (or train a simple one if none exists)
    model, model_version, threshold = _load_or_train_model(config)

    # Set up SHAP explainer
    explainer = _create_explainer(model)

    # Set up A/B router (both variants use the same model for demo)
    ab_router_instance = _create_ab_router(model, model_version)

    _state = AppState(
        db=db,
        model=model,
        model_version=model_version,
        threshold=threshold,
        explainer=explainer,
        ab_router=ab_router_instance,
    )

    logger.info(
        "API started – model=%s, version=%s", type(model).__name__, model_version
    )
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
