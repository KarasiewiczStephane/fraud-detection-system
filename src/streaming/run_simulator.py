"""Entry-point script for the transaction stream simulator container."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path so `src.*` imports resolve
# when run directly (e.g. `python src/streaming/run_simulator.py`).
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.streaming.consumer import StreamConsumer  # noqa: E402
from src.streaming.simulator import TransactionSimulator  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

_FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id TEXT NOT NULL,
    features TEXT NOT NULL,
    prediction INTEGER NOT NULL,
    confidence REAL NOT NULL,
    model_version TEXT NOT NULL,
    shap_values TEXT,
    timestamp TEXT NOT NULL
);
"""


def _load_model():
    """Load or train a model (same logic as the API)."""
    from src.utils.config import get_config
    from src.models.registry import ModelRegistry

    config = get_config()
    registry_path = Path(config.get("model", "registry_path", default="models"))
    registry = ModelRegistry(registry_path)
    model_name = config.get("model", "default_model", default="xgboost")

    if registry.list_models():
        try:
            model, meta = registry.load(model_name)
            version = meta.get("version", "unknown")
            threshold = meta.get("threshold", 0.5)
            return model, f"{model_name}_{version}", threshold
        except FileNotFoundError:
            pass

    # Fallback: train on sample data
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    csv_path = Path("data/sample/sample_transactions.csv")
    df = pd.read_csv(csv_path)
    X = df[_FEATURE_COLS]
    y = df["Class"]
    model = RandomForestClassifier(
        n_estimators=50, max_depth=8, class_weight="balanced", random_state=42
    )
    model.fit(X, y)
    logger.info("Trained default RandomForest on %d samples", len(df))
    return model, "rf_default", 0.5


async def main() -> None:
    rate = float(os.environ.get("STREAM_RATE", "10"))
    data_path = os.environ.get("DATA_PATH", "data/sample/sample_transactions.csv")
    db_path = os.environ.get("DB_PATH", "data/predictions.db")

    logger.info("Starting simulator: rate=%.1f, data=%s", rate, data_path)

    model, model_version, threshold = _load_model()
    logger.info("Loaded model: %s (threshold=%.2f)", model_version, threshold)

    # Open a sync SQLite connection for writing predictions
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(db_path)
    db.execute(_CREATE_TABLE)
    db.commit()

    sim = TransactionSimulator(Path(data_path), rate=rate)
    consumer = StreamConsumer()
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)

    async def _inference(txn):
        """Run local inference and store the result in SQLite."""
        txn_id = txn.get("transaction_id", "unknown")
        features = {k: float(txn[k]) for k in _FEATURE_COLS if k in txn}
        x = np.array([features.get(c, 0.0) for c in _FEATURE_COLS]).reshape(1, -1)
        proba = float(model.predict_proba(x)[0, 1])
        is_fraud = proba >= threshold
        confidence = proba if is_fraud else 1.0 - proba

        db.execute(
            """INSERT INTO predictions
               (transaction_id, features, prediction, confidence,
                model_version, shap_values, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                txn_id,
                json.dumps(features),
                int(is_fraud),
                round(confidence, 6),
                model_version,
                None,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        db.commit()

        return {
            "transaction_id": txn_id,
            "is_fraud": is_fraud,
            "confidence": confidence,
        }

    async def _run_simulator():
        """Run the simulator then signal the consumer to stop."""
        await sim.stream(queue)
        await queue.join()
        consumer.stop()

    try:
        await asyncio.gather(
            _run_simulator(),
            consumer.consume(queue, _inference),
        )
    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
