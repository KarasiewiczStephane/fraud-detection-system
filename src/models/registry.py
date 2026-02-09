"""Versioned model registry backed by joblib and JSON metadata."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib

from src.models.evaluator import EvaluationMetrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """Persist and retrieve trained models with metadata.

    Directory layout::

        base_path/
            <name>/
                <version>/
                    model.joblib
                    metadata.json
    """

    _MODEL_FILE = "model.joblib"
    _META_FILE = "metadata.json"

    def __init__(self, base_path: Path) -> None:
        self.base_path = Path(base_path)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        model: Any,
        name: str,
        metrics: EvaluationMetrics,
        params: Dict[str, Any],
        threshold: float,
    ) -> str:
        """Save a trained model with its metadata.

        Parameters
        ----------
        model:
            The trained model object (must be joblib-serializable).
        name:
            Logical model name (e.g. ``"xgboost"``).
        metrics:
            Evaluation metrics to store alongside the model.
        params:
            Hyperparameters used to train the model.
        threshold:
            Classification threshold associated with this model.

        Returns
        -------
        str
            The auto-generated version string (``YYYYMMDD_HHMMSS``).
        """
        version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        version_dir = self.base_path / name / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Serialize model
        model_path = version_dir / self._MODEL_FILE
        joblib.dump(model, model_path)

        # Write metadata
        metadata = {
            "name": name,
            "version": version,
            "metrics": asdict(metrics),
            "params": params,
            "threshold": threshold,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_path = version_dir / self._META_FILE
        meta_path.write_text(json.dumps(metadata, indent=2))

        logger.info("Saved model '%s' version '%s'", name, version)
        return version

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(
        self,
        name: str,
        version: str = "latest",
    ) -> Tuple[Any, Dict[str, Any]]:
        """Load a model and its metadata.

        Parameters
        ----------
        name:
            Logical model name.
        version:
            Exact version string or ``"latest"`` for the most recent.

        Returns
        -------
        tuple[Any, dict]
            ``(model, metadata_dict)``

        Raises
        ------
        FileNotFoundError
            If the model or version does not exist.
        """
        if version == "latest":
            version = self._resolve_latest(name)

        version_dir = self.base_path / name / version
        model_path = version_dir / self._MODEL_FILE
        meta_path = version_dir / self._META_FILE

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        model = joblib.load(model_path)

        metadata: Dict[str, Any] = {}
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text())

        logger.info("Loaded model '%s' version '%s'", name, version)
        return model, metadata

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_models(self) -> List[str]:
        """Return the names of all registered models."""
        if not self.base_path.exists():
            return []
        return sorted(
            d.name
            for d in self.base_path.iterdir()
            if d.is_dir()
        )

    def list_versions(self, name: str) -> List[str]:
        """Return all version strings for a given model name, oldest first."""
        model_dir = self.base_path / name
        if not model_dir.exists():
            return []
        return sorted(
            d.name
            for d in model_dir.iterdir()
            if d.is_dir() and (d / self._META_FILE).exists()
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_latest(self, name: str) -> str:
        versions = self.list_versions(name)
        if not versions:
            raise FileNotFoundError(
                f"No versions found for model '{name}' in {self.base_path}"
            )
        return versions[-1]
