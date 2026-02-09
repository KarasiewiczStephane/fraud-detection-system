"""SHAP-based model explainability for fraud detection."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import shap

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Tree-based model types that can use the fast TreeExplainer
_TREE_MODELS = {"xgboost", "random_forest"}


@dataclass
class LocalExplanation:
    """Explanation for a single prediction."""

    base_value: float
    contributions: List[Dict[str, Any]]
    prediction: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LocalExplanation":
        return cls(
            base_value=data["base_value"],
            contributions=data["contributions"],
            prediction=data["prediction"],
        )


class SHAPExplainer:
    """Compute SHAP explanations for fraud-detection models.

    Parameters
    ----------
    model:
        A trained model exposing ``predict_proba``.
    model_type:
        One of ``"xgboost"``, ``"random_forest"``, ``"logistic_regression"``.
        Tree-based models use the fast :class:`shap.TreeExplainer`;
        others fall back to :class:`shap.KernelExplainer`.
    background_data:
        A small representative sample of the training data used as the
        background dataset for :class:`shap.KernelExplainer`.  Required
        for non-tree models; ignored for tree models.
    """

    def __init__(
        self,
        model: Any,
        model_type: str = "xgboost",
        background_data: Optional[np.ndarray] = None,
    ) -> None:
        self.model = model
        self.model_type = model_type

        if model_type in _TREE_MODELS:
            self.explainer = shap.TreeExplainer(model)
        else:
            if background_data is None:
                raise ValueError(
                    "background_data is required for non-tree models "
                    f"(model_type={model_type!r})"
                )
            # Wrap predict_proba to return only the positive-class probability
            # so SHAP values have a consistent (n_samples, n_features) shape.
            self.explainer = shap.KernelExplainer(
                lambda X: model.predict_proba(X)[:, 1],
                background_data,
            )

    # ------------------------------------------------------------------
    # SHAP values
    # ------------------------------------------------------------------

    def compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        """Return SHAP values of shape ``(n_samples, n_features)``.

        For tree models the values are in log-odds space; for kernel
        models they are in probability space (positive class).
        """
        sv = self.explainer.shap_values(X)
        sv = np.asarray(sv)

        # TreeExplainer for multi-output models (e.g. RandomForest) returns
        # shape (n_samples, n_features, n_classes).  Extract the positive
        # class (index 1) to get a consistent (n_samples, n_features) array.
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        elif sv.ndim == 1:
            sv = sv.reshape(1, -1)
        return sv

    # ------------------------------------------------------------------
    # Global importance
    # ------------------------------------------------------------------

    def global_importance(
        self,
        X: np.ndarray,
        feature_names: List[str],
        top_n: int = 15,
    ) -> Dict[str, float]:
        """Return the top *top_n* features ranked by mean |SHAP value|.

        Parameters
        ----------
        X:
            Feature matrix used to compute SHAP values.
        feature_names:
            Column names matching the features in *X*.
        top_n:
            Number of top features to return.

        Returns
        -------
        dict[str, float]
            Ordered mapping of feature name → mean |SHAP value|,
            sorted descending by importance.
        """
        shap_values = self.compute_shap_values(X)
        importance = np.abs(shap_values).mean(axis=0)
        pairs = sorted(
            zip(feature_names, importance.tolist()),
            key=lambda p: -p[1],
        )
        return dict(pairs[:top_n])

    # ------------------------------------------------------------------
    # Local explanation
    # ------------------------------------------------------------------

    def local_explanation(
        self,
        x: np.ndarray,
        feature_names: List[str],
        top_n: int = 10,
    ) -> LocalExplanation:
        """Explain a single prediction.

        Parameters
        ----------
        x:
            1-D feature vector for one transaction.
        feature_names:
            Column names matching the features in *x*.
        top_n:
            Number of top contributing features to include.

        Returns
        -------
        LocalExplanation
        """
        x_2d = np.asarray(x).reshape(1, -1)
        shap_values = self.explainer.shap_values(x_2d)
        sv = np.asarray(shap_values)
        # Handle 3-D output from multi-output TreeExplainer (RandomForest)
        if sv.ndim == 3:
            sv = sv[:, :, 1]  # positive class
        sv = sv.flatten()

        base_value = self.explainer.expected_value
        if isinstance(base_value, (np.ndarray, list)):
            bv = np.asarray(base_value)
            # Multi-output (e.g. RandomForest): pick positive-class base value
            base_value = float(bv[1]) if bv.size > 1 else float(bv.flat[0])
        else:
            base_value = float(base_value)

        contributions = [
            {
                "feature": name,
                "value": float(x[i]),
                "contribution": float(sv[i]),
            }
            for i, name in enumerate(feature_names)
        ]
        contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)

        return LocalExplanation(
            base_value=base_value,
            contributions=contributions[:top_n],
            prediction=float(base_value + sv.sum()),
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @staticmethod
    def save_explanation(explanation: LocalExplanation, transaction_id: str) -> Dict[str, Any]:
        """Serialize a local explanation for database / audit storage.

        Returns a JSON-safe dictionary keyed by *transaction_id*.
        """
        payload = {
            "transaction_id": transaction_id,
            **explanation.to_dict(),
        }
        logger.info("Serialized explanation for transaction %s", transaction_id)
        return payload

    @staticmethod
    def load_explanation(data: Dict[str, Any]) -> LocalExplanation:
        """Reconstruct a :class:`LocalExplanation` from a serialized dict."""
        return LocalExplanation.from_dict(data)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def generate_plots(
        self,
        X: np.ndarray,
        feature_names: List[str],
        output_dir: Path,
    ) -> List[Path]:
        """Generate and save SHAP summary plots.

        Creates:
        - ``shap_beeswarm.png`` — global beeswarm summary plot
        - ``shap_bar.png`` — global feature importance bar chart

        Parameters
        ----------
        X:
            Feature matrix used to compute SHAP values.
        feature_names:
            Column names matching the features in *X*.
        output_dir:
            Directory where PNG files are written.

        Returns
        -------
        list[Path]
            Paths to the generated plot files.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        shap_values = self.compute_shap_values(X)
        paths: List[Path] = []

        # --- Beeswarm summary plot ---
        plt.figure()
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        beeswarm_path = output_dir / "shap_beeswarm.png"
        plt.savefig(beeswarm_path, bbox_inches="tight", dpi=150)
        plt.close()
        paths.append(beeswarm_path)

        # --- Bar importance plot ---
        plt.figure()
        shap.summary_plot(
            shap_values, X, feature_names=feature_names, plot_type="bar", show=False,
        )
        bar_path = output_dir / "shap_bar.png"
        plt.savefig(bar_path, bbox_inches="tight", dpi=150)
        plt.close()
        paths.append(bar_path)

        logger.info("Saved %d SHAP plots to %s", len(paths), output_dir)
        return paths
