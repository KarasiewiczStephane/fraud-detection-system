"""Model evaluation and comparison report generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for a binary classifier."""

    precision: float
    recall: float
    f1: float
    auc_roc: float
    auc_pr: float
    confusion_matrix: List[List[int]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelEvaluator:
    """Evaluate models and generate comparison reports."""

    def evaluate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        threshold: float = 0.5,
    ) -> EvaluationMetrics:
        """Compute evaluation metrics for a trained model.

        Parameters
        ----------
        model:
            A trained model exposing ``predict_proba``.
        X:
            Feature matrix.
        y:
            True binary labels.
        threshold:
            Classification threshold applied to the positive-class
            probability.

        Returns
        -------
        EvaluationMetrics
        """
        y_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        metrics = EvaluationMetrics(
            precision=float(precision_score(y, y_pred, zero_division=0)),
            recall=float(recall_score(y, y_pred, zero_division=0)),
            f1=float(f1_score(y, y_pred, zero_division=0)),
            auc_roc=float(roc_auc_score(y, y_proba)),
            auc_pr=float(average_precision_score(y, y_proba)),
            confusion_matrix=confusion_matrix(y, y_pred).tolist(),
        )

        logger.info(
            "Evaluation: precision=%.4f recall=%.4f F1=%.4f AUC-ROC=%.4f AUC-PR=%.4f",
            metrics.precision,
            metrics.recall,
            metrics.f1,
            metrics.auc_roc,
            metrics.auc_pr,
        )
        return metrics

    def compare_models(self, results: Dict[str, EvaluationMetrics]) -> str:
        """Generate a Markdown comparison table from named evaluation results.

        Parameters
        ----------
        results:
            Mapping of model name to its ``EvaluationMetrics``.

        Returns
        -------
        str
            Markdown-formatted table with the best value per metric
            highlighted with ``**bold**``.
        """
        if not results:
            return ""

        metric_names = ["precision", "recall", "f1", "auc_roc", "auc_pr"]
        model_names = list(results.keys())

        # Find best model per metric
        best: Dict[str, str] = {}
        for m in metric_names:
            best_name = max(model_names, key=lambda n: getattr(results[n], m))
            best[m] = best_name

        # Build header
        header = "| Model | " + " | ".join(metric_names) + " |"
        sep = "|---|" + "|".join(["---"] * len(metric_names)) + "|"

        rows: List[str] = []
        for name in model_names:
            metrics = results[name]
            cells: List[str] = []
            for m in metric_names:
                val = f"{getattr(metrics, m):.4f}"
                if best[m] == name:
                    val = f"**{val}**"
                cells.append(val)
            rows.append(f"| {name} | " + " | ".join(cells) + " |")

        table = "\n".join([header, sep, *rows])

        logger.info("Generated comparison table for %d models", len(results))
        return table
