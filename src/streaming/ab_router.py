"""A/B testing framework for comparing fraud-detection model variants."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------


@dataclass
class SignificanceResult:
    """Outcome of a statistical significance test between two models."""

    chi2_statistic: float
    p_value: float
    is_significant: bool
    sample_size_a: int
    sample_size_b: int
    fraud_rate_a: float
    fraud_rate_b: float


class MetricsTracker:
    """Accumulate per-model prediction metrics for A/B evaluation."""

    def __init__(self) -> None:
        self.predictions: List[int] = []
        self.actuals: List[Optional[int]] = []
        self.latencies_ms: List[float] = []

    def record(
        self,
        prediction: int,
        actual: Optional[int] = None,
        latency_ms: float = 0.0,
    ) -> None:
        self.predictions.append(prediction)
        self.actuals.append(actual)
        self.latencies_ms.append(latency_ms)

    @property
    def count(self) -> int:
        return len(self.predictions)

    @property
    def fraud_rate(self) -> float:
        """Fraction of predictions that were flagged as fraud."""
        if not self.predictions:
            return 0.0
        return sum(self.predictions) / len(self.predictions)

    @property
    def mean_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return float(np.mean(self.latencies_ms))

    @property
    def accuracy(self) -> Optional[float]:
        """Accuracy computed over predictions that have a known actual label."""
        pairs = [
            (p, a)
            for p, a in zip(self.predictions, self.actuals)
            if a is not None
        ]
        if not pairs:
            return None
        correct = sum(1 for p, a in pairs if p == a)
        return correct / len(pairs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "fraud_rate": round(self.fraud_rate, 6),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "accuracy": (
                round(self.accuracy, 6) if self.accuracy is not None else None
            ),
        }


# ------------------------------------------------------------------
# A/B Test Router
# ------------------------------------------------------------------


class ABTestRouter:
    """Route transactions to one of two models for A/B comparison.

    Routing is **deterministic**: the same ``transaction_id`` always maps
    to the same model variant, ensuring consistency across retries or
    replays.

    Parameters
    ----------
    model_a:
        First model (variant *A*).
    model_b:
        Second model (variant *B*).
    split_ratio:
        Fraction of traffic sent to model *A* (``0.0``–``1.0``).
    threshold_a:
        Classification threshold for model *A*.
    threshold_b:
        Classification threshold for model *B*.
    """

    def __init__(
        self,
        model_a: Any,
        model_b: Any,
        split_ratio: float = 0.5,
        threshold_a: float = 0.5,
        threshold_b: float = 0.5,
    ) -> None:
        self.models: Dict[str, Any] = {"A": model_a, "B": model_b}
        self.split_ratio = split_ratio
        self.thresholds: Dict[str, float] = {
            "A": threshold_a,
            "B": threshold_b,
        }
        self.metrics: Dict[str, MetricsTracker] = {
            "A": MetricsTracker(),
            "B": MetricsTracker(),
        }

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(self, transaction_id: str) -> str:
        """Return ``'A'`` or ``'B'`` based on a deterministic hash."""
        digest = hashlib.md5(transaction_id.encode()).hexdigest()
        bucket = int(digest, 16) % 100
        return "A" if bucket < self.split_ratio * 100 else "B"

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        transaction_id: str,
        features: np.ndarray,
        actual: Optional[int] = None,
    ) -> Tuple[str, int, float]:
        """Route, predict, and record metrics.

        Parameters
        ----------
        transaction_id:
            Unique ID used for deterministic routing.
        features:
            1-D feature array for one transaction.
        actual:
            Optional ground-truth label (for later accuracy computation).

        Returns
        -------
        tuple[str, int, float]
            ``(variant, prediction, latency_ms)``
        """
        variant = self.route(transaction_id)
        model = self.models[variant]
        threshold = self.thresholds[variant]

        x = np.asarray(features).reshape(1, -1)

        start = time.perf_counter()
        proba = float(model.predict_proba(x)[0, 1])
        latency_ms = (time.perf_counter() - start) * 1000

        prediction = int(proba >= threshold)

        self.metrics[variant].record(
            prediction=prediction,
            actual=actual,
            latency_ms=latency_ms,
        )

        return variant, prediction, latency_ms

    # ------------------------------------------------------------------
    # Statistical significance
    # ------------------------------------------------------------------

    def compute_significance(
        self, alpha: float = 0.05,
    ) -> SignificanceResult:
        """Run a chi-squared test comparing fraud detection rates.

        Parameters
        ----------
        alpha:
            Significance level.  ``is_significant`` is ``True`` when
            ``p_value < alpha``.

        Returns
        -------
        SignificanceResult
        """
        from scipy.stats import chi2_contingency

        ma = self.metrics["A"]
        mb = self.metrics["B"]

        fraud_a = sum(ma.predictions)
        normal_a = ma.count - fraud_a
        fraud_b = sum(mb.predictions)
        normal_b = mb.count - fraud_b

        table = np.array([[fraud_a, normal_a], [fraud_b, normal_b]])

        # Guard against degenerate tables (all zeros in a row/col)
        if (
            table.sum() == 0
            or (table.sum(axis=0) == 0).any()
            or (table.sum(axis=1) == 0).any()
        ):
            return SignificanceResult(
                chi2_statistic=0.0,
                p_value=1.0,
                is_significant=False,
                sample_size_a=ma.count,
                sample_size_b=mb.count,
                fraud_rate_a=ma.fraud_rate,
                fraud_rate_b=mb.fraud_rate,
            )

        chi2, p_value, _, _ = chi2_contingency(table)

        return SignificanceResult(
            chi2_statistic=float(chi2),
            p_value=float(p_value),
            is_significant=bool(p_value < alpha),
            sample_size_a=ma.count,
            sample_size_b=mb.count,
            fraud_rate_a=ma.fraud_rate,
            fraud_rate_b=mb.fraud_rate,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def get_results(self) -> Dict[str, Any]:
        """Return a JSON-safe summary of both variants and significance."""
        sig = self.compute_significance()
        return {
            "split_ratio": self.split_ratio,
            "model_a": self.metrics["A"].to_dict(),
            "model_b": self.metrics["B"].to_dict(),
            "significance": {
                "chi2_statistic": round(sig.chi2_statistic, 6),
                "p_value": round(sig.p_value, 6),
                "is_significant": sig.is_significant,
            },
        }
