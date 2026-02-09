"""Model training pipeline with cross-validation for fraud detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FoldMetrics:
    """Metrics for a single cross-validation fold."""

    fold: int
    precision: float
    recall: float
    f1: float
    auc_roc: float


@dataclass
class CVResults:
    """Aggregated cross-validation results across all folds."""

    folds: List[FoldMetrics] = field(default_factory=list)

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    @property
    def mean_precision(self) -> float:
        return float(np.mean([f.precision for f in self.folds]))

    @property
    def mean_recall(self) -> float:
        return float(np.mean([f.recall for f in self.folds]))

    @property
    def mean_f1(self) -> float:
        return float(np.mean([f.f1 for f in self.folds]))

    @property
    def mean_auc_roc(self) -> float:
        return float(np.mean([f.auc_roc for f in self.folds]))

    def summary(self) -> Dict[str, float]:
        return {
            "mean_precision": self.mean_precision,
            "mean_recall": self.mean_recall,
            "mean_f1": self.mean_f1,
            "mean_auc_roc": self.mean_auc_roc,
        }


@dataclass
class TrainingMetadata:
    """Metadata captured during model training."""

    model_type: str
    params: Dict[str, Any]
    n_samples: int
    n_features: int
    class_distribution: Dict[int, int]
    trained_at: str


class ModelTrainer:
    """Train and cross-validate fraud detection models.

    Supported model types: ``logistic_regression``, ``random_forest``,
    ``xgboost``.
    """

    MODELS: Dict[str, Type] = {
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "xgboost": XGBClassifier,
    }

    def __init__(self, model_type: str, params: Optional[Dict[str, Any]] = None) -> None:
        if model_type not in self.MODELS:
            raise ValueError(
                f"Unknown model type: {model_type!r}. "
                f"Choose from {list(self.MODELS.keys())}"
            )
        self.model_type = model_type
        self.params = params or {}
        self.model = self._build_model(self.params)
        self.metadata: Optional[TrainingMetadata] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_weight: Optional[Dict[int, float]] = None,
    ) -> "ModelTrainer":
        """Fit the model on the provided data.

        Parameters
        ----------
        X:
            Feature matrix ``(n_samples, n_features)``.
        y:
            Binary label array ``(n_samples,)``.
        class_weight:
            Optional class weights.  For sklearn models this is set via
            ``class_weight``; for XGBoost it sets ``scale_pos_weight``.

        Returns
        -------
        ModelTrainer
            ``self``, for method chaining.
        """
        fit_params = self.params.copy()

        if class_weight is not None:
            if self.model_type == "xgboost":
                # XGBoost uses scale_pos_weight (ratio of neg to pos)
                fit_params["scale_pos_weight"] = class_weight.get(1, 1.0) / max(
                    class_weight.get(0, 1.0), 1e-9
                )
            else:
                fit_params["class_weight"] = class_weight

            # Rebuild model with updated params
            self.model = self._build_model(fit_params)

        self.model.fit(X, y)

        # Capture training metadata
        unique, counts = np.unique(y.astype(int), return_counts=True)
        self.metadata = TrainingMetadata(
            model_type=self.model_type,
            params=fit_params,
            n_samples=int(X.shape[0]),
            n_features=int(X.shape[1]),
            class_distribution=dict(zip(unique.tolist(), counts.tolist())),
            trained_at=datetime.now(timezone.utc).isoformat(),
        )

        logger.info(
            "Trained %s on %d samples (%d features)",
            self.model_type,
            X.shape[0],
            X.shape[1],
        )
        return self

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
    ) -> CVResults:
        """Run stratified k-fold cross-validation.

        Parameters
        ----------
        X:
            Feature matrix.
        y:
            Binary label array.
        cv:
            Number of folds.

        Returns
        -------
        CVResults
            Per-fold and aggregated metrics (precision, recall, F1, AUC-ROC).
        """
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        results = CVResults()

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = self._build_model(self.params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1]

            fold = FoldMetrics(
                fold=fold_idx,
                precision=float(precision_score(y_val, y_pred, zero_division=0)),
                recall=float(recall_score(y_val, y_pred, zero_division=0)),
                f1=float(f1_score(y_val, y_pred, zero_division=0)),
                auc_roc=float(roc_auc_score(y_val, y_proba)),
            )
            results.folds.append(fold)

        logger.info(
            "Cross-validation (%d folds): mean F1=%.4f, mean AUC=%.4f",
            cv,
            results.mean_f1,
            results.mean_auc_roc,
        )
        return results

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates for each class."""
        return self.model.predict_proba(X)

    # ------------------------------------------------------------------
    # Hyperparameter optimization (Optuna)
    # ------------------------------------------------------------------

    def optimize_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 20,
    ) -> Dict[str, Any]:
        """Search for the best hyperparameters using Optuna.

        Parameters
        ----------
        X:
            Feature matrix.
        y:
            Binary label array.
        n_trials:
            Number of Optuna trials to run.

        Returns
        -------
        dict
            Best hyperparameters found during the study.
        """
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            params = self._suggest_params(trial)
            model = self._build_model(params)
            scores = cross_val_score(model, X, y, cv=3, scoring="f1")
            return float(scores.mean())

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        best = study.best_params
        logger.info(
            "Optuna optimization complete (%d trials): best F1=%.4f, params=%s",
            n_trials,
            study.best_value,
            best,
        )
        return best

    def _suggest_params(self, trial) -> Dict[str, Any]:
        """Return trial-suggested hyperparameters for the current model type."""
        if self.model_type == "xgboost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }
        if self.model_type == "random_forest":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            }
        if self.model_type == "logistic_regression":
            return {
                "C": trial.suggest_float("C", 0.01, 10.0, log=True),
                "max_iter": 1000,
            }
        raise ValueError(f"No search space defined for {self.model_type!r}")

    # ------------------------------------------------------------------
    # Threshold optimization
    # ------------------------------------------------------------------

    def optimize_threshold(self, X: np.ndarray, y: np.ndarray) -> float:
        """Find the classification threshold that maximises recall at precision >= 0.5.

        The model must already be trained before calling this method.

        Parameters
        ----------
        X:
            Feature matrix (validation set).
        y:
            True binary labels (validation set).

        Returns
        -------
        float
            Optimal threshold in ``[0, 1]``.
        """
        y_proba = self.model.predict_proba(X)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y, y_proba)

        # Find threshold where precision >= 0.5 and recall is maximized
        valid_idx = precision[:-1] >= 0.5
        if valid_idx.any():
            best_idx = np.argmax(recall[:-1][valid_idx])
            threshold = float(thresholds[valid_idx][best_idx])
        else:
            threshold = 0.5

        logger.info("Optimized threshold: %.4f", threshold)
        return threshold

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_model(self, params: Dict[str, Any]) -> Any:
        """Instantiate a fresh model with the given parameters."""
        cls = self.MODELS[self.model_type]
        filtered = self._filter_params(cls, params)
        return cls(**filtered)

    @staticmethod
    def _filter_params(cls: Type, params: Dict[str, Any]) -> Dict[str, Any]:
        """Remove parameters that the model class does not accept."""
        import inspect

        sig = inspect.signature(cls)
        valid = set(sig.parameters.keys())
        # If **kwargs is accepted, allow everything through
        if any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        ):
            return params
        return {k: v for k, v in params.items() if k in valid}
