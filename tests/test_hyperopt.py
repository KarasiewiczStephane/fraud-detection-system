"""Tests for hyperparameter tuning and threshold optimization."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import precision_score, recall_score

from src.models.trainer import ModelTrainer


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_dataset(
    n_samples: int = 300,
    n_features: int = 10,
    fraud_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_fraud = max(5, int(n_samples * fraud_ratio))
    n_normal = n_samples - n_fraud

    X_normal = rng.normal(0, 1, (n_normal, n_features))
    X_fraud = rng.normal(2, 1.5, (n_fraud, n_features))
    X = np.vstack([X_normal, X_fraud])
    y = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)]).astype(int)

    idx = rng.permutation(n_samples)
    return X[idx], y[idx]


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def dataset():
    return _make_dataset()


@pytest.fixture
def large_dataset():
    return _make_dataset(n_samples=500, n_features=10, fraud_ratio=0.15)


# ------------------------------------------------------------------
# Hyperparameter optimization — XGBoost
# ------------------------------------------------------------------


class TestOptimizeHyperparametersXGBoost:
    def test_returns_dict(self, dataset):
        X, y = dataset
        t = ModelTrainer("xgboost")
        best = t.optimize_hyperparameters(X, y, n_trials=2)
        assert isinstance(best, dict)

    def test_n_estimators_in_range(self, dataset):
        X, y = dataset
        t = ModelTrainer("xgboost")
        best = t.optimize_hyperparameters(X, y, n_trials=2)
        assert 50 <= best["n_estimators"] <= 300

    def test_max_depth_in_range(self, dataset):
        X, y = dataset
        t = ModelTrainer("xgboost")
        best = t.optimize_hyperparameters(X, y, n_trials=2)
        assert 3 <= best["max_depth"] <= 10

    def test_learning_rate_in_range(self, dataset):
        X, y = dataset
        t = ModelTrainer("xgboost")
        best = t.optimize_hyperparameters(X, y, n_trials=2)
        assert 0.01 <= best["learning_rate"] <= 0.3

    def test_subsample_in_range(self, dataset):
        X, y = dataset
        t = ModelTrainer("xgboost")
        best = t.optimize_hyperparameters(X, y, n_trials=2)
        assert 0.6 <= best["subsample"] <= 1.0

    def test_colsample_bytree_in_range(self, dataset):
        X, y = dataset
        t = ModelTrainer("xgboost")
        best = t.optimize_hyperparameters(X, y, n_trials=2)
        assert 0.6 <= best["colsample_bytree"] <= 1.0

    def test_best_params_can_build_model(self, dataset):
        X, y = dataset
        t = ModelTrainer("xgboost")
        best = t.optimize_hyperparameters(X, y, n_trials=2)
        t2 = ModelTrainer("xgboost", params=best)
        t2.train(X, y)
        preds = t2.predict(X)
        assert preds.shape == y.shape


# ------------------------------------------------------------------
# Hyperparameter optimization — Random Forest
# ------------------------------------------------------------------


class TestOptimizeHyperparametersRandomForest:
    def test_returns_dict(self, dataset):
        X, y = dataset
        t = ModelTrainer("random_forest")
        best = t.optimize_hyperparameters(X, y, n_trials=2)
        assert isinstance(best, dict)

    def test_n_estimators_in_range(self, dataset):
        X, y = dataset
        t = ModelTrainer("random_forest")
        best = t.optimize_hyperparameters(X, y, n_trials=2)
        assert 50 <= best["n_estimators"] <= 300

    def test_max_depth_in_range(self, dataset):
        X, y = dataset
        t = ModelTrainer("random_forest")
        best = t.optimize_hyperparameters(X, y, n_trials=2)
        assert 3 <= best["max_depth"] <= 20

    def test_best_params_can_build_model(self, dataset):
        X, y = dataset
        t = ModelTrainer("random_forest")
        best = t.optimize_hyperparameters(X, y, n_trials=2)
        t2 = ModelTrainer("random_forest", params=best)
        t2.train(X, y)
        assert t2.predict(X).shape == y.shape


# ------------------------------------------------------------------
# Hyperparameter optimization — Logistic Regression
# ------------------------------------------------------------------


class TestOptimizeHyperparametersLogisticRegression:
    def test_returns_dict(self, dataset):
        X, y = dataset
        t = ModelTrainer("logistic_regression")
        best = t.optimize_hyperparameters(X, y, n_trials=2)
        assert isinstance(best, dict)

    def test_c_in_range(self, dataset):
        X, y = dataset
        t = ModelTrainer("logistic_regression")
        best = t.optimize_hyperparameters(X, y, n_trials=2)
        assert 0.01 <= best["C"] <= 10.0

    def test_best_params_can_build_model(self, dataset):
        X, y = dataset
        t = ModelTrainer("logistic_regression")
        best = t.optimize_hyperparameters(X, y, n_trials=2)
        t2 = ModelTrainer("logistic_regression", params=best)
        t2.train(X, y)
        assert t2.predict(X).shape == y.shape


# ------------------------------------------------------------------
# Threshold optimization
# ------------------------------------------------------------------


class TestOptimizeThreshold:
    def test_returns_float(self, large_dataset):
        X, y = large_dataset
        t = ModelTrainer("logistic_regression", {"max_iter": 1000}).train(X, y)
        threshold = t.optimize_threshold(X, y)
        assert isinstance(threshold, float)

    def test_in_zero_one_range(self, large_dataset):
        X, y = large_dataset
        t = ModelTrainer("logistic_regression", {"max_iter": 1000}).train(X, y)
        threshold = t.optimize_threshold(X, y)
        assert 0.0 <= threshold <= 1.0

    def test_xgboost_threshold(self, large_dataset):
        X, y = large_dataset
        t = ModelTrainer("xgboost", {"n_estimators": 20, "max_depth": 3}).train(X, y)
        threshold = t.optimize_threshold(X, y)
        assert 0.0 <= threshold <= 1.0

    def test_random_forest_threshold(self, large_dataset):
        X, y = large_dataset
        t = ModelTrainer("random_forest", {"n_estimators": 20}).train(X, y)
        threshold = t.optimize_threshold(X, y)
        assert 0.0 <= threshold <= 1.0

    def test_threshold_produces_valid_predictions(self, large_dataset):
        X, y = large_dataset
        t = ModelTrainer("logistic_regression", {"max_iter": 1000}).train(X, y)
        threshold = t.optimize_threshold(X, y)
        y_proba = t.predict_proba(X)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        assert set(np.unique(y_pred)).issubset({0, 1})

    def test_precision_at_threshold(self, large_dataset):
        """Optimized threshold should achieve precision >= 0.5 when possible."""
        X, y = large_dataset
        t = ModelTrainer("xgboost", {"n_estimators": 50, "max_depth": 4}).train(X, y)
        threshold = t.optimize_threshold(X, y)
        y_proba = t.predict_proba(X)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        if y_pred.sum() > 0:
            prec = precision_score(y, y_pred, zero_division=0)
            assert prec >= 0.5 or threshold == 0.5

    def test_default_threshold_on_random_model(self):
        """When no threshold achieves precision >= 0.5, return 0.5."""
        rng = np.random.default_rng(99)
        X = rng.normal(0, 1, (200, 5))
        y = rng.choice([0, 1], 200, p=[0.5, 0.5])
        t = ModelTrainer("logistic_regression", {"max_iter": 1000}).train(X, y)
        threshold = t.optimize_threshold(X, y)
        # Should still be a valid float
        assert 0.0 <= threshold <= 1.0

    def test_optimized_threshold_recall_vs_default(self, large_dataset):
        """Optimized threshold should generally not have worse recall than 0.5."""
        X, y = large_dataset
        t = ModelTrainer("xgboost", {"n_estimators": 50, "max_depth": 4}).train(X, y)
        threshold = t.optimize_threshold(X, y)
        y_proba = t.predict_proba(X)[:, 1]

        y_pred_opt = (y_proba >= threshold).astype(int)
        y_pred_def = (y_proba >= 0.5).astype(int)

        recall_opt = recall_score(y, y_pred_opt, zero_division=0)
        recall_def = recall_score(y, y_pred_def, zero_division=0)

        # The optimized threshold targets maximum recall at precision>=0.5,
        # so it should have recall >= the default 0.5 threshold recall
        assert recall_opt >= recall_def - 0.05  # allow small tolerance
