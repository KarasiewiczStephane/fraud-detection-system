"""Tests for the model training pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.models.trainer import CVResults, FoldMetrics, ModelTrainer, TrainingMetadata


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

MODEL_PARAMS_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "model_params.yaml"
)


def _make_dataset(
    n_samples: int = 300,
    n_features: int = 10,
    fraud_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic binary-classification dataset."""
    rng = np.random.default_rng(seed)
    n_fraud = max(2, int(n_samples * fraud_ratio))
    n_normal = n_samples - n_fraud

    X_normal = rng.normal(0, 1, (n_normal, n_features))
    X_fraud = rng.normal(2, 1.5, (n_fraud, n_features))
    X = np.vstack([X_normal, X_fraud])
    y = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)]).astype(int)

    # Shuffle
    idx = rng.permutation(n_samples)
    return X[idx], y[idx]


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def dataset():
    return _make_dataset()


@pytest.fixture
def small_dataset():
    return _make_dataset(n_samples=100, n_features=5, fraud_ratio=0.2)


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------


class TestInit:
    def test_logistic_regression(self):
        t = ModelTrainer("logistic_regression")
        assert t.model_type == "logistic_regression"
        assert t.model is not None

    def test_random_forest(self):
        t = ModelTrainer("random_forest")
        assert t.model_type == "random_forest"

    def test_xgboost(self):
        t = ModelTrainer("xgboost")
        assert t.model_type == "xgboost"

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model type"):
            ModelTrainer("unknown_model")

    def test_custom_params(self):
        t = ModelTrainer("logistic_regression", params={"C": 0.5, "max_iter": 500})
        assert t.params == {"C": 0.5, "max_iter": 500}

    def test_default_params_empty(self):
        t = ModelTrainer("logistic_regression")
        assert t.params == {}

    def test_metadata_none_before_training(self):
        t = ModelTrainer("logistic_regression")
        assert t.metadata is None


# ------------------------------------------------------------------
# Training — each model type
# ------------------------------------------------------------------


class TestTrainLogisticRegression:
    def test_trains_without_error(self, dataset):
        X, y = dataset
        t = ModelTrainer("logistic_regression", {"max_iter": 1000})
        result = t.train(X, y)
        assert result is t  # returns self

    def test_can_predict_after_training(self, dataset):
        X, y = dataset
        t = ModelTrainer("logistic_regression", {"max_iter": 1000}).train(X, y)
        preds = t.predict(X)
        assert preds.shape == y.shape
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba_shape(self, dataset):
        X, y = dataset
        t = ModelTrainer("logistic_regression", {"max_iter": 1000}).train(X, y)
        proba = t.predict_proba(X)
        assert proba.shape == (len(y), 2)
        assert (proba >= 0).all() and (proba <= 1).all()


class TestTrainRandomForest:
    def test_trains_without_error(self, dataset):
        X, y = dataset
        t = ModelTrainer("random_forest", {"n_estimators": 10})
        result = t.train(X, y)
        assert result is t

    def test_can_predict_after_training(self, dataset):
        X, y = dataset
        t = ModelTrainer("random_forest", {"n_estimators": 10}).train(X, y)
        preds = t.predict(X)
        assert preds.shape == y.shape

    def test_predict_proba_shape(self, dataset):
        X, y = dataset
        t = ModelTrainer("random_forest", {"n_estimators": 10}).train(X, y)
        proba = t.predict_proba(X)
        assert proba.shape == (len(y), 2)


class TestTrainXGBoost:
    def test_trains_without_error(self, dataset):
        X, y = dataset
        t = ModelTrainer("xgboost", {"n_estimators": 10, "max_depth": 3})
        result = t.train(X, y)
        assert result is t

    def test_can_predict_after_training(self, dataset):
        X, y = dataset
        t = ModelTrainer("xgboost", {"n_estimators": 10, "max_depth": 3}).train(X, y)
        preds = t.predict(X)
        assert preds.shape == y.shape

    def test_predict_proba_shape(self, dataset):
        X, y = dataset
        t = ModelTrainer("xgboost", {"n_estimators": 10, "max_depth": 3}).train(X, y)
        proba = t.predict_proba(X)
        assert proba.shape == (len(y), 2)


# ------------------------------------------------------------------
# Class weight handling
# ------------------------------------------------------------------


class TestClassWeight:
    def test_sklearn_class_weight(self, dataset):
        X, y = dataset
        weights = {0: 1.0, 1: 10.0}
        t = ModelTrainer("logistic_regression", {"max_iter": 1000})
        t.train(X, y, class_weight=weights)
        # Should complete without error
        assert t.metadata is not None

    def test_random_forest_class_weight(self, dataset):
        X, y = dataset
        weights = {0: 1.0, 1: 5.0}
        t = ModelTrainer("random_forest", {"n_estimators": 10})
        t.train(X, y, class_weight=weights)
        assert t.metadata is not None

    def test_xgboost_scale_pos_weight(self, dataset):
        X, y = dataset
        weights = {0: 1.0, 1: 10.0}
        t = ModelTrainer("xgboost", {"n_estimators": 10, "max_depth": 3})
        t.train(X, y, class_weight=weights)
        assert t.metadata is not None
        # scale_pos_weight should appear in metadata params
        assert "scale_pos_weight" in t.metadata.params

    def test_no_class_weight(self, dataset):
        X, y = dataset
        t = ModelTrainer("logistic_regression", {"max_iter": 1000})
        t.train(X, y, class_weight=None)
        assert t.metadata is not None


# ------------------------------------------------------------------
# Training metadata
# ------------------------------------------------------------------


class TestTrainingMetadata:
    def test_metadata_populated(self, dataset):
        X, y = dataset
        t = ModelTrainer("logistic_regression", {"max_iter": 1000}).train(X, y)
        assert t.metadata is not None
        assert isinstance(t.metadata, TrainingMetadata)

    def test_metadata_model_type(self, dataset):
        X, y = dataset
        t = ModelTrainer("random_forest", {"n_estimators": 10}).train(X, y)
        assert t.metadata.model_type == "random_forest"

    def test_metadata_n_samples(self, dataset):
        X, y = dataset
        t = ModelTrainer("logistic_regression", {"max_iter": 1000}).train(X, y)
        assert t.metadata.n_samples == X.shape[0]

    def test_metadata_n_features(self, dataset):
        X, y = dataset
        t = ModelTrainer("logistic_regression", {"max_iter": 1000}).train(X, y)
        assert t.metadata.n_features == X.shape[1]

    def test_metadata_class_distribution(self, dataset):
        X, y = dataset
        t = ModelTrainer("logistic_regression", {"max_iter": 1000}).train(X, y)
        dist = t.metadata.class_distribution
        assert 0 in dist
        assert 1 in dist
        assert dist[0] + dist[1] == len(y)

    def test_metadata_trained_at(self, dataset):
        X, y = dataset
        t = ModelTrainer("logistic_regression", {"max_iter": 1000}).train(X, y)
        assert t.metadata.trained_at is not None
        assert len(t.metadata.trained_at) > 0
        # Should be valid ISO timestamp
        assert "T" in t.metadata.trained_at

    def test_metadata_params(self, dataset):
        X, y = dataset
        t = ModelTrainer("logistic_regression", {"C": 0.5, "max_iter": 500}).train(X, y)
        assert t.metadata.params["C"] == 0.5
        assert t.metadata.params["max_iter"] == 500


# ------------------------------------------------------------------
# Cross-validation
# ------------------------------------------------------------------


class TestCrossValidation:
    def test_returns_cv_results(self, small_dataset):
        X, y = small_dataset
        t = ModelTrainer("logistic_regression", {"max_iter": 1000})
        results = t.cross_validate(X, y, cv=3)
        assert isinstance(results, CVResults)

    def test_correct_number_of_folds(self, small_dataset):
        X, y = small_dataset
        t = ModelTrainer("logistic_regression", {"max_iter": 1000})
        results = t.cross_validate(X, y, cv=3)
        assert results.n_folds == 3

    def test_five_folds_default(self, dataset):
        X, y = dataset
        t = ModelTrainer("logistic_regression", {"max_iter": 1000})
        results = t.cross_validate(X, y, cv=5)
        assert results.n_folds == 5

    def test_fold_metrics_have_valid_precision(self, small_dataset):
        X, y = small_dataset
        t = ModelTrainer("logistic_regression", {"max_iter": 1000})
        results = t.cross_validate(X, y, cv=3)
        for fold in results.folds:
            assert 0 <= fold.precision <= 1

    def test_fold_metrics_have_valid_recall(self, small_dataset):
        X, y = small_dataset
        t = ModelTrainer("logistic_regression", {"max_iter": 1000})
        results = t.cross_validate(X, y, cv=3)
        for fold in results.folds:
            assert 0 <= fold.recall <= 1

    def test_fold_metrics_have_valid_f1(self, small_dataset):
        X, y = small_dataset
        t = ModelTrainer("logistic_regression", {"max_iter": 1000})
        results = t.cross_validate(X, y, cv=3)
        for fold in results.folds:
            assert 0 <= fold.f1 <= 1

    def test_fold_metrics_have_valid_auc_roc(self, small_dataset):
        X, y = small_dataset
        t = ModelTrainer("logistic_regression", {"max_iter": 1000})
        results = t.cross_validate(X, y, cv=3)
        for fold in results.folds:
            assert 0 <= fold.auc_roc <= 1

    def test_mean_metrics(self, small_dataset):
        X, y = small_dataset
        t = ModelTrainer("logistic_regression", {"max_iter": 1000})
        results = t.cross_validate(X, y, cv=3)
        assert 0 <= results.mean_precision <= 1
        assert 0 <= results.mean_recall <= 1
        assert 0 <= results.mean_f1 <= 1
        assert 0 <= results.mean_auc_roc <= 1

    def test_summary_dict(self, small_dataset):
        X, y = small_dataset
        t = ModelTrainer("logistic_regression", {"max_iter": 1000})
        results = t.cross_validate(X, y, cv=3)
        s = results.summary()
        assert set(s.keys()) == {
            "mean_precision",
            "mean_recall",
            "mean_f1",
            "mean_auc_roc",
        }

    def test_cv_random_forest(self, small_dataset):
        X, y = small_dataset
        t = ModelTrainer("random_forest", {"n_estimators": 10})
        results = t.cross_validate(X, y, cv=3)
        assert results.n_folds == 3
        assert results.mean_auc_roc > 0

    def test_cv_xgboost(self, small_dataset):
        X, y = small_dataset
        t = ModelTrainer("xgboost", {"n_estimators": 10, "max_depth": 3})
        results = t.cross_validate(X, y, cv=3)
        assert results.n_folds == 3
        assert results.mean_auc_roc > 0

    def test_fold_indices_are_sequential(self, small_dataset):
        X, y = small_dataset
        t = ModelTrainer("logistic_regression", {"max_iter": 1000})
        results = t.cross_validate(X, y, cv=3)
        assert [f.fold for f in results.folds] == [0, 1, 2]


# ------------------------------------------------------------------
# CVResults / FoldMetrics dataclasses
# ------------------------------------------------------------------


class TestCVResultsDataclass:
    def test_empty_results(self):
        r = CVResults()
        assert r.n_folds == 0

    def test_mean_of_single_fold(self):
        r = CVResults(
            folds=[FoldMetrics(fold=0, precision=0.8, recall=0.6, f1=0.7, auc_roc=0.9)]
        )
        assert r.mean_precision == pytest.approx(0.8)
        assert r.mean_recall == pytest.approx(0.6)
        assert r.mean_f1 == pytest.approx(0.7)
        assert r.mean_auc_roc == pytest.approx(0.9)

    def test_mean_of_multiple_folds(self):
        r = CVResults(
            folds=[
                FoldMetrics(fold=0, precision=0.8, recall=0.6, f1=0.7, auc_roc=0.9),
                FoldMetrics(fold=1, precision=0.6, recall=0.8, f1=0.7, auc_roc=0.85),
            ]
        )
        assert r.mean_precision == pytest.approx(0.7)
        assert r.mean_recall == pytest.approx(0.7)
        assert r.mean_f1 == pytest.approx(0.7)
        assert r.mean_auc_roc == pytest.approx(0.875)


# ------------------------------------------------------------------
# Config file
# ------------------------------------------------------------------


class TestModelParamsConfig:
    @pytest.fixture(autouse=True)
    def _require_config(self):
        if not MODEL_PARAMS_PATH.exists():
            pytest.skip("model_params.yaml not found")

    def test_file_exists(self):
        assert MODEL_PARAMS_PATH.exists()

    def test_valid_yaml(self):
        with open(MODEL_PARAMS_PATH) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)

    def test_contains_all_model_types(self):
        with open(MODEL_PARAMS_PATH) as f:
            data = yaml.safe_load(f)
        assert "logistic_regression" in data
        assert "random_forest" in data
        assert "xgboost" in data

    def test_logistic_regression_params(self):
        with open(MODEL_PARAMS_PATH) as f:
            data = yaml.safe_load(f)
        lr = data["logistic_regression"]
        assert "C" in lr
        assert "max_iter" in lr
        assert lr["class_weight"] == "balanced"

    def test_random_forest_params(self):
        with open(MODEL_PARAMS_PATH) as f:
            data = yaml.safe_load(f)
        rf = data["random_forest"]
        assert "n_estimators" in rf
        assert "max_depth" in rf
        assert rf["class_weight"] == "balanced"

    def test_xgboost_params(self):
        with open(MODEL_PARAMS_PATH) as f:
            data = yaml.safe_load(f)
        xgb = data["xgboost"]
        assert "n_estimators" in xgb
        assert "max_depth" in xgb
        assert "learning_rate" in xgb
        assert "scale_pos_weight" in xgb
