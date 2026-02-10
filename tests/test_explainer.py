"""Tests for SHAP-based model explainability."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.models.explainer import LocalExplanation, SHAPExplainer


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_dataset(n: int = 200, n_features: int = 10, seed: int = 42):
    rng = np.random.default_rng(seed)
    n_fraud = max(10, int(n * 0.15))
    n_normal = n - n_fraud
    X = np.vstack(
        [
            rng.normal(0, 1, (n_normal, n_features)),
            rng.normal(2, 1, (n_fraud, n_features)),
        ]
    )
    y = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)]).astype(int)
    idx = rng.permutation(n)
    return X[idx], y[idx]


def _feature_names(n: int = 10):
    return [f"feat_{i}" for i in range(n)]


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def dataset():
    return _make_dataset()


@pytest.fixture
def xgb_model(dataset):
    X, y = dataset
    m = XGBClassifier(n_estimators=10, max_depth=3)
    m.fit(X, y)
    return m


@pytest.fixture
def rf_model(dataset):
    X, y = dataset
    m = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    m.fit(X, y)
    return m


@pytest.fixture
def lr_model(dataset):
    X, y = dataset
    m = LogisticRegression(max_iter=500)
    m.fit(X, y)
    return m


@pytest.fixture
def xgb_explainer(xgb_model):
    return SHAPExplainer(xgb_model, model_type="xgboost")


@pytest.fixture
def rf_explainer(rf_model):
    return SHAPExplainer(rf_model, model_type="random_forest")


@pytest.fixture
def lr_explainer(lr_model, dataset):
    X, _ = dataset
    bg = X[:20]
    return SHAPExplainer(lr_model, model_type="logistic_regression", background_data=bg)


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------


class TestInit:
    def test_xgboost_tree_explainer(self, xgb_model):
        exp = SHAPExplainer(xgb_model, model_type="xgboost")
        assert exp.model_type == "xgboost"
        assert exp.explainer is not None

    def test_random_forest_tree_explainer(self, rf_model):
        exp = SHAPExplainer(rf_model, model_type="random_forest")
        assert exp.model_type == "random_forest"

    def test_logistic_regression_kernel_explainer(self, lr_model, dataset):
        X, _ = dataset
        exp = SHAPExplainer(
            lr_model, model_type="logistic_regression", background_data=X[:20]
        )
        assert exp.model_type == "logistic_regression"

    def test_kernel_explainer_requires_background(self, lr_model):
        with pytest.raises(ValueError, match="background_data is required"):
            SHAPExplainer(lr_model, model_type="logistic_regression")


# ------------------------------------------------------------------
# compute_shap_values
# ------------------------------------------------------------------


class TestComputeShapValues:
    def test_xgboost_shape(self, xgb_explainer, dataset):
        X, _ = dataset
        sv = xgb_explainer.compute_shap_values(X[:5])
        assert sv.shape == (5, X.shape[1])

    def test_random_forest_shape(self, rf_explainer, dataset):
        X, _ = dataset
        sv = rf_explainer.compute_shap_values(X[:5])
        assert sv.shape == (5, X.shape[1])

    def test_kernel_explainer_shape(self, lr_explainer, dataset):
        X, _ = dataset
        sv = lr_explainer.compute_shap_values(X[:3])
        assert sv.shape == (3, X.shape[1])

    def test_returns_numpy_array(self, xgb_explainer, dataset):
        X, _ = dataset
        sv = xgb_explainer.compute_shap_values(X[:5])
        assert isinstance(sv, np.ndarray)

    def test_single_sample(self, xgb_explainer, dataset):
        X, _ = dataset
        sv = xgb_explainer.compute_shap_values(X[:1])
        assert sv.shape == (1, X.shape[1])


# ------------------------------------------------------------------
# global_importance
# ------------------------------------------------------------------


class TestGlobalImportance:
    def test_returns_dict(self, xgb_explainer, dataset):
        X, _ = dataset
        names = _feature_names(X.shape[1])
        result = xgb_explainer.global_importance(X[:30], names)
        assert isinstance(result, dict)

    def test_top_15_default(self, xgb_explainer, dataset):
        X, _ = dataset
        names = _feature_names(X.shape[1])
        result = xgb_explainer.global_importance(X[:30], names)
        assert len(result) <= 15

    def test_fewer_features_than_top_n(self, xgb_explainer, dataset):
        """With 10 features, top_n=15 should return all 10."""
        X, _ = dataset
        names = _feature_names(X.shape[1])
        result = xgb_explainer.global_importance(X[:30], names, top_n=15)
        assert len(result) == X.shape[1]  # 10

    def test_custom_top_n(self, xgb_explainer, dataset):
        X, _ = dataset
        names = _feature_names(X.shape[1])
        result = xgb_explainer.global_importance(X[:30], names, top_n=3)
        assert len(result) == 3

    def test_values_are_non_negative(self, xgb_explainer, dataset):
        X, _ = dataset
        names = _feature_names(X.shape[1])
        result = xgb_explainer.global_importance(X[:30], names)
        for val in result.values():
            assert val >= 0

    def test_sorted_descending(self, xgb_explainer, dataset):
        X, _ = dataset
        names = _feature_names(X.shape[1])
        result = xgb_explainer.global_importance(X[:30], names)
        vals = list(result.values())
        assert vals == sorted(vals, reverse=True)

    def test_feature_names_match(self, xgb_explainer, dataset):
        X, _ = dataset
        names = _feature_names(X.shape[1])
        result = xgb_explainer.global_importance(X[:30], names)
        for name in result:
            assert name in names

    def test_random_forest(self, rf_explainer, dataset):
        X, _ = dataset
        names = _feature_names(X.shape[1])
        result = rf_explainer.global_importance(X[:30], names)
        assert isinstance(result, dict)
        assert len(result) > 0


# ------------------------------------------------------------------
# local_explanation
# ------------------------------------------------------------------


class TestLocalExplanation:
    def test_returns_local_explanation(self, xgb_explainer, dataset):
        X, _ = dataset
        names = _feature_names(X.shape[1])
        result = xgb_explainer.local_explanation(X[0], names)
        assert isinstance(result, LocalExplanation)

    def test_has_base_value(self, xgb_explainer, dataset):
        X, _ = dataset
        names = _feature_names(X.shape[1])
        result = xgb_explainer.local_explanation(X[0], names)
        assert isinstance(result.base_value, float)

    def test_has_prediction(self, xgb_explainer, dataset):
        X, _ = dataset
        names = _feature_names(X.shape[1])
        result = xgb_explainer.local_explanation(X[0], names)
        assert isinstance(result.prediction, float)

    def test_contributions_top_10_default(self, xgb_explainer, dataset):
        X, _ = dataset
        names = _feature_names(X.shape[1])
        result = xgb_explainer.local_explanation(X[0], names)
        assert len(result.contributions) <= 10

    def test_contributions_have_required_keys(self, xgb_explainer, dataset):
        X, _ = dataset
        names = _feature_names(X.shape[1])
        result = xgb_explainer.local_explanation(X[0], names)
        for c in result.contributions:
            assert "feature" in c
            assert "value" in c
            assert "contribution" in c

    def test_contributions_sorted_by_abs_contribution(self, xgb_explainer, dataset):
        X, _ = dataset
        names = _feature_names(X.shape[1])
        result = xgb_explainer.local_explanation(X[0], names)
        abs_contribs = [abs(c["contribution"]) for c in result.contributions]
        assert abs_contribs == sorted(abs_contribs, reverse=True)

    def test_prediction_equals_base_plus_contributions(self, xgb_explainer, dataset):
        """prediction ≈ base_value + sum(all shap values)."""
        X, _ = dataset
        names = _feature_names(X.shape[1])
        # Use top_n == all features to get all contributions
        result = xgb_explainer.local_explanation(X[0], names, top_n=X.shape[1])
        total = result.base_value + sum(c["contribution"] for c in result.contributions)
        assert result.prediction == pytest.approx(total, abs=1e-4)

    def test_custom_top_n(self, xgb_explainer, dataset):
        X, _ = dataset
        names = _feature_names(X.shape[1])
        result = xgb_explainer.local_explanation(X[0], names, top_n=3)
        assert len(result.contributions) == 3

    def test_random_forest_local(self, rf_explainer, dataset):
        X, _ = dataset
        names = _feature_names(X.shape[1])
        result = rf_explainer.local_explanation(X[0], names)
        assert isinstance(result, LocalExplanation)
        assert len(result.contributions) > 0


# ------------------------------------------------------------------
# LocalExplanation dataclass
# ------------------------------------------------------------------


class TestLocalExplanationDataclass:
    def test_to_dict(self):
        le = LocalExplanation(
            base_value=0.5,
            contributions=[{"feature": "a", "value": 1.0, "contribution": 0.3}],
            prediction=0.8,
        )
        d = le.to_dict()
        assert d["base_value"] == 0.5
        assert d["prediction"] == 0.8
        assert len(d["contributions"]) == 1

    def test_from_dict(self):
        data = {
            "base_value": 0.5,
            "contributions": [{"feature": "a", "value": 1.0, "contribution": 0.3}],
            "prediction": 0.8,
        }
        le = LocalExplanation.from_dict(data)
        assert le.base_value == 0.5
        assert le.prediction == 0.8
        assert len(le.contributions) == 1

    def test_roundtrip(self):
        le = LocalExplanation(
            base_value=-0.1,
            contributions=[
                {"feature": "f1", "value": 2.0, "contribution": 0.5},
                {"feature": "f2", "value": -1.0, "contribution": -0.2},
            ],
            prediction=0.2,
        )
        d = le.to_dict()
        le2 = LocalExplanation.from_dict(d)
        assert le2.base_value == le.base_value
        assert le2.prediction == le.prediction
        assert le2.contributions == le.contributions


# ------------------------------------------------------------------
# Serialization (save / load explanation)
# ------------------------------------------------------------------


class TestSerialization:
    def test_save_returns_dict(self, xgb_explainer, dataset):
        X, _ = dataset
        names = _feature_names(X.shape[1])
        expl = xgb_explainer.local_explanation(X[0], names)
        payload = SHAPExplainer.save_explanation(expl, "txn_001")
        assert isinstance(payload, dict)

    def test_save_contains_transaction_id(self, xgb_explainer, dataset):
        X, _ = dataset
        names = _feature_names(X.shape[1])
        expl = xgb_explainer.local_explanation(X[0], names)
        payload = SHAPExplainer.save_explanation(expl, "txn_001")
        assert payload["transaction_id"] == "txn_001"

    def test_save_contains_explanation_fields(self, xgb_explainer, dataset):
        X, _ = dataset
        names = _feature_names(X.shape[1])
        expl = xgb_explainer.local_explanation(X[0], names)
        payload = SHAPExplainer.save_explanation(expl, "txn_001")
        assert "base_value" in payload
        assert "contributions" in payload
        assert "prediction" in payload

    def test_save_is_json_safe(self, xgb_explainer, dataset):
        import json

        X, _ = dataset
        names = _feature_names(X.shape[1])
        expl = xgb_explainer.local_explanation(X[0], names)
        payload = SHAPExplainer.save_explanation(expl, "txn_001")
        # Should not raise
        serialized = json.dumps(payload)
        assert isinstance(serialized, str)

    def test_load_roundtrip(self, xgb_explainer, dataset):
        X, _ = dataset
        names = _feature_names(X.shape[1])
        expl = xgb_explainer.local_explanation(X[0], names)
        payload = SHAPExplainer.save_explanation(expl, "txn_001")
        loaded = SHAPExplainer.load_explanation(payload)
        assert loaded.base_value == expl.base_value
        assert loaded.prediction == pytest.approx(expl.prediction, abs=1e-6)
        assert len(loaded.contributions) == len(expl.contributions)


# ------------------------------------------------------------------
# Plot generation
# ------------------------------------------------------------------


class TestPlotGeneration:
    def test_generates_beeswarm(self, xgb_explainer, dataset, tmp_path):
        X, _ = dataset
        names = _feature_names(X.shape[1])
        xgb_explainer.generate_plots(X[:20], names, tmp_path / "plots")
        beeswarm = tmp_path / "plots" / "shap_beeswarm.png"
        assert beeswarm.exists()
        assert beeswarm.stat().st_size > 0

    def test_generates_bar_plot(self, xgb_explainer, dataset, tmp_path):
        X, _ = dataset
        names = _feature_names(X.shape[1])
        xgb_explainer.generate_plots(X[:20], names, tmp_path / "plots")
        bar = tmp_path / "plots" / "shap_bar.png"
        assert bar.exists()
        assert bar.stat().st_size > 0

    def test_returns_paths(self, xgb_explainer, dataset, tmp_path):
        X, _ = dataset
        names = _feature_names(X.shape[1])
        paths = xgb_explainer.generate_plots(X[:20], names, tmp_path / "plots")
        assert len(paths) == 2
        for p in paths:
            assert isinstance(p, Path)
            assert p.exists()

    def test_creates_output_directory(self, xgb_explainer, dataset, tmp_path):
        X, _ = dataset
        names = _feature_names(X.shape[1])
        out = tmp_path / "nested" / "plots"
        xgb_explainer.generate_plots(X[:20], names, out)
        assert out.is_dir()

    def test_random_forest_plots(self, rf_explainer, dataset, tmp_path):
        X, _ = dataset
        names = _feature_names(X.shape[1])
        paths = rf_explainer.generate_plots(X[:20], names, tmp_path / "rf_plots")
        assert len(paths) == 2
        for p in paths:
            assert p.exists()
