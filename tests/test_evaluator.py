"""Tests for model evaluation and comparison reports."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.evaluator import EvaluationMetrics, ModelEvaluator


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_dataset(n: int = 300, fraud_ratio: float = 0.15, seed: int = 42):
    rng = np.random.default_rng(seed)
    n_fraud = max(5, int(n * fraud_ratio))
    n_normal = n - n_fraud
    X = np.vstack([rng.normal(0, 1, (n_normal, 5)), rng.normal(2, 1, (n_fraud, 5))])
    y = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)]).astype(int)
    idx = rng.permutation(n)
    return X[idx], y[idx]


def _trained_model(X, y):
    m = LogisticRegression(max_iter=1000)
    m.fit(X, y)
    return m


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def dataset():
    return _make_dataset()


@pytest.fixture
def model_and_data(dataset):
    X, y = dataset
    return _trained_model(X, y), X, y


@pytest.fixture
def evaluator():
    return ModelEvaluator()


# ------------------------------------------------------------------
# EvaluationMetrics dataclass
# ------------------------------------------------------------------


class TestEvaluationMetrics:
    def test_to_dict(self):
        m = EvaluationMetrics(
            precision=0.8,
            recall=0.7,
            f1=0.75,
            auc_roc=0.9,
            auc_pr=0.85,
            confusion_matrix=[[90, 10], [5, 45]],
        )
        d = m.to_dict()
        assert d["precision"] == 0.8
        assert d["auc_pr"] == 0.85
        assert d["confusion_matrix"] == [[90, 10], [5, 45]]

    def test_all_fields_present(self):
        m = EvaluationMetrics(
            precision=0.5,
            recall=0.5,
            f1=0.5,
            auc_roc=0.5,
            auc_pr=0.5,
            confusion_matrix=[[1, 0], [0, 1]],
        )
        d = m.to_dict()
        expected_keys = {
            "precision",
            "recall",
            "f1",
            "auc_roc",
            "auc_pr",
            "confusion_matrix",
        }
        assert set(d.keys()) == expected_keys


# ------------------------------------------------------------------
# ModelEvaluator.evaluate
# ------------------------------------------------------------------


class TestEvaluate:
    def test_returns_evaluation_metrics(self, evaluator, model_and_data):
        model, X, y = model_and_data
        result = evaluator.evaluate(model, X, y)
        assert isinstance(result, EvaluationMetrics)

    def test_precision_matches_sklearn(self, evaluator, model_and_data):
        model, X, y = model_and_data
        result = evaluator.evaluate(model, X, y, threshold=0.5)
        y_pred = (model.predict_proba(X)[:, 1] >= 0.5).astype(int)
        expected = precision_score(y, y_pred, zero_division=0)
        assert result.precision == pytest.approx(expected)

    def test_recall_matches_sklearn(self, evaluator, model_and_data):
        model, X, y = model_and_data
        result = evaluator.evaluate(model, X, y, threshold=0.5)
        y_pred = (model.predict_proba(X)[:, 1] >= 0.5).astype(int)
        expected = recall_score(y, y_pred, zero_division=0)
        assert result.recall == pytest.approx(expected)

    def test_f1_matches_sklearn(self, evaluator, model_and_data):
        model, X, y = model_and_data
        result = evaluator.evaluate(model, X, y, threshold=0.5)
        y_pred = (model.predict_proba(X)[:, 1] >= 0.5).astype(int)
        expected = f1_score(y, y_pred, zero_division=0)
        assert result.f1 == pytest.approx(expected)

    def test_auc_roc_matches_sklearn(self, evaluator, model_and_data):
        model, X, y = model_and_data
        result = evaluator.evaluate(model, X, y)
        y_proba = model.predict_proba(X)[:, 1]
        expected = roc_auc_score(y, y_proba)
        assert result.auc_roc == pytest.approx(expected)

    def test_auc_pr_matches_sklearn(self, evaluator, model_and_data):
        model, X, y = model_and_data
        result = evaluator.evaluate(model, X, y)
        y_proba = model.predict_proba(X)[:, 1]
        expected = average_precision_score(y, y_proba)
        assert result.auc_pr == pytest.approx(expected)

    def test_confusion_matrix_matches_sklearn(self, evaluator, model_and_data):
        model, X, y = model_and_data
        result = evaluator.evaluate(model, X, y, threshold=0.5)
        y_pred = (model.predict_proba(X)[:, 1] >= 0.5).astype(int)
        expected = confusion_matrix(y, y_pred).tolist()
        assert result.confusion_matrix == expected

    def test_confusion_matrix_shape(self, evaluator, model_and_data):
        model, X, y = model_and_data
        result = evaluator.evaluate(model, X, y)
        cm = result.confusion_matrix
        assert len(cm) == 2
        assert len(cm[0]) == 2
        assert len(cm[1]) == 2

    def test_custom_threshold(self, evaluator, model_and_data):
        model, X, y = model_and_data
        result_low = evaluator.evaluate(model, X, y, threshold=0.1)
        result_high = evaluator.evaluate(model, X, y, threshold=0.9)
        # Lower threshold → more positives → higher recall (usually)
        assert result_low.recall >= result_high.recall

    def test_metrics_in_valid_range(self, evaluator, model_and_data):
        model, X, y = model_and_data
        result = evaluator.evaluate(model, X, y)
        assert 0 <= result.precision <= 1
        assert 0 <= result.recall <= 1
        assert 0 <= result.f1 <= 1
        assert 0 <= result.auc_roc <= 1
        assert 0 <= result.auc_pr <= 1


# ------------------------------------------------------------------
# ModelEvaluator.compare_models
# ------------------------------------------------------------------


class TestCompareModels:
    def _make_results(self):
        return {
            "model_a": EvaluationMetrics(
                precision=0.8,
                recall=0.7,
                f1=0.75,
                auc_roc=0.9,
                auc_pr=0.85,
                confusion_matrix=[[90, 10], [15, 35]],
            ),
            "model_b": EvaluationMetrics(
                precision=0.7,
                recall=0.9,
                f1=0.79,
                auc_roc=0.88,
                auc_pr=0.82,
                confusion_matrix=[[80, 20], [5, 45]],
            ),
            "model_c": EvaluationMetrics(
                precision=0.6,
                recall=0.6,
                f1=0.6,
                auc_roc=0.75,
                auc_pr=0.70,
                confusion_matrix=[[85, 15], [20, 30]],
            ),
        }

    def test_returns_string(self, evaluator):
        table = evaluator.compare_models(self._make_results())
        assert isinstance(table, str)

    def test_contains_all_model_names(self, evaluator):
        table = evaluator.compare_models(self._make_results())
        assert "model_a" in table
        assert "model_b" in table
        assert "model_c" in table

    def test_contains_all_metric_names(self, evaluator):
        table = evaluator.compare_models(self._make_results())
        for metric in ("precision", "recall", "f1", "auc_roc", "auc_pr"):
            assert metric in table

    def test_markdown_table_has_header_separator(self, evaluator):
        table = evaluator.compare_models(self._make_results())
        lines = table.strip().split("\n")
        assert len(lines) >= 3  # header + separator + at least 1 row
        assert "---" in lines[1]

    def test_best_values_are_bold(self, evaluator):
        table = evaluator.compare_models(self._make_results())
        # model_a has best precision (0.8) → should be bolded
        assert "**0.8000**" in table
        # model_b has best recall (0.9)
        assert "**0.9000**" in table

    def test_empty_results(self, evaluator):
        table = evaluator.compare_models({})
        assert table == ""

    def test_single_model(self, evaluator):
        results = {
            "only_model": EvaluationMetrics(
                precision=0.5,
                recall=0.5,
                f1=0.5,
                auc_roc=0.5,
                auc_pr=0.5,
                confusion_matrix=[[1, 0], [0, 1]],
            ),
        }
        table = evaluator.compare_models(results)
        assert "only_model" in table
        # Single model is always best → all values bold
        assert table.count("**") >= 10  # 5 metrics * 2 asterisks each

    def test_row_count(self, evaluator):
        results = self._make_results()
        table = evaluator.compare_models(results)
        lines = table.strip().split("\n")
        # header + separator + 3 model rows = 5
        assert len(lines) == 5
