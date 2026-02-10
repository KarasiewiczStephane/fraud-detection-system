"""Tests for the model registry."""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.models.evaluator import EvaluationMetrics
from src.models.registry import ModelRegistry


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _dummy_model():
    m = LogisticRegression(max_iter=200)
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (50, 3))
    y = (X[:, 0] > 0).astype(int)
    m.fit(X, y)
    return m


def _dummy_metrics():
    return EvaluationMetrics(
        precision=0.85,
        recall=0.80,
        f1=0.82,
        auc_roc=0.92,
        auc_pr=0.88,
        confusion_matrix=[[40, 5], [10, 45]],
    )


def _dummy_params():
    return {"max_iter": 200, "C": 1.0}


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(base_path=tmp_path / "models")


@pytest.fixture
def model():
    return _dummy_model()


@pytest.fixture
def metrics():
    return _dummy_metrics()


@pytest.fixture
def params():
    return _dummy_params()


# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------


class TestSave:
    def test_returns_version_string(self, registry, model, metrics, params):
        version = registry.save(model, "lr", metrics, params, threshold=0.5)
        assert isinstance(version, str)
        assert len(version) == 15  # YYYYMMDD_HHMMSS

    def test_creates_model_file(self, registry, model, metrics, params):
        version = registry.save(model, "lr", metrics, params, threshold=0.5)
        model_path = registry.base_path / "lr" / version / "model.joblib"
        assert model_path.exists()

    def test_creates_metadata_file(self, registry, model, metrics, params):
        version = registry.save(model, "lr", metrics, params, threshold=0.5)
        meta_path = registry.base_path / "lr" / version / "metadata.json"
        assert meta_path.exists()

    def test_metadata_is_valid_json(self, registry, model, metrics, params):
        version = registry.save(model, "lr", metrics, params, threshold=0.5)
        meta_path = registry.base_path / "lr" / version / "metadata.json"
        data = json.loads(meta_path.read_text())
        assert isinstance(data, dict)

    def test_metadata_contains_name(self, registry, model, metrics, params):
        version = registry.save(model, "lr", metrics, params, threshold=0.5)
        meta = json.loads(
            (registry.base_path / "lr" / version / "metadata.json").read_text()
        )
        assert meta["name"] == "lr"

    def test_metadata_contains_version(self, registry, model, metrics, params):
        version = registry.save(model, "lr", metrics, params, threshold=0.5)
        meta = json.loads(
            (registry.base_path / "lr" / version / "metadata.json").read_text()
        )
        assert meta["version"] == version

    def test_metadata_contains_metrics(self, registry, model, metrics, params):
        version = registry.save(model, "lr", metrics, params, threshold=0.5)
        meta = json.loads(
            (registry.base_path / "lr" / version / "metadata.json").read_text()
        )
        assert "metrics" in meta
        assert meta["metrics"]["precision"] == 0.85
        assert meta["metrics"]["auc_roc"] == 0.92

    def test_metadata_contains_params(self, registry, model, metrics, params):
        version = registry.save(model, "lr", metrics, params, threshold=0.5)
        meta = json.loads(
            (registry.base_path / "lr" / version / "metadata.json").read_text()
        )
        assert meta["params"] == params

    def test_metadata_contains_threshold(self, registry, model, metrics, params):
        version = registry.save(model, "lr", metrics, params, threshold=0.42)
        meta = json.loads(
            (registry.base_path / "lr" / version / "metadata.json").read_text()
        )
        assert meta["threshold"] == 0.42

    def test_metadata_contains_created_at(self, registry, model, metrics, params):
        version = registry.save(model, "lr", metrics, params, threshold=0.5)
        meta = json.loads(
            (registry.base_path / "lr" / version / "metadata.json").read_text()
        )
        assert "created_at" in meta
        assert "T" in meta["created_at"]  # ISO format

    def test_creates_directory_tree(self, registry, model, metrics, params):
        assert not registry.base_path.exists()
        registry.save(model, "lr", metrics, params, threshold=0.5)
        assert registry.base_path.exists()


# ------------------------------------------------------------------
# Load
# ------------------------------------------------------------------


class TestLoad:
    def test_roundtrip(self, registry, model, metrics, params):
        version = registry.save(model, "lr", metrics, params, threshold=0.5)
        loaded_model, loaded_meta = registry.load("lr", version)
        # Model should produce the same predictions
        rng = np.random.default_rng(99)
        X_test = rng.normal(0, 1, (10, 3))
        np.testing.assert_array_equal(
            model.predict(X_test), loaded_model.predict(X_test)
        )

    def test_metadata_roundtrip(self, registry, model, metrics, params):
        version = registry.save(model, "lr", metrics, params, threshold=0.5)
        _, meta = registry.load("lr", version)
        assert meta["name"] == "lr"
        assert meta["version"] == version
        assert meta["params"] == params
        assert meta["threshold"] == 0.5

    def test_load_latest(self, registry, model, metrics, params):
        registry.save(model, "lr", metrics, params, threshold=0.5)
        # Save a second version (force different timestamp by manipulating directory)
        v2_dir = registry.base_path / "lr" / "99999999_999999"
        v2_dir.mkdir(parents=True)
        joblib.dump(model, v2_dir / "model.joblib")
        meta = {
            "name": "lr",
            "version": "99999999_999999",
            "metrics": metrics.to_dict(),
            "params": params,
            "threshold": 0.6,
            "created_at": "2099-01-01T00:00:00",
        }
        (v2_dir / "metadata.json").write_text(json.dumps(meta))

        _, loaded_meta = registry.load("lr", "latest")
        assert loaded_meta["version"] == "99999999_999999"
        assert loaded_meta["threshold"] == 0.6

    def test_load_nonexistent_model_raises(self, registry):
        with pytest.raises(FileNotFoundError):
            registry.load("nonexistent", "v1")

    def test_load_latest_no_versions_raises(self, registry, model, metrics, params):
        (registry.base_path / "empty_model").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="No versions found"):
            registry.load("empty_model", "latest")

    def test_load_specific_version(self, registry, model, metrics, params):
        v1 = registry.save(model, "lr", metrics, params, threshold=0.3)
        # Create second version
        time.sleep(1.1)  # ensure different timestamp
        v2 = registry.save(model, "lr", metrics, params, threshold=0.7)

        _, meta_v1 = registry.load("lr", v1)
        _, meta_v2 = registry.load("lr", v2)
        assert meta_v1["threshold"] == 0.3
        assert meta_v2["threshold"] == 0.7

    def test_loaded_model_can_predict(self, registry, model, metrics, params):
        version = registry.save(model, "lr", metrics, params, threshold=0.5)
        loaded_model, _ = registry.load("lr", version)
        rng = np.random.default_rng(42)
        X_test = rng.normal(0, 1, (20, 3))
        preds = loaded_model.predict(X_test)
        assert preds.shape == (20,)

    def test_loaded_model_predict_proba(self, registry, model, metrics, params):
        version = registry.save(model, "lr", metrics, params, threshold=0.5)
        loaded_model, _ = registry.load("lr", version)
        rng = np.random.default_rng(42)
        X_test = rng.normal(0, 1, (20, 3))
        proba = loaded_model.predict_proba(X_test)
        assert proba.shape == (20, 2)


# ------------------------------------------------------------------
# Listing
# ------------------------------------------------------------------


class TestListing:
    def test_list_models_empty(self, registry):
        assert registry.list_models() == []

    def test_list_models(self, registry, model, metrics, params):
        registry.save(model, "lr", metrics, params, threshold=0.5)
        registry.save(model, "rf", metrics, params, threshold=0.5)
        models = registry.list_models()
        assert "lr" in models
        assert "rf" in models

    def test_list_versions_empty(self, registry):
        assert registry.list_versions("nonexistent") == []

    def test_list_versions(self, registry, model, metrics, params):
        v1 = registry.save(model, "lr", metrics, params, threshold=0.5)
        time.sleep(1.1)
        v2 = registry.save(model, "lr", metrics, params, threshold=0.5)
        versions = registry.list_versions("lr")
        assert len(versions) == 2
        assert versions[0] == v1
        assert versions[1] == v2

    def test_list_versions_sorted(self, registry, model, metrics, params):
        # Manually create versions out of order
        for v in ["20240103_000000", "20240101_000000", "20240102_000000"]:
            vdir = registry.base_path / "lr" / v
            vdir.mkdir(parents=True)
            joblib.dump(model, vdir / "model.joblib")
            (vdir / "metadata.json").write_text(json.dumps({"version": v}))

        versions = registry.list_versions("lr")
        assert versions == ["20240101_000000", "20240102_000000", "20240103_000000"]

    def test_list_versions_ignores_dirs_without_metadata(
        self, registry, model, metrics, params
    ):
        registry.save(model, "lr", metrics, params, threshold=0.5)
        (registry.base_path / "lr" / "stray_dir").mkdir()
        versions = registry.list_versions("lr")
        assert len(versions) == 1
