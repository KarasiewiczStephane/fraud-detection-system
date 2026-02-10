"""Tests for the versioned feature store."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data.feature_store import FeatureStore, VersionInfo


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    return FeatureStore(base_path=tmp_path / "features")


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "Time": np.arange(100, dtype=float),
            "Amount": rng.uniform(1, 500, 100),
            "V1": rng.normal(0, 1, 100),
            "V2": rng.normal(0, 1, 100),
            "Class": rng.choice([0, 1], 100, p=[0.98, 0.02]),
        }
    )


@pytest.fixture
def small_df():
    return pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})


# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------


class TestSave:
    def test_returns_version_string(self, store, small_df):
        version = store.save(small_df)
        assert isinstance(version, str)
        assert len(version) > 0

    def test_explicit_version(self, store, small_df):
        version = store.save(small_df, version="v1_test")
        assert version == "v1_test"

    def test_auto_version_format(self, store, small_df):
        version = store.save(small_df)
        # Format: YYYYMMDD_HHMMSS
        assert len(version) == 15
        assert version[8] == "_"

    def test_creates_parquet_file(self, store, small_df):
        store.save(small_df, version="v1")
        parquet = store.base_path / "v1" / "features.parquet"
        assert parquet.exists()

    def test_creates_metadata_file(self, store, small_df):
        store.save(small_df, version="v1")
        meta = store.base_path / "v1" / "metadata.json"
        assert meta.exists()

    def test_metadata_content(self, store, small_df):
        store.save(small_df, version="v1")
        meta_path = store.base_path / "v1" / "metadata.json"
        meta = json.loads(meta_path.read_text())
        assert meta["version"] == "v1"
        assert meta["columns"] == ["a", "b"]
        assert meta["row_count"] == 3
        assert "created_at" in meta
        assert "dtypes" in meta

    def test_metadata_dtypes(self, store, small_df):
        store.save(small_df, version="v1")
        meta_path = store.base_path / "v1" / "metadata.json"
        meta = json.loads(meta_path.read_text())
        assert "int" in meta["dtypes"]["a"].lower()
        assert "float" in meta["dtypes"]["b"].lower()

    def test_creates_directory(self, store, small_df):
        assert not store.base_path.exists()
        store.save(small_df, version="v1")
        assert store.base_path.exists()
        assert (store.base_path / "v1").is_dir()


# ------------------------------------------------------------------
# Load
# ------------------------------------------------------------------


class TestLoad:
    def test_roundtrip_preserves_data(self, store, sample_df):
        store.save(sample_df, version="v1")
        loaded = store.load("v1")
        pd.testing.assert_frame_equal(loaded, sample_df)

    def test_roundtrip_preserves_dtypes(self, store, sample_df):
        store.save(sample_df, version="v1")
        loaded = store.load("v1")
        for col in sample_df.columns:
            assert loaded[col].dtype == sample_df[col].dtype, f"dtype mismatch on {col}"

    def test_roundtrip_small_df(self, store, small_df):
        store.save(small_df, version="v1")
        loaded = store.load("v1")
        pd.testing.assert_frame_equal(loaded, small_df)

    def test_load_latest(self, store, small_df):
        store.save(small_df, version="20240101_000000")
        df2 = small_df.copy()
        df2["c"] = [7, 8, 9]
        store.save(df2, version="20240102_000000")
        loaded = store.load("latest")
        pd.testing.assert_frame_equal(loaded, df2)

    def test_load_latest_picks_lexicographic_max(self, store, small_df):
        store.save(small_df, version="aaa")
        store.save(small_df, version="zzz")
        loaded = store.load("latest")
        # Should resolve to "zzz"
        pd.testing.assert_frame_equal(loaded, small_df)

    def test_load_nonexistent_version_raises(self, store):
        store.base_path.mkdir(parents=True, exist_ok=True)
        with pytest.raises(FileNotFoundError):
            store.load("nonexistent")

    def test_load_latest_empty_store_raises(self, store):
        store.base_path.mkdir(parents=True, exist_ok=True)
        with pytest.raises(FileNotFoundError, match="No versions found"):
            store.load("latest")

    def test_load_latest_no_store_dir_raises(self, store):
        with pytest.raises(FileNotFoundError):
            store.load("latest")

    def test_metadata_column_mismatch_raises(self, store, small_df):
        store.save(small_df, version="v1")
        # Tamper with metadata
        meta_path = store.base_path / "v1" / "metadata.json"
        meta = json.loads(meta_path.read_text())
        meta["columns"] = ["x", "y"]
        meta_path.write_text(json.dumps(meta))
        with pytest.raises(ValueError, match="Column mismatch"):
            store.load("v1")

    def test_metadata_row_count_mismatch_raises(self, store, small_df):
        store.save(small_df, version="v1")
        meta_path = store.base_path / "v1" / "metadata.json"
        meta = json.loads(meta_path.read_text())
        meta["row_count"] = 999
        meta_path.write_text(json.dumps(meta))
        with pytest.raises(ValueError, match="Row count mismatch"):
            store.load("v1")


# ------------------------------------------------------------------
# List versions
# ------------------------------------------------------------------


class TestListVersions:
    def test_empty_store(self, store):
        assert store.list_versions() == []

    def test_nonexistent_base_path(self, store):
        assert store.list_versions() == []

    def test_single_version(self, store, small_df):
        store.save(small_df, version="v1")
        versions = store.list_versions()
        assert len(versions) == 1
        assert versions[0].version == "v1"

    def test_multiple_versions_sorted(self, store, small_df):
        store.save(small_df, version="v2")
        store.save(small_df, version="v1")
        store.save(small_df, version="v3")
        versions = store.list_versions()
        assert [v.version for v in versions] == ["v1", "v2", "v3"]

    def test_version_info_fields(self, store, small_df):
        store.save(small_df, version="v1")
        info = store.list_versions()[0]
        assert isinstance(info, VersionInfo)
        assert info.version == "v1"
        assert info.columns == ["a", "b"]
        assert info.row_count == 3
        assert len(info.created_at) > 0
        assert isinstance(info.dtypes, dict)

    def test_ignores_dirs_without_metadata(self, store, small_df):
        store.save(small_df, version="v1")
        # Create a stray directory with no metadata
        (store.base_path / "stray_dir").mkdir()
        versions = store.list_versions()
        assert len(versions) == 1

    def test_multiple_saves_accumulate(self, store, small_df):
        for i in range(5):
            store.save(small_df, version=f"v{i:02d}")
        assert len(store.list_versions()) == 5
