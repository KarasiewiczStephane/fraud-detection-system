"""Tests for the dataset downloader and validation pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data.downloader import (
    EXPECTED_COL_COUNT,
    EXPECTED_COLUMNS,
    DatasetDownloader,
    ValidationResult,
)

SAMPLE_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "sample" / "sample_transactions.csv"
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def downloader():
    return DatasetDownloader()


@pytest.fixture
def sample_df():
    """Load the sample dataset."""
    return pd.read_csv(SAMPLE_PATH)


@pytest.fixture
def valid_csv(tmp_path, sample_df):
    """Write the sample data to a temp CSV and return its path."""
    path = tmp_path / "creditcard.csv"
    sample_df.to_csv(path, index=False)
    return path


@pytest.fixture
def corrupted_missing_columns(tmp_path):
    """CSV with missing columns."""
    df = pd.DataFrame({"Time": [0, 1], "Amount": [10, 20], "Class": [0, 1]})
    path = tmp_path / "bad_cols.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def corrupted_wrong_dtypes(tmp_path):
    """CSV with non-numeric column."""
    data = {"Time": [0, 1], "Amount": [10, 20], "Class": [0, 1]}
    for i in range(1, 29):
        data[f"V{i}"] = ["abc", "def"]  # strings instead of floats
    df = pd.DataFrame(data)
    path = tmp_path / "bad_dtypes.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def corrupted_null_values(tmp_path):
    """CSV with null values."""
    rng = np.random.default_rng(42)
    n = 100
    data = {
        "Time": np.arange(n, dtype=float),
        "Amount": rng.uniform(0, 100, n),
        "Class": np.zeros(n, dtype=int),
    }
    for i in range(1, 29):
        vals = rng.normal(0, 1, n)
        vals[0] = np.nan  # inject null
        data[f"V{i}"] = vals
    df = pd.DataFrame(data)
    path = tmp_path / "null_vals.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def corrupted_negative_amount(tmp_path):
    """CSV with negative Amount values."""
    rng = np.random.default_rng(42)
    n = 100
    data = {
        "Time": np.arange(n, dtype=float),
        "Amount": rng.uniform(-100, 100, n),
        "Class": np.zeros(n, dtype=int),
    }
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(0, 1, n)
    df = pd.DataFrame(data)
    path = tmp_path / "neg_amount.csv"
    df.to_csv(path, index=False)
    return path


# ------------------------------------------------------------------
# ValidationResult unit tests
# ------------------------------------------------------------------


class TestValidationResult:
    def test_empty_result_is_valid(self):
        result = ValidationResult()
        assert result.is_valid is True

    def test_all_pass(self):
        result = ValidationResult()
        result.add("check_a", True, "ok")
        result.add("check_b", True, "ok")
        assert result.is_valid is True

    def test_any_fail_makes_invalid(self):
        result = ValidationResult()
        result.add("check_a", True, "ok")
        result.add("check_b", False, "bad")
        assert result.is_valid is False

    def test_summary_format(self):
        result = ValidationResult()
        result.add("check_a", True, "passed")
        result.add("check_b", False, "failed")
        summary = result.summary()
        assert "[PASS] check_a: passed" in summary
        assert "[FAIL] check_b: failed" in summary


# ------------------------------------------------------------------
# Validation checks
# ------------------------------------------------------------------


class TestValidation:
    def test_file_not_found(self, downloader, tmp_path):
        result = downloader.validate(tmp_path / "nonexistent.csv")
        assert result.is_valid is False
        assert result.checks[0].name == "file_exists"
        assert result.checks[0].passed is False

    def test_valid_sample_passes_schema(self, downloader, valid_csv):
        result = downloader.validate(valid_csv)
        schema_check = next(c for c in result.checks if c.name == "schema_columns")
        assert schema_check.passed is True

    def test_valid_sample_column_count(self, downloader, valid_csv):
        result = downloader.validate(valid_csv)
        col_check = next(c for c in result.checks if c.name == "column_count")
        assert col_check.passed is True

    def test_valid_sample_dtypes(self, downloader, valid_csv):
        result = downloader.validate(valid_csv)
        dtype_check = next(c for c in result.checks if c.name == "column_dtypes")
        assert dtype_check.passed is True

    def test_valid_sample_no_missing(self, downloader, valid_csv):
        result = downloader.validate(valid_csv)
        missing_check = next(c for c in result.checks if c.name == "missing_values")
        assert missing_check.passed is True

    def test_valid_sample_amount_non_negative(self, downloader, valid_csv):
        result = downloader.validate(valid_csv)
        amount_check = next(c for c in result.checks if c.name == "amount_range")
        assert amount_check.passed is True

    def test_missing_columns_detected(self, downloader, corrupted_missing_columns):
        result = downloader.validate(corrupted_missing_columns)
        schema_check = next(c for c in result.checks if c.name == "schema_columns")
        assert schema_check.passed is False

    def test_wrong_dtypes_detected(self, downloader, corrupted_wrong_dtypes):
        result = downloader.validate(corrupted_wrong_dtypes)
        dtype_check = next(c for c in result.checks if c.name == "column_dtypes")
        assert dtype_check.passed is False

    def test_null_values_detected(self, downloader, corrupted_null_values):
        result = downloader.validate(corrupted_null_values)
        missing_check = next(c for c in result.checks if c.name == "missing_values")
        assert missing_check.passed is False

    def test_negative_amount_detected(self, downloader, corrupted_negative_amount):
        result = downloader.validate(corrupted_negative_amount)
        amount_check = next(c for c in result.checks if c.name == "amount_range")
        assert amount_check.passed is False

    def test_wrong_column_count(self, downloader, corrupted_missing_columns):
        result = downloader.validate(corrupted_missing_columns)
        col_check = next(c for c in result.checks if c.name == "column_count")
        assert col_check.passed is False


# ------------------------------------------------------------------
# Download (mocked)
# ------------------------------------------------------------------


class TestDownload:
    def test_skip_if_exists(self, downloader, valid_csv):
        """download() returns immediately if the file already exists."""
        dest = valid_csv.parent
        # Rename so it matches expected filename
        target = dest / "creditcard.csv"
        if valid_csv.name != "creditcard.csv":
            valid_csv.rename(target)
        result = downloader.download(dest)
        assert result == target
        assert target.exists()

    def test_kagglehub_download(self, downloader, tmp_path):
        """download() uses kagglehub when available."""
        dest = tmp_path / "download_test"
        dest.mkdir()
        kaggle_dir = tmp_path / "kaggle_cache"
        kaggle_dir.mkdir()
        # Create a fake dataset file where kagglehub would put it
        fake_csv = kaggle_dir / "creditcard.csv"
        fake_csv.write_text("Time,V1,Amount,Class\n0,1,10,0\n")

        mock_kaggle = MagicMock()
        mock_kaggle.dataset_download.return_value = str(kaggle_dir)

        with patch.dict("sys.modules", {"kagglehub": mock_kaggle}):
            result = downloader.download(dest)

        assert result.exists()
        assert result.name == "creditcard.csv"

    def test_raises_when_all_methods_fail(self, downloader, tmp_path):
        """download() raises RuntimeError when all methods fail."""
        dest = tmp_path / "fail_test"
        dest.mkdir()

        with patch.dict("sys.modules", {"kagglehub": None, "requests": None}):
            with patch("src.data.downloader.logger"):
                with pytest.raises(RuntimeError, match="Unable to download"):
                    downloader.download(dest)


# ------------------------------------------------------------------
# Sample creation
# ------------------------------------------------------------------


class TestCreateSample:
    def test_sample_row_count(self, downloader, valid_csv, tmp_path):
        sample_dir = tmp_path / "sample_out"
        result = downloader.create_sample(valid_csv, sample_dir, n_rows=100)
        df = pd.read_csv(result)
        assert len(df) == 100

    def test_sample_preserves_columns(self, downloader, valid_csv, tmp_path):
        sample_dir = tmp_path / "sample_out"
        result = downloader.create_sample(valid_csv, sample_dir, n_rows=100)
        df = pd.read_csv(result)
        assert list(df.columns) == EXPECTED_COLUMNS

    def test_sample_contains_fraud(self, downloader, valid_csv, tmp_path):
        sample_dir = tmp_path / "sample_out"
        result = downloader.create_sample(valid_csv, sample_dir, n_rows=100)
        df = pd.read_csv(result)
        assert df["Class"].sum() >= 1, "Sample should contain at least 1 fraud case"

    def test_sample_filename(self, downloader, valid_csv, tmp_path):
        sample_dir = tmp_path / "sample_out"
        result = downloader.create_sample(valid_csv, sample_dir)
        assert result.name == "sample_transactions.csv"

    def test_sample_creates_directory(self, downloader, valid_csv, tmp_path):
        sample_dir = tmp_path / "nested" / "sample_out"
        result = downloader.create_sample(valid_csv, sample_dir)
        assert sample_dir.exists()
        assert result.exists()


# ------------------------------------------------------------------
# Integration: validate the actual sample file
# ------------------------------------------------------------------


class TestSampleIntegration:
    @pytest.fixture(autouse=True)
    def _require_sample(self):
        if not SAMPLE_PATH.exists():
            pytest.skip("Sample data not available")

    def test_sample_file_exists(self):
        assert SAMPLE_PATH.exists()

    def test_sample_schema(self):
        df = pd.read_csv(SAMPLE_PATH)
        assert len(df.columns) == EXPECTED_COL_COUNT
        missing = set(EXPECTED_COLUMNS) - set(df.columns)
        assert len(missing) == 0, f"Missing columns: {missing}"

    def test_sample_row_count(self):
        df = pd.read_csv(SAMPLE_PATH)
        assert len(df) == 1000

    def test_sample_no_missing_values(self):
        df = pd.read_csv(SAMPLE_PATH)
        assert df.isna().sum().sum() == 0

    def test_sample_amount_non_negative(self):
        df = pd.read_csv(SAMPLE_PATH)
        assert (df["Amount"] >= 0).all()

    def test_sample_class_distribution(self):
        df = pd.read_csv(SAMPLE_PATH)
        fraud_count = int(df["Class"].sum())
        assert fraud_count >= 1, "Sample should contain at least 1 fraud case"
        assert fraud_count <= 50, "Fraud count should be small for realistic imbalance"

    def test_sample_all_numeric(self):
        df = pd.read_csv(SAMPLE_PATH)
        for col in df.columns:
            assert np.issubdtype(df[col].dtype, np.number), f"{col} is not numeric"

    def test_sample_class_values(self):
        df = pd.read_csv(SAMPLE_PATH)
        assert set(df["Class"].unique()).issubset({0, 1})
