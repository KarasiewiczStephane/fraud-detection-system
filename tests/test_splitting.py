"""Tests for temporal splitting and class balancing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import balance_classes, temporal_split


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_time_df(n: int = 100, fraud_rate: float = 0.05, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic time-ordered dataframe with a Class column."""
    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(n * fraud_rate))
    n_normal = n - n_fraud
    data = {
        "Time": np.sort(rng.uniform(0, 172_800, n)),
        "Amount": rng.uniform(1, 500, n),
        "Class": np.concatenate([np.zeros(n_normal), np.ones(n_fraud)]).astype(int),
    }
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(0, 1, n)
    df = pd.DataFrame(data)
    # Shuffle then sort by Time so Class labels aren't always at the end
    df = df.sample(frac=1, random_state=seed).sort_values("Time").reset_index(drop=True)
    return df


# ------------------------------------------------------------------
# temporal_split
# ------------------------------------------------------------------


class TestTemporalSplit:
    def test_returns_two_dataframes(self):
        df = _make_time_df()
        train, test = temporal_split(df)
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)

    def test_no_overlap(self):
        df = _make_time_df()
        train, test = temporal_split(df)
        assert train["Time"].max() < test["Time"].min()

    def test_covers_all_rows(self):
        df = _make_time_df(n=200)
        train, test = temporal_split(df)
        assert len(train) + len(test) == len(df)

    def test_train_before_test(self):
        df = _make_time_df()
        train, test = temporal_split(df)
        assert train["Time"].max() <= test["Time"].min()

    def test_train_sorted_by_time(self):
        df = _make_time_df()
        train, _ = temporal_split(df)
        assert (train["Time"].diff().dropna() >= 0).all()

    def test_test_sorted_by_time(self):
        df = _make_time_df()
        _, test = temporal_split(df)
        assert (test["Time"].diff().dropna() >= 0).all()

    def test_default_ratio_approximately_80_20(self):
        df = _make_time_df(n=1000)
        train, test = temporal_split(df, test_ratio=0.2)
        train_ratio = len(train) / len(df)
        # Allow tolerance because split is by time percentile, not exact count
        assert 0.70 <= train_ratio <= 0.90

    def test_custom_ratio(self):
        df = _make_time_df(n=1000)
        train, test = temporal_split(df, test_ratio=0.5)
        # Roughly 50/50
        train_ratio = len(train) / len(df)
        assert 0.40 <= train_ratio <= 0.60

    def test_small_test_ratio(self):
        df = _make_time_df(n=1000)
        train, test = temporal_split(df, test_ratio=0.05)
        assert len(train) > len(test)

    def test_preserves_columns(self):
        df = _make_time_df()
        train, test = temporal_split(df)
        assert list(train.columns) == list(df.columns)
        assert list(test.columns) == list(df.columns)

    def test_no_data_leakage(self):
        """Every Time value in test must be strictly greater than all in train."""
        df = _make_time_df(n=500)
        train, test = temporal_split(df)
        if len(test) > 0 and len(train) > 0:
            assert train["Time"].max() < test["Time"].min()

    def test_single_row_df(self):
        df = _make_time_df(n=1)
        train, test = temporal_split(df, test_ratio=0.2)
        assert len(train) + len(test) == 1


# ------------------------------------------------------------------
# balance_classes — SMOTE
# ------------------------------------------------------------------


class TestBalanceClassesSMOTE:
    @pytest.fixture
    def imbalanced_data(self):
        rng = np.random.default_rng(42)
        n_normal, n_fraud = 200, 10
        X = rng.normal(0, 1, (n_normal + n_fraud, 5))
        y = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)]).astype(int)
        return X, y

    def test_returns_tuple(self, imbalanced_data):
        X, y = imbalanced_data
        result = balance_classes(X, y, strategy="smote")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_balanced_output(self, imbalanced_data):
        X, y = imbalanced_data
        X_res, y_res = balance_classes(X, y, strategy="smote")
        counts = np.bincount(y_res.astype(int))
        assert counts[0] == counts[1]

    def test_more_samples_than_original(self, imbalanced_data):
        X, y = imbalanced_data
        X_res, y_res = balance_classes(X, y, strategy="smote")
        assert len(y_res) > len(y)

    def test_feature_count_preserved(self, imbalanced_data):
        X, y = imbalanced_data
        X_res, _ = balance_classes(X, y, strategy="smote")
        assert X_res.shape[1] == X.shape[1]

    def test_original_samples_present(self, imbalanced_data):
        """All original majority-class samples should still be present."""
        X, y = imbalanced_data
        X_res, y_res = balance_classes(X, y, strategy="smote")
        # The number of class-0 samples should be unchanged
        assert (y_res == 0).sum() == (y == 0).sum()

    def test_deterministic_with_seed(self, imbalanced_data):
        X, y = imbalanced_data
        X1, y1 = balance_classes(X, y, strategy="smote")
        X2, y2 = balance_classes(X, y, strategy="smote")
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


# ------------------------------------------------------------------
# balance_classes — class_weight
# ------------------------------------------------------------------


class TestBalanceClassesWeight:
    @pytest.fixture
    def imbalanced_y(self):
        return np.concatenate([np.zeros(180), np.ones(20)]).astype(int)

    def test_returns_dict(self, imbalanced_y):
        X_dummy = np.zeros((len(imbalanced_y), 3))
        result = balance_classes(X_dummy, imbalanced_y, strategy="class_weight")
        assert isinstance(result, dict)

    def test_keys(self, imbalanced_y):
        X_dummy = np.zeros((len(imbalanced_y), 3))
        result = balance_classes(X_dummy, imbalanced_y, strategy="class_weight")
        assert 0 in result
        assert 1 in result

    def test_minority_has_higher_weight(self, imbalanced_y):
        X_dummy = np.zeros((len(imbalanced_y), 3))
        weights = balance_classes(X_dummy, imbalanced_y, strategy="class_weight")
        assert weights[1] > weights[0]

    def test_balanced_data_equal_weights(self):
        y = np.concatenate([np.zeros(100), np.ones(100)]).astype(int)
        X_dummy = np.zeros((200, 3))
        weights = balance_classes(X_dummy, y, strategy="class_weight")
        assert weights[0] == pytest.approx(1.0)
        assert weights[1] == pytest.approx(1.0)

    def test_weight_formula(self, imbalanced_y):
        """Verify weights match sklearn's 'balanced' formula: n / (n_classes * count_k)."""
        X_dummy = np.zeros((len(imbalanced_y), 3))
        weights = balance_classes(X_dummy, imbalanced_y, strategy="class_weight")
        n = len(imbalanced_y)
        n_classes = 2
        expected_0 = n / (n_classes * (imbalanced_y == 0).sum())
        expected_1 = n / (n_classes * (imbalanced_y == 1).sum())
        assert weights[0] == pytest.approx(expected_0)
        assert weights[1] == pytest.approx(expected_1)


# ------------------------------------------------------------------
# balance_classes — error handling
# ------------------------------------------------------------------


class TestBalanceClassesErrors:
    def test_unknown_strategy_raises(self):
        X = np.zeros((10, 3))
        y = np.zeros(10)
        with pytest.raises(ValueError, match="Unknown balancing strategy"):
            balance_classes(X, y, strategy="unknown")
