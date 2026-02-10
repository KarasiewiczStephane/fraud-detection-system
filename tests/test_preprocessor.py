"""Tests for the feature engineering pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import FeatureEngineer, _make_card_id

SAMPLE_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "sample" / "sample_transactions.csv"
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_df(
    n: int = 20,
    n_cards: int = 3,
    seed: int = 42,
    time_span: float = 100_000.0,
) -> pd.DataFrame:
    """Build a small synthetic dataset with controllable card identities."""
    rng = np.random.default_rng(seed)
    # Create distinct V1-V5 patterns for each card
    cards = []
    for card_idx in range(n_cards):
        count = n // n_cards + (1 if card_idx < n % n_cards else 0)
        v_base = np.full(5, card_idx * 10.0)  # well-separated cards
        rows = {f"V{i}": np.full(count, v_base[i - 1]) for i in range(1, 6)}
        for i in range(6, 29):
            rows[f"V{i}"] = rng.normal(0, 1, count)
        rows["Time"] = np.sort(rng.uniform(0, time_span, count))
        rows["Amount"] = rng.uniform(1, 500, count)
        rows["Class"] = np.zeros(count, dtype=int)
        cards.append(pd.DataFrame(rows))
    df = pd.concat(cards, ignore_index=True).sort_values("Time").reset_index(drop=True)
    return df


def _single_card_df(n: int = 5, time_step: float = 1000.0) -> pd.DataFrame:
    """Dataset where every row belongs to the same card."""
    rng = np.random.default_rng(99)
    data = {f"V{i}": np.full(n, 1.0) for i in range(1, 29)}
    data["Time"] = np.arange(n) * time_step
    data["Amount"] = rng.uniform(10, 100, n)
    data["Class"] = np.zeros(n, dtype=int)
    return pd.DataFrame(data)


def _constant_amount_df(n: int = 10, amount: float = 50.0) -> pd.DataFrame:
    """Dataset where all transactions have the same Amount."""
    data = {f"V{i}": np.full(n, 2.0) for i in range(1, 29)}
    data["Time"] = np.arange(n) * 500.0
    data["Amount"] = np.full(n, amount)
    data["Class"] = np.zeros(n, dtype=int)
    return pd.DataFrame(data)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def fe():
    return FeatureEngineer()


@pytest.fixture
def small_df():
    return _make_df(n=20, n_cards=3)


@pytest.fixture
def single_card():
    return _single_card_df(n=5)


@pytest.fixture
def constant_amount():
    return _constant_amount_df()


# ------------------------------------------------------------------
# card_id helper
# ------------------------------------------------------------------


class TestCardId:
    def test_deterministic(self, small_df):
        ids1 = _make_card_id(small_df)
        ids2 = _make_card_id(small_df)
        assert (ids1 == ids2).all()

    def test_different_cards_get_different_ids(self):
        df = _make_df(n=6, n_cards=3)
        ids = _make_card_id(df)
        assert ids.nunique() == 3

    def test_same_card_rows_share_id(self):
        df = _single_card_df(n=5)
        ids = _make_card_id(df)
        assert ids.nunique() == 1


# ------------------------------------------------------------------
# Velocity features
# ------------------------------------------------------------------


class TestVelocityFeatures:
    def test_columns_present(self, fe, small_df):
        result = fe.compute_velocity_features(small_df)
        for col in ("vel_1h", "vel_6h", "vel_24h"):
            assert col in result.columns

    def test_first_transaction_has_zero_velocity(self, fe, single_card):
        result = fe.compute_velocity_features(single_card)
        # First row (smallest Time) for the only card has no prior transactions
        first = result.iloc[0]
        assert first["vel_1h"] == 0
        assert first["vel_6h"] == 0
        assert first["vel_24h"] == 0

    def test_velocity_counts_only_past(self, fe):
        """Transactions at or after current time must not be counted."""
        df = _single_card_df(n=3, time_step=100.0)
        # Times: 0, 100, 200 — all within 1h (3600s) of each other
        result = fe.compute_velocity_features(df)
        # Row 0: no past → 0
        assert result.iloc[0]["vel_1h"] == 0
        # Row 1: 1 past transaction within 1h
        assert result.iloc[1]["vel_1h"] == 1
        # Row 2: 2 past transactions within 1h
        assert result.iloc[2]["vel_1h"] == 2

    def test_velocity_window_boundary(self, fe):
        """Transaction exactly at window boundary should NOT be counted."""
        data = {f"V{i}": np.full(3, 1.0) for i in range(1, 29)}
        data["Time"] = [0.0, 3600.0, 7200.0]  # exactly 1h apart
        data["Amount"] = [10, 20, 30]
        data["Class"] = [0, 0, 0]
        df = pd.DataFrame(data)
        result = fe.compute_velocity_features(df)
        # Row at t=3600: past window [0, 3600) includes t=0
        assert result.iloc[1]["vel_1h"] == 1
        # Row at t=7200: past window [3600, 7200) includes t=3600
        assert result.iloc[2]["vel_1h"] == 1

    def test_preserves_original_columns(self, fe, small_df):
        result = fe.compute_velocity_features(small_df)
        for col in small_df.columns:
            assert col in result.columns

    def test_no_nan_values(self, fe, small_df):
        result = fe.compute_velocity_features(small_df)
        vel_cols = [c for c in result.columns if c.startswith("vel_")]
        assert result[vel_cols].isna().sum().sum() == 0

    def test_row_count_preserved(self, fe, small_df):
        result = fe.compute_velocity_features(small_df)
        assert len(result) == len(small_df)


# ------------------------------------------------------------------
# Amount statistics
# ------------------------------------------------------------------


class TestAmountStatistics:
    def test_columns_present(self, fe, small_df):
        result = fe.compute_amount_statistics(small_df)
        for win in (5, 10, 20):
            assert f"amt_mean_{win}" in result.columns
            assert f"amt_std_{win}" in result.columns
            assert f"amt_max_{win}" in result.columns

    def test_first_transaction_filled_zero(self, fe, single_card):
        result = fe.compute_amount_statistics(single_card)
        first = result.iloc[0]
        for win in (5, 10, 20):
            assert first[f"amt_mean_{win}"] == 0
            assert first[f"amt_std_{win}"] == 0
            assert first[f"amt_max_{win}"] == 0

    def test_rolling_mean_uses_past_only(self, fe):
        """The rolling mean for row i must not include row i's Amount."""
        df = _single_card_df(n=6, time_step=500.0)
        amounts = df["Amount"].values.copy()
        result = fe.compute_amount_statistics(df)

        # For win=5, row 3: past window is rows 0,1,2
        expected_mean = np.mean(amounts[:3])
        actual_mean = result.iloc[3]["amt_mean_5"]
        assert abs(actual_mean - expected_mean) < 1e-6

    def test_constant_amount_zero_std(self, fe, constant_amount):
        result = fe.compute_amount_statistics(constant_amount)
        # After the first row (which is 0-filled), all std should be 0
        for idx in range(2, len(result)):
            for win in (5, 10, 20):
                assert result.iloc[idx][f"amt_std_{win}"] == pytest.approx(0, abs=1e-6)

    def test_no_nan_values(self, fe, small_df):
        result = fe.compute_amount_statistics(small_df)
        amt_cols = [c for c in result.columns if c.startswith("amt_")]
        assert result[amt_cols].isna().sum().sum() == 0

    def test_row_count_preserved(self, fe, small_df):
        result = fe.compute_amount_statistics(small_df)
        assert len(result) == len(small_df)


# ------------------------------------------------------------------
# Time features
# ------------------------------------------------------------------


class TestTimeFeatures:
    def test_columns_present(self, fe, small_df):
        result = fe.compute_time_features(small_df)
        assert "time_hour_of_day" in result.columns
        assert "time_day_of_week" in result.columns
        assert "time_since_last" in result.columns

    def test_hour_of_day_range(self, fe, small_df):
        result = fe.compute_time_features(small_df)
        assert (result["time_hour_of_day"] >= 0).all()
        assert (result["time_hour_of_day"] < 24).all()

    def test_day_of_week_range(self, fe, small_df):
        result = fe.compute_time_features(small_df)
        assert (result["time_day_of_week"] >= 0).all()
        assert (result["time_day_of_week"] <= 6).all()

    def test_hour_of_day_calculation(self, fe):
        data = {f"V{i}": [1.0] for i in range(1, 29)}
        data["Time"] = [43200.0]  # 12 hours into the day
        data["Amount"] = [10.0]
        data["Class"] = [0]
        df = pd.DataFrame(data)
        result = fe.compute_time_features(df)
        assert result.iloc[0]["time_hour_of_day"] == pytest.approx(12.0)

    def test_time_since_last_first_transaction(self, fe, single_card):
        result = fe.compute_time_features(single_card)
        # First transaction for the card should have time_since_last == 0
        assert result.iloc[0]["time_since_last"] == 0.0

    def test_time_since_last_computation(self, fe):
        df = _single_card_df(n=3, time_step=1000.0)
        result = fe.compute_time_features(df)
        assert result.iloc[0]["time_since_last"] == 0.0
        assert result.iloc[1]["time_since_last"] == pytest.approx(1000.0)
        assert result.iloc[2]["time_since_last"] == pytest.approx(1000.0)

    def test_time_since_last_per_card(self, fe):
        """time_since_last should be computed independently per card."""
        df = _make_df(n=6, n_cards=2)
        result = fe.compute_time_features(df)
        card_ids = _make_card_id(result)
        for card_id in card_ids.unique():
            card_mask = card_ids == card_id
            card_rows = result[card_mask].sort_values("Time")
            # First row per card must be 0
            assert card_rows.iloc[0]["time_since_last"] == 0.0

    def test_no_nan_values(self, fe, small_df):
        result = fe.compute_time_features(small_df)
        time_cols = [c for c in result.columns if c.startswith("time_")]
        assert result[time_cols].isna().sum().sum() == 0

    def test_row_count_preserved(self, fe, small_df):
        result = fe.compute_time_features(small_df)
        assert len(result) == len(small_df)


# ------------------------------------------------------------------
# Amount deviation
# ------------------------------------------------------------------


class TestAmountDeviation:
    def test_columns_present(self, fe, small_df):
        result = fe.compute_amount_deviation(small_df)
        assert "dev_zscore" in result.columns
        assert "dev_ratio_to_max" in result.columns

    def test_first_transaction_zero(self, fe, single_card):
        result = fe.compute_amount_deviation(single_card)
        assert result.iloc[0]["dev_zscore"] == 0.0
        assert result.iloc[0]["dev_ratio_to_max"] == 0.0

    def test_zscore_uses_past_only(self, fe):
        """z-score at row i must be computed from rows 0..i-1 only."""
        df = _single_card_df(n=4, time_step=500.0)
        amounts = df["Amount"].values.copy()
        result = fe.compute_amount_deviation(df)

        # Row 2: past is rows 0,1
        past = amounts[:2]
        mean, std = past.mean(), past.std(ddof=0)
        expected_z = (amounts[2] - mean) / std if std > 0 else 0.0
        assert result.iloc[2]["dev_zscore"] == pytest.approx(expected_z, abs=1e-6)

    def test_constant_amount_zero_zscore(self, fe, constant_amount):
        result = fe.compute_amount_deviation(constant_amount)
        # All amounts are identical → std is 0 → z-score should be 0
        assert (result["dev_zscore"] == 0.0).all()

    def test_ratio_to_max(self, fe):
        df = _single_card_df(n=3, time_step=500.0)
        amounts = df["Amount"].values.copy()
        result = fe.compute_amount_deviation(df)
        # Row 2: ratio = amounts[2] / max(amounts[0], amounts[1])
        expected = amounts[2] / max(amounts[:2])
        assert result.iloc[2]["dev_ratio_to_max"] == pytest.approx(expected, abs=1e-6)

    def test_no_nan_values(self, fe, small_df):
        result = fe.compute_amount_deviation(small_df)
        dev_cols = [c for c in result.columns if c.startswith("dev_")]
        assert result[dev_cols].isna().sum().sum() == 0

    def test_row_count_preserved(self, fe, small_df):
        result = fe.compute_amount_deviation(small_df)
        assert len(result) == len(small_df)


# ------------------------------------------------------------------
# fit_transform (full pipeline)
# ------------------------------------------------------------------


class TestFitTransform:
    def test_all_feature_columns_present(self, fe, small_df):
        result = fe.fit_transform(small_df)
        expected_prefixes = ("vel_", "amt_", "time_", "dev_")
        for prefix in expected_prefixes:
            matching = [c for c in result.columns if c.startswith(prefix)]
            assert len(matching) > 0, f"No columns with prefix '{prefix}'"

    def test_original_columns_preserved(self, fe, small_df):
        result = fe.fit_transform(small_df)
        for col in small_df.columns:
            assert col in result.columns

    def test_no_nan_values(self, fe, small_df):
        result = fe.fit_transform(small_df)
        assert result.isna().sum().sum() == 0

    def test_row_count_preserved(self, fe, small_df):
        result = fe.fit_transform(small_df)
        assert len(result) == len(small_df)

    def test_sorted_by_time(self, fe, small_df):
        result = fe.fit_transform(small_df)
        assert (result["Time"].diff().dropna() >= 0).all()

    def test_expected_column_count(self, fe, small_df):
        result = fe.fit_transform(small_df)
        original_count = len(small_df.columns)
        # vel: 3, amt: 9 (3 stats * 3 windows), time: 3, dev: 2 = 17 new
        new_cols = len(result.columns) - original_count
        assert new_cols == 17

    def test_feature_naming_convention(self, fe, small_df):
        result = fe.fit_transform(small_df)
        new_cols = set(result.columns) - set(small_df.columns)
        valid_prefixes = ("vel_", "amt_", "time_", "dev_")
        for col in new_cols:
            assert any(col.startswith(p) for p in valid_prefixes), (
                f"Column '{col}' doesn't follow naming convention"
            )

    def test_single_row_no_error(self, fe):
        """Pipeline should handle a single-row dataframe gracefully."""
        data = {f"V{i}": [1.0] for i in range(1, 29)}
        data["Time"] = [0.0]
        data["Amount"] = [50.0]
        data["Class"] = [0]
        df = pd.DataFrame(data)
        result = fe.fit_transform(df)
        assert len(result) == 1
        assert result.isna().sum().sum() == 0


# ------------------------------------------------------------------
# Data leakage checks
# ------------------------------------------------------------------


class TestNoDataLeakage:
    """Verify that features at row i only depend on rows 0..i-1."""

    def test_velocity_no_future_counts(self, fe):
        df = _single_card_df(n=5, time_step=100.0)
        result = fe.compute_velocity_features(df)
        # Row 0 should never count any other transactions
        assert result.iloc[0]["vel_1h"] == 0
        assert result.iloc[0]["vel_6h"] == 0
        assert result.iloc[0]["vel_24h"] == 0

    def test_amount_stats_no_future(self, fe):
        df = _single_card_df(n=6, time_step=500.0)
        amounts = df["Amount"].values.copy()
        result = fe.compute_amount_statistics(df)
        # Row 1: rolling mean_5 should only use row 0
        assert result.iloc[1]["amt_mean_5"] == pytest.approx(amounts[0], abs=1e-6)

    def test_deviation_no_future(self, fe):
        df = _single_card_df(n=4, time_step=500.0)
        amounts = df["Amount"].values.copy()
        result = fe.compute_amount_deviation(df)
        # Row 1: ratio_to_max should be amounts[1] / amounts[0]
        expected = amounts[1] / amounts[0] if amounts[0] > 0 else 0
        assert result.iloc[1]["dev_ratio_to_max"] == pytest.approx(expected, abs=1e-6)


# ------------------------------------------------------------------
# Integration with sample data
# ------------------------------------------------------------------


class TestSampleIntegration:
    @pytest.fixture(autouse=True)
    def _require_sample(self):
        if not SAMPLE_PATH.exists():
            pytest.skip("Sample data not available")

    def test_fit_transform_on_sample(self, fe):
        """Run the full pipeline on the sample CSV (smoke test)."""
        df = pd.read_csv(SAMPLE_PATH)
        # Use a small subset for speed
        subset = df.head(50).copy()
        result = fe.fit_transform(subset)
        assert len(result) == 50
        assert result.isna().sum().sum() == 0
        # All new columns follow naming convention
        new_cols = set(result.columns) - set(subset.columns)
        for col in new_cols:
            assert col.startswith(("vel_", "amt_", "time_", "dev_"))
