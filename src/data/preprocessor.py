"""Feature engineering pipeline for credit card fraud detection."""

from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Columns used to derive a simulated card identifier
_CARD_ID_COLS = ["V1", "V2", "V3", "V4", "V5"]


def _make_card_id(df: pd.DataFrame) -> pd.Series:
    """Derive a deterministic card identifier by hashing V1–V5."""
    raw = df[_CARD_ID_COLS].round(2).astype(str).agg("|".join, axis=1)
    return raw.apply(lambda x: hashlib.md5(x.encode()).hexdigest()[:8])


class FeatureEngineer:
    """Compute derived features from raw transaction data.

    All rolling / expanding computations respect temporal ordering and only
    use **past** data to prevent leakage.  The dataframe is always sorted by
    ``Time`` before computation.

    Feature naming convention
    -------------------------
    - ``vel_``  – transaction velocity / frequency features
    - ``amt_``  – amount-based rolling statistics
    - ``time_`` – time-derived features
    - ``dev_``  – personal deviation features
    """

    # ------------------------------------------------------------------
    # Velocity features
    # ------------------------------------------------------------------

    def compute_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Count transactions per card in the last 1 h, 6 h and 24 h.

        Uses the ``Time`` column (seconds since first transaction) and a
        simulated ``card_id`` derived from V1–V5.
        """
        df = df.copy()
        df["_card_id"] = _make_card_id(df)
        df = df.sort_values("Time").reset_index(drop=True)

        windows = {"vel_1h": 3600, "vel_6h": 21600, "vel_24h": 86400}

        for col_name, window_sec in windows.items():
            counts: List[int] = []
            for idx, row in df.iterrows():
                card = row["_card_id"]
                t = row["Time"]
                mask = (
                    (df["_card_id"] == card)
                    & (df["Time"] >= t - window_sec)
                    & (df["Time"] < t)  # strictly past
                )
                counts.append(int(mask.sum()))
            df[col_name] = counts

        df.drop(columns=["_card_id"], inplace=True)
        return df

    # ------------------------------------------------------------------
    # Amount statistics
    # ------------------------------------------------------------------

    def compute_amount_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling mean, std and max of ``Amount`` per card.

        Window sizes: 5, 10, 20 previous transactions.  Only past
        transactions are included (the current row is excluded).
        """
        df = df.copy()
        df["_card_id"] = _make_card_id(df)
        df = df.sort_values("Time").reset_index(drop=True)

        for win in (5, 10, 20):
            mean_col = f"amt_mean_{win}"
            std_col = f"amt_std_{win}"
            max_col = f"amt_max_{win}"
            df[mean_col] = np.nan
            df[std_col] = np.nan
            df[max_col] = np.nan

            for card_id, grp in df.groupby("_card_id"):
                idxs = grp.index.tolist()
                for pos, idx in enumerate(idxs):
                    past = grp.loc[idxs[max(0, pos - win) : pos], "Amount"]
                    if len(past) == 0:
                        continue
                    df.loc[idx, mean_col] = past.mean()
                    df.loc[idx, std_col] = past.std(ddof=0)
                    df.loc[idx, max_col] = past.max()

        # Fill remaining NaNs (first transaction for a card) with 0
        amt_cols = [c for c in df.columns if c.startswith("amt_")]
        df[amt_cols] = df[amt_cols].fillna(0)

        df.drop(columns=["_card_id"], inplace=True)
        return df

    # ------------------------------------------------------------------
    # Time features
    # ------------------------------------------------------------------

    def compute_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive time-based features from the ``Time`` column.

        - ``time_hour_of_day``: hour within the day (0–23.99)
        - ``time_day_of_week``: approximate day of week (0–6)
        - ``time_since_last``: seconds since previous transaction for same card
        """
        df = df.copy()
        df["_card_id"] = _make_card_id(df)
        df = df.sort_values("Time").reset_index(drop=True)

        df["time_hour_of_day"] = (df["Time"] % 86400) / 3600
        df["time_day_of_week"] = ((df["Time"] / 86400) % 7).astype(int)

        df["time_since_last"] = np.nan
        for _, grp in df.groupby("_card_id"):
            idxs = grp.index
            df.loc[idxs, "time_since_last"] = grp["Time"].diff()

        # First transaction for each card has no prior – fill with 0
        df["time_since_last"] = df["time_since_last"].fillna(0)

        df.drop(columns=["_card_id"], inplace=True)
        return df

    # ------------------------------------------------------------------
    # Amount deviation
    # ------------------------------------------------------------------

    def compute_amount_deviation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deviation of current ``Amount`` from personal history.

        - ``dev_zscore``: z-score vs expanding mean/std of past amounts
        - ``dev_ratio_to_max``: ratio of current amount to personal max
        """
        df = df.copy()
        df["_card_id"] = _make_card_id(df)
        df = df.sort_values("Time").reset_index(drop=True)

        df["dev_zscore"] = 0.0
        df["dev_ratio_to_max"] = 0.0

        for card_id, grp in df.groupby("_card_id"):
            idxs = grp.index.tolist()
            for pos, idx in enumerate(idxs):
                past_amounts = grp.loc[idxs[:pos], "Amount"]
                current = df.loc[idx, "Amount"]
                if len(past_amounts) == 0:
                    # No history – deviation is 0
                    df.loc[idx, "dev_zscore"] = 0.0
                    df.loc[idx, "dev_ratio_to_max"] = 0.0
                    continue

                mean = past_amounts.mean()
                std = past_amounts.std(ddof=0)
                if std > 0:
                    df.loc[idx, "dev_zscore"] = (current - mean) / std
                else:
                    df.loc[idx, "dev_zscore"] = 0.0

                past_max = past_amounts.max()
                if past_max > 0:
                    df.loc[idx, "dev_ratio_to_max"] = current / past_max
                else:
                    df.loc[idx, "dev_ratio_to_max"] = 0.0

        df.drop(columns=["_card_id"], inplace=True)
        return df

    # ------------------------------------------------------------------
    # fit_transform
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full feature engineering pipeline.

        Chains all feature computations and returns an enriched dataframe
        with the original columns plus all derived features.
        """
        logger.info("Starting feature engineering on %d rows", len(df))

        df = df.sort_values("Time").reset_index(drop=True)

        df = self.compute_velocity_features(df)
        df = self.compute_amount_statistics(df)
        df = self.compute_time_features(df)
        df = self.compute_amount_deviation(df)

        # Final safety net: fill any remaining NaN with 0
        df = df.fillna(0)

        logger.info(
            "Feature engineering complete: %d rows, %d columns",
            len(df),
            len(df.columns),
        )
        return df


# ------------------------------------------------------------------
# Temporal train / test split
# ------------------------------------------------------------------


def temporal_split(
    df: pd.DataFrame, test_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe temporally so the training set precedes the test set.

    The dataframe is sorted by ``Time`` and split at the
    ``(1 - test_ratio)`` percentile of the ``Time`` column.  This
    guarantees no future data leaks into the training set.

    Parameters
    ----------
    df:
        DataFrame that must contain a ``Time`` column.
    test_ratio:
        Fraction of data (by time) to allocate to the test set.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(train_df, test_df)`` — disjoint, temporally ordered.
    """
    df = df.sort_values("Time").reset_index(drop=True)
    split_time = np.percentile(df["Time"], (1 - test_ratio) * 100)
    train_df = df[df["Time"] <= split_time].reset_index(drop=True)
    test_df = df[df["Time"] > split_time].reset_index(drop=True)
    logger.info(
        "Temporal split: %d train, %d test (split at Time=%.1f)",
        len(train_df),
        len(test_df),
        split_time,
    )
    return train_df, test_df


# ------------------------------------------------------------------
# Class balancing
# ------------------------------------------------------------------


def balance_classes(
    X: np.ndarray,
    y: np.ndarray,
    strategy: str = "smote",
) -> Union[Tuple[np.ndarray, np.ndarray], Dict[int, float]]:
    """Handle class imbalance via SMOTE resampling or class-weight computation.

    Parameters
    ----------
    X:
        Feature matrix of shape ``(n_samples, n_features)``.
    y:
        Binary label array of shape ``(n_samples,)``.
    strategy:
        ``"smote"`` — returns ``(X_resampled, y_resampled)``
        ``"class_weight"`` — returns a ``{class: weight}`` dict.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] | dict[int, float]
    """
    if strategy == "smote":
        from imblearn.over_sampling import SMOTE

        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        logger.info(
            "SMOTE resampling: %d → %d samples (class 0: %d, class 1: %d)",
            len(y),
            len(y_res),
            int((y_res == 0).sum()),
            int((y_res == 1).sum()),
        )
        return X_res, y_res

    if strategy == "class_weight":
        weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
        weight_dict = {0: float(weights[0]), 1: float(weights[1])}
        logger.info("Computed class weights: %s", weight_dict)
        return weight_dict

    raise ValueError(f"Unknown balancing strategy: {strategy!r}")
