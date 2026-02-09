"""Async transaction stream simulator for fraud detection."""

from __future__ import annotations

import asyncio
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TransactionSimulator:
    """Emit transactions from a dataset at a configurable rate.

    Parameters
    ----------
    data_path:
        Path to a CSV or Parquet file with transaction data.  The file must
        contain columns ``Time``, ``V1``–``V28``, ``Amount``, and ``Class``.
    rate:
        Target transactions per second.
    fraud_injection_rate:
        Probability of replacing a normal transaction with a synthetic
        fraud pattern on each step.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        data_path: Path | str,
        rate: float = 10.0,
        fraud_injection_rate: float = 0.02,
        seed: int | None = None,
    ) -> None:
        path = Path(data_path)
        if path.suffix == ".parquet":
            self.data: pd.DataFrame = pd.read_parquet(path)
        else:
            self.data = pd.read_csv(path)
        self.rate = rate
        self.fraud_rate = fraud_injection_rate
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self._stopped = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def stream(self, queue: asyncio.Queue) -> None:  # type: ignore[type-arg]
        """Push transactions onto *queue* at ``self.rate`` per second.

        Iterates through every row in the dataset.  Each row is emitted as
        a ``dict``.  If ``fraud_injection_rate`` triggers, the row is
        replaced with a synthetic fraud pattern before being enqueued.
        """
        self._stopped = False
        delay = 1.0 / self.rate if self.rate > 0 else 0.0

        for idx, row in self.data.iterrows():
            if self._stopped:
                break

            record = row.to_dict()
            record["transaction_id"] = f"txn_{idx}"

            if self._rng.random() < self.fraud_rate:
                record = self._inject_fraud_pattern(record)

            await queue.put(record)
            if delay > 0:
                await asyncio.sleep(delay)

        logger.info(
            "Simulator finished — emitted %d transactions", len(self.data)
        )

    def stop(self) -> None:
        """Signal the stream loop to stop after the current iteration."""
        self._stopped = True

    # ------------------------------------------------------------------
    # Fraud injection
    # ------------------------------------------------------------------

    def _inject_fraud_pattern(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Modify *record* in-place to look like a fraudulent transaction.

        The synthetic fraud pattern:
        - Large amount (95th-percentile-like spike)
        - Shift several V-features toward abnormal values
        - Set ``Class`` to 1 (fraud)
        """
        record = dict(record)  # shallow copy
        record["Amount"] = float(self._np_rng.uniform(500, 5000))
        record["Class"] = 1

        # Shift a handful of V-features that are typically discriminative
        for col in ("V1", "V3", "V4", "V7", "V10", "V14"):
            if col in record:
                record[col] = float(
                    record[col] + self._np_rng.normal(-3, 1)
                )
        return record

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def num_transactions(self) -> int:
        """Number of transactions in the loaded dataset."""
        return len(self.data)
