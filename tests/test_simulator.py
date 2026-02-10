"""Tests for the async transaction stream simulator."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.streaming.simulator import TransactionSimulator


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_SAMPLE_CSV = Path("data/sample/sample_transactions.csv")


def _tiny_csv(tmp_path: Path, n: int = 20) -> Path:
    """Create a small synthetic CSV for fast tests."""
    rng = np.random.default_rng(42)
    cols = {f"V{i}": rng.normal(0, 1, n) for i in range(1, 29)}
    cols["Time"] = np.linspace(0, 1000, n)
    cols["Amount"] = rng.uniform(1, 500, n)
    cols["Class"] = np.zeros(n, dtype=int)
    df = pd.DataFrame(cols)
    path = tmp_path / "tiny.csv"
    df.to_csv(path, index=False)
    return path


def _tiny_parquet(tmp_path: Path, n: int = 20) -> Path:
    """Create a small synthetic Parquet file for fast tests."""
    rng = np.random.default_rng(42)
    cols = {f"V{i}": rng.normal(0, 1, n) for i in range(1, 29)}
    cols["Time"] = np.linspace(0, 1000, n)
    cols["Amount"] = rng.uniform(1, 500, n)
    cols["Class"] = np.zeros(n, dtype=int)
    df = pd.DataFrame(cols)
    path = tmp_path / "tiny.parquet"
    df.to_parquet(path)
    return path


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def tiny_csv(tmp_path):
    return _tiny_csv(tmp_path)


@pytest.fixture
def tiny_parquet(tmp_path):
    return _tiny_parquet(tmp_path)


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------


class TestInit:
    def test_loads_csv(self, tiny_csv):
        sim = TransactionSimulator(tiny_csv, rate=100)
        assert sim.num_transactions == 20

    def test_loads_parquet(self, tiny_parquet):
        sim = TransactionSimulator(tiny_parquet, rate=100)
        assert sim.num_transactions == 20

    def test_default_rate(self, tiny_csv):
        sim = TransactionSimulator(tiny_csv)
        assert sim.rate == 10.0

    def test_custom_rate(self, tiny_csv):
        sim = TransactionSimulator(tiny_csv, rate=50.0)
        assert sim.rate == 50.0

    def test_default_fraud_rate(self, tiny_csv):
        sim = TransactionSimulator(tiny_csv)
        assert sim.fraud_rate == 0.02


# ------------------------------------------------------------------
# Streaming
# ------------------------------------------------------------------


class TestStream:
    @pytest.mark.asyncio
    async def test_emits_all_transactions(self, tiny_csv):
        sim = TransactionSimulator(tiny_csv, rate=10000, seed=42)
        queue: asyncio.Queue = asyncio.Queue()
        await sim.stream(queue)
        assert queue.qsize() == 20

    @pytest.mark.asyncio
    async def test_emitted_records_are_dicts(self, tiny_csv):
        sim = TransactionSimulator(tiny_csv, rate=10000, seed=42)
        queue: asyncio.Queue = asyncio.Queue()
        await sim.stream(queue)
        record = await queue.get()
        assert isinstance(record, dict)

    @pytest.mark.asyncio
    async def test_records_have_transaction_id(self, tiny_csv):
        sim = TransactionSimulator(tiny_csv, rate=10000, seed=42)
        queue: asyncio.Queue = asyncio.Queue()
        await sim.stream(queue)
        record = await queue.get()
        assert "transaction_id" in record

    @pytest.mark.asyncio
    async def test_records_have_amount(self, tiny_csv):
        sim = TransactionSimulator(tiny_csv, rate=10000, seed=42)
        queue: asyncio.Queue = asyncio.Queue()
        await sim.stream(queue)
        record = await queue.get()
        assert "Amount" in record

    @pytest.mark.asyncio
    async def test_records_have_v_features(self, tiny_csv):
        sim = TransactionSimulator(tiny_csv, rate=10000, seed=42)
        queue: asyncio.Queue = asyncio.Queue()
        await sim.stream(queue)
        record = await queue.get()
        for i in range(1, 29):
            assert f"V{i}" in record

    @pytest.mark.asyncio
    async def test_rate_control(self, tmp_path):
        """Simulator should take ~0.1s for 10 items at rate=100."""
        path = _tiny_csv(tmp_path, n=10)
        sim = TransactionSimulator(path, rate=100, seed=42)
        queue: asyncio.Queue = asyncio.Queue()

        start = time.monotonic()
        await sim.stream(queue)
        elapsed = time.monotonic() - start

        # 10 items at 100/s → ~0.1s.  Allow tolerance.
        assert elapsed < 1.0  # should be well under 1 second
        assert queue.qsize() == 10

    @pytest.mark.asyncio
    async def test_stop(self, tmp_path):
        """Calling stop() should end the stream early."""
        path = _tiny_csv(tmp_path, n=100)
        sim = TransactionSimulator(path, rate=10000, seed=42)
        queue: asyncio.Queue = asyncio.Queue()

        async def _stop_after(n: int):
            while queue.qsize() < n:
                await asyncio.sleep(0.001)
            sim.stop()

        await asyncio.gather(sim.stream(queue), _stop_after(5))
        # Should have stopped early — far fewer than 100
        assert queue.qsize() < 100


# ------------------------------------------------------------------
# Fraud injection
# ------------------------------------------------------------------


class TestFraudInjection:
    @pytest.mark.asyncio
    async def test_injection_at_rate_1(self, tiny_csv):
        """With fraud_injection_rate=1.0 every record should be fraud."""
        sim = TransactionSimulator(
            tiny_csv,
            rate=10000,
            fraud_injection_rate=1.0,
            seed=42,
        )
        queue: asyncio.Queue = asyncio.Queue()
        await sim.stream(queue)

        while not queue.empty():
            record = await queue.get()
            assert record["Class"] == 1
            assert record["Amount"] >= 500

    @pytest.mark.asyncio
    async def test_injection_at_rate_0(self, tiny_csv):
        """With fraud_injection_rate=0.0 no records should be modified."""
        sim = TransactionSimulator(
            tiny_csv,
            rate=10000,
            fraud_injection_rate=0.0,
            seed=42,
        )
        queue: asyncio.Queue = asyncio.Queue()
        await sim.stream(queue)

        while not queue.empty():
            record = await queue.get()
            assert record["Class"] == 0

    def test_inject_fraud_pattern_sets_class_1(self, tiny_csv):
        sim = TransactionSimulator(tiny_csv, rate=100, seed=42)
        record = {
            "Amount": 10.0,
            "Class": 0,
            "V1": 0.0,
            "V3": 0.0,
            "V4": 0.0,
            "V7": 0.0,
            "V10": 0.0,
            "V14": 0.0,
        }
        modified = sim._inject_fraud_pattern(record)
        assert modified["Class"] == 1

    def test_inject_fraud_pattern_large_amount(self, tiny_csv):
        sim = TransactionSimulator(tiny_csv, rate=100, seed=42)
        record = {
            "Amount": 10.0,
            "Class": 0,
            "V1": 0.0,
            "V3": 0.0,
            "V4": 0.0,
            "V7": 0.0,
            "V10": 0.0,
            "V14": 0.0,
        }
        modified = sim._inject_fraud_pattern(record)
        assert modified["Amount"] >= 500

    def test_inject_fraud_pattern_does_not_mutate_original(self, tiny_csv):
        sim = TransactionSimulator(tiny_csv, rate=100, seed=42)
        record = {
            "Amount": 10.0,
            "Class": 0,
            "V1": 0.0,
            "V3": 0.0,
            "V4": 0.0,
            "V7": 0.0,
            "V10": 0.0,
            "V14": 0.0,
        }
        sim._inject_fraud_pattern(record)
        assert record["Class"] == 0  # original unchanged
