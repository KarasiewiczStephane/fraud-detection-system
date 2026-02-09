"""Tests for the async stream consumer."""

from __future__ import annotations

import asyncio

import pytest

from src.streaming.consumer import StreamConsumer


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


async def _identity_inference(txn):
    """Dummy inference that returns the transaction itself."""
    return {"prediction": 0, "transaction_id": txn.get("transaction_id")}


async def _delayed_inference(txn):
    await asyncio.sleep(0.01)
    return {"prediction": 1, "transaction_id": txn.get("transaction_id")}


async def _failing_inference(txn):
    raise RuntimeError("boom")


# ------------------------------------------------------------------
# Queue consumption
# ------------------------------------------------------------------


class TestConsume:
    @pytest.mark.asyncio
    async def test_processes_all_items(self):
        consumer = StreamConsumer()
        queue: asyncio.Queue = asyncio.Queue()
        for i in range(5):
            await queue.put({"transaction_id": f"txn_{i}"})

        async def _run():
            await consumer.consume(queue, _identity_inference)

        async def _stop_when_done():
            await queue.join()  # wait for all items to be processed
            consumer.stop()

        await asyncio.gather(_run(), _stop_when_done())
        assert consumer.processed == 5

    @pytest.mark.asyncio
    async def test_processed_counter(self):
        consumer = StreamConsumer()
        queue: asyncio.Queue = asyncio.Queue()
        for i in range(3):
            await queue.put({"transaction_id": f"txn_{i}"})

        async def _run():
            await consumer.consume(queue, _identity_inference)

        async def _stop_when_done():
            await queue.join()
            consumer.stop()

        await asyncio.gather(_run(), _stop_when_done())
        assert consumer.processed == 3

    @pytest.mark.asyncio
    async def test_stop_halts_loop(self):
        consumer = StreamConsumer()
        queue: asyncio.Queue = asyncio.Queue()

        # Start consumer with no items — it should block on empty queue
        task = asyncio.create_task(consumer.consume(queue, _identity_inference))
        await asyncio.sleep(0.05)
        consumer.stop()
        # Give the consumer time to react to the stop
        await asyncio.sleep(2.5)
        assert task.done()

    @pytest.mark.asyncio
    async def test_handles_inference_error_gracefully(self):
        consumer = StreamConsumer()
        queue: asyncio.Queue = asyncio.Queue()
        await queue.put({"transaction_id": "err_txn"})

        async def _run():
            await consumer.consume(queue, _failing_inference)

        async def _stop_when_done():
            await queue.join()
            consumer.stop()

        # Should not raise — errors are caught and logged
        await asyncio.gather(_run(), _stop_when_done())
        assert consumer.processed == 0  # failed items don't count


# ------------------------------------------------------------------
# Logging callback
# ------------------------------------------------------------------


class TestLogCallback:
    @pytest.mark.asyncio
    async def test_log_fn_called(self):
        logged = []

        async def _log(txn, pred):
            logged.append((txn, pred))

        consumer = StreamConsumer(log_fn=_log)
        queue: asyncio.Queue = asyncio.Queue()
        for i in range(3):
            await queue.put({"transaction_id": f"txn_{i}"})

        async def _run():
            await consumer.consume(queue, _identity_inference)

        async def _stop():
            await queue.join()
            consumer.stop()

        await asyncio.gather(_run(), _stop())
        assert len(logged) == 3

    @pytest.mark.asyncio
    async def test_log_fn_receives_transaction_and_prediction(self):
        logged = []

        async def _log(txn, pred):
            logged.append((txn, pred))

        consumer = StreamConsumer(log_fn=_log)
        queue: asyncio.Queue = asyncio.Queue()
        await queue.put({"transaction_id": "txn_0", "Amount": 99.0})

        async def _run():
            await consumer.consume(queue, _identity_inference)

        async def _stop():
            await queue.join()
            consumer.stop()

        await asyncio.gather(_run(), _stop())
        txn, pred = logged[0]
        assert txn["transaction_id"] == "txn_0"
        assert "prediction" in pred

    @pytest.mark.asyncio
    async def test_no_log_fn_by_default(self):
        consumer = StreamConsumer()
        assert consumer.log_fn is None
        queue: asyncio.Queue = asyncio.Queue()
        await queue.put({"transaction_id": "x"})

        async def _run():
            await consumer.consume(queue, _identity_inference)

        async def _stop():
            await queue.join()
            consumer.stop()

        await asyncio.gather(_run(), _stop())
        assert consumer.processed == 1
