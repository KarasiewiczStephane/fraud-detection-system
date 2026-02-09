"""Entry-point script for the transaction stream simulator container."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from src.streaming.consumer import StreamConsumer
from src.streaming.simulator import TransactionSimulator
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def main() -> None:
    api_url = os.environ.get("API_URL", "http://localhost:8000")
    rate = float(os.environ.get("STREAM_RATE", "10"))
    data_path = os.environ.get(
        "DATA_PATH", "data/sample/sample_transactions.csv"
    )

    logger.info(
        "Starting simulator: rate=%.1f, data=%s, api=%s",
        rate, data_path, api_url,
    )

    sim = TransactionSimulator(Path(data_path), rate=rate)
    consumer = StreamConsumer()
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)

    async def _inference(txn):
        """Send transaction to the API for scoring."""
        import httpx

        payload = {
            "transaction_id": txn.get("transaction_id", "unknown"),
            **{k: float(v) for k, v in txn.items()
               if k.startswith("V") or k in ("Time", "Amount")},
        }
        async with httpx.AsyncClient(base_url=api_url) as client:
            resp = await client.post("/api/v1/predict", json=payload)
            return resp.json()

    await asyncio.gather(
        sim.stream(queue),
        consumer.consume(queue, _inference),
    )


if __name__ == "__main__":
    asyncio.run(main())
