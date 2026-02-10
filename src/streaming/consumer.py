"""Async stream consumer that processes transactions from a queue."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine, Dict, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class StreamConsumer:
    """Consume transactions from an :class:`asyncio.Queue` and run inference.

    Parameters
    ----------
    log_fn:
        Optional async callback invoked after each prediction with the
        ``(transaction, prediction)`` pair.  Useful for logging to a
        database or metrics system.
    """

    def __init__(
        self,
        log_fn: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Coroutine]] = None,
    ) -> None:
        self.log_fn = log_fn
        self.processed: int = 0
        self._stopped = False

    async def consume(
        self,
        queue: asyncio.Queue,  # type: ignore[type-arg]
        inference_fn: Callable[[Dict[str, Any]], Coroutine],
    ) -> None:
        """Pull transactions from *queue* and pass them to *inference_fn*.

        Parameters
        ----------
        queue:
            The asyncio queue from which transactions are read.
        inference_fn:
            An async callable that accepts a transaction dict and returns
            a prediction dict.
        """
        self._stopped = False

        while not self._stopped:
            try:
                transaction: Dict[str, Any] = await asyncio.wait_for(
                    queue.get(),
                    timeout=2.0,
                )
            except asyncio.TimeoutError:
                # No items for 2 s — check stop flag and retry
                continue

            try:
                prediction = await inference_fn(transaction)
                self.processed += 1

                if self.log_fn is not None:
                    await self.log_fn(transaction, prediction)

            except Exception:
                logger.exception(
                    "Error processing transaction %s",
                    transaction.get("transaction_id", "unknown"),
                )
            finally:
                queue.task_done()

    def stop(self) -> None:
        """Signal the consumer to exit its loop."""
        self._stopped = True
