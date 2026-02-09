"""A/B test results API endpoint."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["ab-test"])


@router.get("/ab-test/results")
async def ab_test_results() -> Dict[str, Any]:
    """Return current A/B test metrics and significance results."""
    from src.api.app import get_state

    state = get_state()

    if state.ab_router is None:
        return {
            "enabled": False,
            "message": "A/B testing is not active",
        }

    results = state.ab_router.get_results()
    results["enabled"] = True
    return results
