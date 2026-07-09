"""CPU and memory data API endpoints."""

from fastapi import APIRouter, Request
from typing import Any

router = APIRouter()


def _get_app_state(request: Request) -> Any:
    from app.main import app_state

    return app_state


@router.get("/data")
async def get_cpu_data(request: Request):
    """Return latest CPU and memory summary + metrics for all nodes."""
    state = _get_app_state(request)

    # Try collector_results first, then cached data
    result = state.collector_results.get("cpu_info")
    if result and result.data:
        return {**result.data, "state": str(result.state)}

    latest = getattr(state, "latest_cpu_data", None)
    if latest:
        return {**latest, "state": "ok"}

    return {"summary": {}, "metrics": {}, "errors": {}, "state": "no_data"}
