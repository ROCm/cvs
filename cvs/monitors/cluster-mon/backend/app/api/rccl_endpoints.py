"""
RCCL monitoring REST API endpoints.
Phase 1: status, communicators, events, markers.
"""

import logging
from typing import Any, Optional
from fastapi import APIRouter, HTTPException, Query
from app.models.rccl_models import RCCLMarker

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/status")
async def get_rccl_status() -> dict[str, Any]:
    """
    Current RCCL job state and communicator health summary.
    Returns the latest snapshot from app_state.latest_rccl_snapshot.
    Falls back to {'state': 'no_job'} if no snapshot yet collected.
    """
    from app.main import app_state

    snapshot = getattr(app_state, 'latest_rccl_snapshot', None)
    if snapshot is None:
        return {"state": "no_job", "message": "No RCCL snapshot collected yet"}
    return snapshot


@router.get("/communicators")
async def get_rccl_communicators() -> list[dict]:
    """All communicators with per-rank detail from the latest snapshot."""
    from app.main import app_state

    snapshot = getattr(app_state, 'latest_rccl_snapshot', None)
    if snapshot is None:
        return []
    return snapshot.get("communicators", [])


@router.get("/communicators/{comm_hash}")
async def get_rccl_communicator(comm_hash: str) -> dict[str, Any]:
    """Single communicator deep-dive by hash."""
    from app.main import app_state

    snapshot = getattr(app_state, 'latest_rccl_snapshot', None)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="No snapshot available")
    for comm in snapshot.get("communicators", []):
        if comm.get("comm_hash") == comm_hash:
            return comm
    raise HTTPException(status_code=404, detail=f"Communicator {comm_hash!r} not found")


@router.get("/events")
async def get_rccl_events(
    since: Optional[float] = Query(None, description="Start timestamp (unix)"),
    until: Optional[float] = Query(None, description="End timestamp (unix)"),
    event_type: Optional[str] = Query(None, alias="type"),
) -> dict:
    """Filtered event log from Redis event stream (or in-memory fallback)."""
    from app.main import app_state
    import time

    data_store = getattr(app_state, 'rccl_data_store', None)
    if data_store is None:
        return {"events": [], "truncated": False}

    start = since or (time.time() - 3600)  # default: last hour
    end = until or time.time()
    events = await data_store.get_events_in_range(start, end)

    if event_type:
        events = [e for e in events if e.get("event_type") == event_type]

    return {
        "events": events,
        "truncated": data_store.is_memory_capped,
    }


@router.post("/markers", status_code=201)
async def post_rccl_marker(marker: RCCLMarker) -> dict[str, str]:
    """
    PyTorch callback endpoint for training step/loss markers.
    Stores marker as an event in the RCCL event stream.
    """
    from app.main import app_state
    import time

    event = marker.model_dump()
    event.setdefault("event_type", "training_marker")
    event.setdefault("timestamp", time.time())

    data_store = getattr(app_state, 'rccl_data_store', None)
    if data_store:
        await data_store.push_event(event)

    return {"status": "accepted"}
