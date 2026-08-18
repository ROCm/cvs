"""
On-demand cluster metrics snapshot gallery API.

Capture up to five snapshots, list them, diff any pair, delete one or all.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.core.snapshot import (
    CAPTURE_TIMEOUT_S,
    MAX_SNAPSHOTS,
    SnapshotEmptyError,
    SnapshotFullError,
    capture_snapshot,
    compare_cluster_metrics_snapshots,
    create_cluster_metrics_snapshot,
    diff_warnings,
    stored_snapshot_list_item,
)

logger = logging.getLogger(__name__)
router = APIRouter()

_snapshot_lock = asyncio.Lock()


class CaptureRequest(BaseModel):
    label: Optional[str] = Field(default=None, max_length=80)


class DiffRequest(BaseModel):
    before_id: str
    after_id: str


def _store():
    from app.main import app_state

    if app_state.snapshot_store is None:
        raise HTTPException(status_code=503, detail="Snapshot store is not initialized")
    return app_state.snapshot_store


def _require_ssh_manager():
    from app.main import app_state

    if not app_state.ssh_manager:
        raise HTTPException(status_code=503, detail="SSH manager not initialized. Configure nodes first.")
    return app_state.ssh_manager


def _list_payload(in_progress: bool) -> Dict[str, Any]:
    store = _store()
    items = [stored_snapshot_list_item(snap) for snap in store.list_newest_first()]
    return {
        "snapshots": items,
        "count": store.count(),
        "max": MAX_SNAPSHOTS,
        "in_progress": in_progress,
    }


@router.get("")
@router.get("/")
async def list_snapshots() -> Dict[str, Any]:
    from app.main import app_state

    return _list_payload(app_state.snapshot_in_progress)


@router.post("")
@router.post("/")
async def capture_new_snapshot(body: Optional[CaptureRequest] = None) -> Any:
    from app.main import app_state

    ssh_manager = _require_ssh_manager()
    store = _store()
    body = body or CaptureRequest()

    if store.count() >= MAX_SNAPSHOTS:
        raise HTTPException(
            status_code=409,
            detail={"code": "full", "message": f"Gallery is full ({MAX_SNAPSHOTS}/5). Delete a snapshot first."},
        )

    async with _snapshot_lock:
        if app_state.snapshot_in_progress:
            raise HTTPException(
                status_code=409,
                detail={"code": "busy", "message": "Snapshot capture already in progress"},
            )
        app_state.snapshot_in_progress = True
        try:
            logger.info("Capturing cluster metrics snapshot (label=%s)", body.label)

            async def _collect():
                return await create_cluster_metrics_snapshot(ssh_manager)

            snap, persist_warning = await capture_snapshot(store, _collect, label=body.label)
        except SnapshotFullError:
            raise HTTPException(
                status_code=409,
                detail={"code": "full", "message": f"Gallery is full ({MAX_SNAPSHOTS}/5). Delete a snapshot first."},
            )
        except asyncio.TimeoutError:
            logger.error("Snapshot capture timed out after %ss", CAPTURE_TIMEOUT_S)
            raise HTTPException(
                status_code=504,
                detail=f"Snapshot capture timed out after {CAPTURE_TIMEOUT_S}s",
            )
        except SnapshotEmptyError:
            raise HTTPException(status_code=502, detail="Snapshot capture returned no node data")
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Snapshot capture failed")
            raise HTTPException(status_code=502, detail=f"Snapshot capture failed: {exc}") from exc
        finally:
            app_state.snapshot_in_progress = False

    item = stored_snapshot_list_item(snap)
    payload = {
        "success": True,
        "snapshot": item,
        "persist_warning": persist_warning,
        "count": store.count(),
        "max": MAX_SNAPSHOTS,
    }
    return JSONResponse(status_code=201, content=payload)


@router.delete("")
@router.delete("/")
async def clear_all_snapshots() -> Dict[str, Any]:
    store = _store()
    store.clear()
    return {"success": True, "count": 0, "max": MAX_SNAPSHOTS}


@router.post("/diff")
async def diff_snapshots(body: DiffRequest) -> Dict[str, Any]:
    store = _store()
    if body.before_id == body.after_id:
        raise HTTPException(status_code=400, detail="before_id and after_id must be different snapshots")

    before = store.get(body.before_id)
    after = store.get(body.after_id)
    if before is None:
        raise HTTPException(status_code=404, detail=f"Snapshot not found: {body.before_id}")
    if after is None:
        raise HTTPException(status_code=404, detail=f"Snapshot not found: {body.after_id}")

    diff_result = compare_cluster_metrics_snapshots(before.data, after.data)
    return {
        "success": True,
        "before_id": before.id,
        "after_id": after.id,
        "before_timestamp": before.captured_at,
        "after_timestamp": after.captured_at,
        "before_label": before.label,
        "after_label": after.label,
        "summary": diff_result["summary"],
        "rows": diff_result["rows"],
        "warnings": diff_warnings(before, after),
    }


@router.delete("/{snapshot_id}")
async def delete_snapshot(snapshot_id: str) -> Dict[str, Any]:
    store = _store()
    if not store.delete(snapshot_id):
        raise HTTPException(status_code=404, detail=f"Snapshot not found: {snapshot_id}")
    return {"success": True, "id": snapshot_id, "count": store.count(), "max": MAX_SNAPSHOTS}
