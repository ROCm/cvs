"""Storage / disk data API — GET /api/disk/data"""

from fastapi import APIRouter, Request
from typing import Any

router = APIRouter()


def _state(request: Request) -> Any:
    from app.main import app_state

    return app_state


@router.get("/data")
async def get_storage_data(request: Request):
    s = _state(request)
    result = s.collector_results.get("storage")
    if result and result.data:
        return {**result.data, "state": str(result.state)}
    latest = getattr(s, "latest_storage_data", None)
    if latest:
        return {**latest, "state": "ok"}
    return {
        "block_devices": {},
        "filesystems": {},
        "io_stats": {},
        "nvme_devices": {},
        "mem_cache": {},
        "vm_stats": {},
        "top_io_procs": {},
        "errors": {},
        "state": "no_data",
    }
