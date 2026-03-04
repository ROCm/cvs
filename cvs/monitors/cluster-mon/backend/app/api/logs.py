"""
System logs API endpoints.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any

router = APIRouter()


@router.get("/dmesg")
async def get_dmesg_errors() -> Dict[str, Any]:
    """
    Get dmesg error logs from all cluster nodes.

    Collects: :emerg, :alert, :crit, :err level messages
    """
    from app.main import app_state

    if not app_state.ssh_manager:
        raise HTTPException(status_code=503, detail="SSH manager not initialized")

    try:
        from app.collectors.logs_collector import LogsCollector

        collector = LogsCollector()
        logs_data = await collector.collect_all_logs(app_state.ssh_manager)

        return logs_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to collect logs: {str(e)}")
