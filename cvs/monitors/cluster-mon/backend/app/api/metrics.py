"""
Metrics API endpoints.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional

router = APIRouter()


@router.get("/latest")
async def get_latest_metrics() -> Dict[str, Any]:
    """
    Get the latest metrics for all nodes.

    Returns the most recent snapshot of all collected metrics.
    """
    from app.main import app_state

    if not app_state.latest_metrics:
        raise HTTPException(status_code=503, detail="No metrics available yet")

    return app_state.latest_metrics


@router.get("/history")
async def get_metrics_history(
    node: Optional[str] = Query(None, description="Filter by node hostname"),
    metric_type: Optional[str] = Query(None, description="Filter by metric type (gpu|nic)"),
    duration: int = Query(3600, description="Time range in seconds (default: 1 hour)"),
) -> Dict[str, Any]:
    """
    Get historical metrics.

    Note: This is a placeholder. In production, this would query InfluxDB.

    Args:
        node: Optional node filter
        metric_type: Optional metric type filter
        duration: Time range in seconds

    Returns:
        Historical metrics data
    """
    # TODO: Implement InfluxDB query
    return {
        "message": "Historical metrics not yet implemented",
        "note": "This will query InfluxDB in production",
        "params": {
            "node": node,
            "metric_type": metric_type,
            "duration": duration,
        },
    }
