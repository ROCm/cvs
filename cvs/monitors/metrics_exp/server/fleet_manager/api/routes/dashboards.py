"""Dashboard management endpoints."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..schemas import DashboardInfo, DashboardList
from ...models import get_db, NodeGroup
from ...services import GrafanaProvisioner

router = APIRouter(prefix="/dashboards", tags=["Dashboards"])
logger = logging.getLogger(__name__)

grafana = GrafanaProvisioner()


@router.get("", response_model=DashboardList)
async def list_dashboards():
    """List all GPU Fleet monitoring dashboards."""
    dashboards = await grafana.list_dashboards(folder_uid="gpu-fleet")

    return DashboardList(
        dashboards=[
            DashboardInfo(
                uid=d.get("uid", ""),
                title=d.get("title", ""),
                url=d.get("url", ""),
            )
            for d in dashboards
        ]
    )


@router.get("/health")
async def check_grafana_health():
    """Check Grafana health status."""
    healthy = await grafana.check_health()
    return {"grafana": "healthy" if healthy else "unhealthy"}


@router.post("/nodegroup/{node_group_id}")
async def create_nodegroup_dashboard(
    node_group_id: int,
    db: Session = Depends(get_db),
):
    """Create or refresh dashboard for a node group."""
    ng = db.query(NodeGroup).filter(NodeGroup.id == node_group_id).first()
    if not ng:
        raise HTTPException(status_code=404, detail="Node group not found")

    uid = await grafana.provision_node_group_dashboard(ng.name)

    if uid:
        return {
            "uid": uid,
            "title": f"Node Group: {ng.name}",
            "url": f"/d/{uid}",
            "message": "Dashboard created/updated successfully",
        }

    raise HTTPException(status_code=500, detail="Failed to create dashboard")


@router.delete("/nodegroup/{node_group_id}")
async def delete_nodegroup_dashboard(
    node_group_id: int,
    db: Session = Depends(get_db),
):
    """Delete dashboard for a node group."""
    ng = db.query(NodeGroup).filter(NodeGroup.id == node_group_id).first()
    if not ng:
        raise HTTPException(status_code=404, detail="Node group not found")

    success = await grafana.remove_node_group_dashboard(ng.name)

    if success:
        return {"message": "Dashboard deleted successfully"}

    raise HTTPException(status_code=500, detail="Failed to delete dashboard")


@router.get("/templates")
async def list_dashboard_templates():
    """List available dashboard templates."""
    return {
        "templates": [
            {
                "id": "fleet_overview",
                "name": "Fleet Overview",
                "description": "High-level view of all node groups and GPU health",
            },
            {
                "id": "gpu_health",
                "name": "GPU Health",
                "description": "Detailed GPU health status and error tracking",
            },
            {
                "id": "thermal_power",
                "name": "Thermal & Power",
                "description": "Temperature and power consumption monitoring",
            },
            {
                "id": "pcie_xgmi",
                "name": "PCIe & XGMI",
                "description": "Interconnect and bus metrics",
            },
            {
                "id": "ecc_ras",
                "name": "ECC & RAS",
                "description": "Error correction and reliability metrics",
            },
            {
                "id": "gpu_utilization",
                "name": "GPU Utilization",
                "description": "GPU compute and memory utilization",
            },
            {
                "id": "cpu_system",
                "name": "CPU & System",
                "description": "Host CPU, memory, and disk metrics",
            },
            {
                "id": "logs_analysis",
                "name": "Logs Analysis",
                "description": "Critical log patterns and error analysis",
            },
        ]
    }
