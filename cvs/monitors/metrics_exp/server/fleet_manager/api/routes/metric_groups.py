"""API routes for metric group management."""

import logging
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ...models import get_db, MetricGroup

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/metric-groups", tags=["Metric Groups"])


# Pydantic schemas
class MetricGroupCreate(BaseModel):
    name: str
    description: Optional[str] = None
    gpu_utilization: bool = True
    gpu_memory: bool = True
    gpu_temperature: bool = True
    gpu_power: bool = True
    gpu_fan: bool = False
    gpu_clocks: bool = False
    gpu_pcie: bool = False
    gpu_ecc: bool = False
    node_cpu: bool = True
    node_memory: bool = True
    node_disk: bool = True
    node_network: bool = False
    collect_logs: bool = True
    log_patterns: List[str] = []


class MetricGroupUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    gpu_utilization: Optional[bool] = None
    gpu_memory: Optional[bool] = None
    gpu_temperature: Optional[bool] = None
    gpu_power: Optional[bool] = None
    gpu_fan: Optional[bool] = None
    gpu_clocks: Optional[bool] = None
    gpu_pcie: Optional[bool] = None
    gpu_ecc: Optional[bool] = None
    node_cpu: Optional[bool] = None
    node_memory: Optional[bool] = None
    node_disk: Optional[bool] = None
    node_network: Optional[bool] = None
    collect_logs: Optional[bool] = None
    log_patterns: Optional[List[str]] = None


class MetricGroupResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    gpu_utilization: bool
    gpu_memory: bool
    gpu_temperature: bool
    gpu_power: bool
    gpu_fan: bool
    gpu_clocks: bool
    gpu_pcie: bool
    gpu_ecc: bool
    node_cpu: bool
    node_memory: bool
    node_disk: bool
    node_network: bool
    collect_logs: bool
    log_patterns: List[str]
    node_group_count: int

    class Config:
        from_attributes = True


def metric_group_to_response(mg: MetricGroup) -> MetricGroupResponse:
    """Convert DB model to response schema."""
    return MetricGroupResponse(
        id=mg.id,
        name=mg.name,
        description=mg.description,
        gpu_utilization=mg.gpu_utilization,
        gpu_memory=mg.gpu_memory,
        gpu_temperature=mg.gpu_temperature,
        gpu_power=mg.gpu_power,
        gpu_fan=mg.gpu_fan,
        gpu_clocks=mg.gpu_clocks,
        gpu_pcie=mg.gpu_pcie,
        gpu_ecc=mg.gpu_ecc,
        node_cpu=mg.node_cpu,
        node_memory=mg.node_memory,
        node_disk=mg.node_disk,
        node_network=mg.node_network,
        collect_logs=mg.collect_logs,
        log_patterns=mg.log_patterns or [],
        node_group_count=len(mg.node_groups) if mg.node_groups else 0,
    )


@router.get("", response_model=List[MetricGroupResponse])
def list_metric_groups(db: Session = Depends(get_db)):
    """List all metric groups."""
    groups = db.query(MetricGroup).all()
    return [metric_group_to_response(g) for g in groups]


@router.post("", response_model=MetricGroupResponse)
def create_metric_group(
    group: MetricGroupCreate,
    db: Session = Depends(get_db),
):
    """Create a new metric group."""
    # Check for duplicate name
    existing = db.query(MetricGroup).filter(MetricGroup.name == group.name).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Metric group '{group.name}' already exists")

    db_group = MetricGroup(
        name=group.name,
        description=group.description,
        gpu_utilization=group.gpu_utilization,
        gpu_memory=group.gpu_memory,
        gpu_temperature=group.gpu_temperature,
        gpu_power=group.gpu_power,
        gpu_fan=group.gpu_fan,
        gpu_clocks=group.gpu_clocks,
        gpu_pcie=group.gpu_pcie,
        gpu_ecc=group.gpu_ecc,
        node_cpu=group.node_cpu,
        node_memory=group.node_memory,
        node_disk=group.node_disk,
        node_network=group.node_network,
        collect_logs=group.collect_logs,
        log_patterns=group.log_patterns,
    )
    db.add(db_group)
    db.commit()
    db.refresh(db_group)

    logger.info(f"Created metric group: {group.name}")
    return metric_group_to_response(db_group)


@router.get("/{group_id}", response_model=MetricGroupResponse)
def get_metric_group(group_id: int, db: Session = Depends(get_db)):
    """Get a metric group by ID."""
    group = db.query(MetricGroup).filter(MetricGroup.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Metric group not found")
    return metric_group_to_response(group)


@router.put("/{group_id}", response_model=MetricGroupResponse)
def update_metric_group(
    group_id: int,
    update: MetricGroupUpdate,
    db: Session = Depends(get_db),
):
    """Update a metric group."""
    group = db.query(MetricGroup).filter(MetricGroup.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Metric group not found")

    # Check for duplicate name if name is being changed
    if update.name and update.name != group.name:
        existing = db.query(MetricGroup).filter(MetricGroup.name == update.name).first()
        if existing:
            raise HTTPException(status_code=400, detail=f"Metric group '{update.name}' already exists")

    update_data = update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        if hasattr(group, field):
            setattr(group, field, value)

    db.commit()
    db.refresh(group)

    logger.info(f"Updated metric group: {group.name}")
    return metric_group_to_response(group)


@router.delete("/{group_id}")
def delete_metric_group(group_id: int, force: bool = False, db: Session = Depends(get_db)):
    """Delete a metric group.

    Args:
        group_id: The ID of the metric group to delete
        force: If True, unassign the metric group from associated node groups before deleting
    """
    group = db.query(MetricGroup).filter(MetricGroup.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Metric group not found")

    if group.name == "default":
        raise HTTPException(status_code=400, detail="Cannot delete the default metric group")

    # Handle associated node groups
    if group.node_groups:
        if not force:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot delete metric group '{group.name}' - it has {len(group.node_groups)} node groups associated. Use force=true to unassign and delete.",
            )
        # Force delete: unassign metric group from all node groups
        for ng in group.node_groups:
            ng.metric_group_id = None
            logger.info(f"Unassigned metric group '{group.name}' from node group '{ng.name}'")
        db.commit()

    db.delete(group)
    db.commit()

    logger.info(f"Deleted metric group: {group.name}")
    return {"message": f"Metric group '{group.name}' deleted"}


@router.get("/{group_id}/dashboard-panels")
def get_dashboard_panels(group_id: int, db: Session = Depends(get_db)):
    """Get the dashboard panel configuration based on enabled metrics."""
    group = db.query(MetricGroup).filter(MetricGroup.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Metric group not found")

    panels = []
    panel_id = 1

    # Summary row
    panels.append(
        {"id": panel_id, "title": "Fleet Summary", "type": "row", "gridPos": {"h": 1, "w": 24, "x": 0, "y": 0}}
    )
    panel_id += 1

    y_offset = 1
    x_offset = 0

    # GPU metrics
    if group.gpu_utilization:
        panels.append(
            {
                "id": panel_id,
                "title": "Avg GPU Utilization",
                "type": "gauge",
                "metric": "avg(amd_gpu_utilization_percent)",
                "gridPos": {"h": 4, "w": 4, "x": x_offset, "y": y_offset},
            }
        )
        panel_id += 1
        x_offset += 4

    if group.gpu_memory:
        panels.append(
            {
                "id": panel_id,
                "title": "Avg Memory Used",
                "type": "gauge",
                "metric": "avg(amd_gpu_memory_utilization_percent)",
                "gridPos": {"h": 4, "w": 4, "x": x_offset, "y": y_offset},
            }
        )
        panel_id += 1
        x_offset += 4

    if group.gpu_temperature:
        panels.append(
            {
                "id": panel_id,
                "title": "Max Temperature",
                "type": "stat",
                "metric": "max(amd_gpu_temperature_junction_celsius)",
                "gridPos": {"h": 4, "w": 4, "x": x_offset, "y": y_offset},
            }
        )
        panel_id += 1
        x_offset += 4

    if group.gpu_power:
        panels.append(
            {
                "id": panel_id,
                "title": "Total Power",
                "type": "stat",
                "metric": "sum(amd_gpu_power_watts)",
                "gridPos": {"h": 4, "w": 4, "x": x_offset, "y": y_offset},
            }
        )
        panel_id += 1
        x_offset += 4

    y_offset += 5
    x_offset = 0

    # Time series panels
    if group.gpu_utilization:
        panels.append(
            {
                "id": panel_id,
                "title": "GPU Utilization Over Time",
                "type": "timeseries",
                "metric": "amd_gpu_utilization_percent",
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": y_offset},
            }
        )
        panel_id += 1

    if group.gpu_memory:
        panels.append(
            {
                "id": panel_id,
                "title": "Memory Utilization Over Time",
                "type": "timeseries",
                "metric": "amd_gpu_memory_utilization_percent",
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": y_offset},
            }
        )
        panel_id += 1

    y_offset += 9

    if group.gpu_temperature:
        panels.append(
            {
                "id": panel_id,
                "title": "GPU Temperature",
                "type": "timeseries",
                "metric": "amd_gpu_temperature_junction_celsius",
                "gridPos": {"h": 8, "w": 8, "x": 0, "y": y_offset},
            }
        )
        panel_id += 1

    if group.gpu_power:
        panels.append(
            {
                "id": panel_id,
                "title": "GPU Power Usage",
                "type": "timeseries",
                "metric": "amd_gpu_power_watts",
                "gridPos": {"h": 8, "w": 8, "x": 8, "y": y_offset},
            }
        )
        panel_id += 1

    if group.gpu_fan:
        panels.append(
            {
                "id": panel_id,
                "title": "Fan Speed",
                "type": "timeseries",
                "metric": "amd_gpu_fan_speed_percent",
                "gridPos": {"h": 8, "w": 8, "x": 16, "y": y_offset},
            }
        )
        panel_id += 1

    y_offset += 9

    if group.gpu_clocks:
        panels.append(
            {
                "id": panel_id,
                "title": "GPU Clock Speeds",
                "type": "timeseries",
                "metric": "amd_gpu_sclk_mhz",
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": y_offset},
            }
        )
        panel_id += 1

    if group.gpu_pcie:
        panels.append(
            {
                "id": panel_id,
                "title": "PCIe Bandwidth",
                "type": "timeseries",
                "metric": "amd_gpu_pcie_bandwidth_mbps",
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": y_offset},
            }
        )
        panel_id += 1

    y_offset += 9

    if group.gpu_ecc:
        panels.append(
            {
                "id": panel_id,
                "title": "ECC Correctable Errors",
                "type": "stat",
                "metric": "sum(amd_gpu_ecc_errors_corrected_total)",
                "gridPos": {"h": 4, "w": 6, "x": 0, "y": y_offset},
            }
        )
        panel_id += 1
        panels.append(
            {
                "id": panel_id,
                "title": "ECC Uncorrectable Errors",
                "type": "stat",
                "metric": "sum(amd_gpu_ecc_errors_uncorrected_total)",
                "gridPos": {"h": 4, "w": 6, "x": 6, "y": y_offset},
            }
        )
        panel_id += 1

    return {
        "metric_group": group.name,
        "enabled_metrics": {
            "gpu_utilization": group.gpu_utilization,
            "gpu_memory": group.gpu_memory,
            "gpu_temperature": group.gpu_temperature,
            "gpu_power": group.gpu_power,
            "gpu_fan": group.gpu_fan,
            "gpu_clocks": group.gpu_clocks,
            "gpu_pcie": group.gpu_pcie,
            "gpu_ecc": group.gpu_ecc,
            "node_cpu": group.node_cpu,
            "node_memory": group.node_memory,
            "node_disk": group.node_disk,
            "node_network": group.node_network,
        },
        "panels": panels,
    }
