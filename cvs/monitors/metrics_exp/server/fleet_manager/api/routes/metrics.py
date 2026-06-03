"""Metric configuration endpoints."""

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..schemas import (
    MetricConfigCreate,
    MetricConfigResponse,
)
from ...models import get_db, MetricConfig

router = APIRouter(prefix="/metrics", tags=["Metric Configurations"])
logger = logging.getLogger(__name__)


# Default GPU metric categories with their metric names
GPU_METRIC_CATEGORIES = {
    "fleet_health": {
        "name": "Fleet Health",
        "description": "Overall GPU health status",
        "metrics": [
            "gpu_health",
            "gpu_afid_errors",
        ],
    },
    "thermal_power": {
        "name": "Thermal & Power",
        "description": "Temperature and power consumption metrics",
        "metrics": [
            "gpu_edge_temperature",
            "gpu_junction_temperature",
            "gpu_memory_temperature",
            "gpu_package_power",
            "gpu_power_usage",
            "gpu_energy_consumed",
        ],
    },
    "utilization": {
        "name": "GPU Utilization",
        "description": "GPU compute and memory utilization",
        "metrics": [
            "gpu_gfx_activity",
            "gpu_umc_activity",
            "gpu_process_cu_occupancy",
        ],
    },
    "memory": {
        "name": "GPU Memory",
        "description": "VRAM usage and bandwidth",
        "metrics": [
            "gpu_total_vram",
            "gpu_used_vram",
            "gpu_free_vram",
            "gpu_vram_max_bandwidth",
        ],
    },
    "pcie": {
        "name": "PCIe",
        "description": "PCIe link status and errors",
        "metrics": [
            "pcie_speed",
            "pcie_max_speed",
            "pcie_bandwidth",
            "pcie_replay_count",
            "pcie_recovery_count",
        ],
    },
    "xgmi": {
        "name": "XGMI",
        "description": "GPU interconnect metrics",
        "metrics": [
            "gpu_xgmi_link_rx",
            "gpu_xgmi_link_tx",
        ],
    },
    "ecc": {
        "name": "ECC",
        "description": "Error Correcting Code counters",
        "metrics": [
            "gpu_ecc_correct_sdma",
            "gpu_ecc_correct_gfx",
            "gpu_ecc_correct_mmhub",
            "gpu_ecc_uncorrect_sdma",
            "gpu_ecc_uncorrect_gfx",
            "gpu_ecc_uncorrect_mmhub",
            "gpu_ecc_uncorrect_umc",
            "gpu_ecc_uncorrect_mpio",
        ],
    },
    "ras": {
        "name": "RAS (Reliability)",
        "description": "Reliability, Availability, and Serviceability metrics",
        "metrics": [
            "gpu_violation_current_accumulated_counter",
            "gpu_violation_processor_hot_residency_accumulated",
            "gpu_violation_ppt_residency_accumulated",
            "gpu_violation_socket_thermal_residency_accumulated",
            "gpu_violation_vr_thermal_residency_accumulated",
            "gpu_violation_hbm_thermal_residency_accumulated",
        ],
    },
}

NODE_METRIC_CATEGORIES = {
    "cpu_metrics": {
        "name": "CPU Metrics",
        "description": "CPU utilization and load",
        "metrics": [
            "node_cpu_seconds_total",
            "node_load1",
            "node_load5",
            "node_load15",
        ],
    },
    "memory_metrics": {
        "name": "Memory Metrics",
        "description": "System memory usage",
        "metrics": [
            "node_memory_MemTotal_bytes",
            "node_memory_MemAvailable_bytes",
            "node_memory_MemFree_bytes",
            "node_memory_Buffers_bytes",
            "node_memory_Cached_bytes",
        ],
    },
    "disk_metrics": {
        "name": "Disk Metrics",
        "description": "Disk I/O and space",
        "metrics": [
            "node_disk_io_time_seconds_total",
            "node_disk_read_bytes_total",
            "node_disk_written_bytes_total",
            "node_filesystem_avail_bytes",
            "node_filesystem_size_bytes",
        ],
    },
    "network_metrics": {
        "name": "Network Metrics",
        "description": "Network interface statistics",
        "metrics": [
            "node_network_receive_bytes_total",
            "node_network_transmit_bytes_total",
            "node_network_receive_errs_total",
            "node_network_transmit_errs_total",
        ],
    },
}


@router.get("/categories")
def get_metric_categories():
    """Get all available metric categories and their metrics."""
    return {
        "gpu_metrics": GPU_METRIC_CATEGORIES,
        "node_metrics": NODE_METRIC_CATEGORIES,
    }


@router.get("/configs", response_model=List[MetricConfigResponse])
def list_metric_configs(db: Session = Depends(get_db)):
    """List all metric configurations."""
    return db.query(MetricConfig).all()


@router.post("/configs", response_model=MetricConfigResponse, status_code=201)
def create_metric_config(
    config: MetricConfigCreate,
    db: Session = Depends(get_db),
):
    """Create a new metric configuration."""
    existing = db.query(MetricConfig).filter(MetricConfig.name == config.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Config with this name already exists")

    db_config = MetricConfig(**config.model_dump())
    db.add(db_config)
    db.commit()
    db.refresh(db_config)

    return db_config


@router.get("/configs/{config_id}", response_model=MetricConfigResponse)
def get_metric_config(config_id: int, db: Session = Depends(get_db)):
    """Get a specific metric configuration."""
    config = db.query(MetricConfig).filter(MetricConfig.id == config_id).first()
    if not config:
        raise HTTPException(status_code=404, detail="Config not found")
    return config


@router.delete("/configs/{config_id}", status_code=204)
def delete_metric_config(config_id: int, db: Session = Depends(get_db)):
    """Delete a metric configuration."""
    config = db.query(MetricConfig).filter(MetricConfig.id == config_id).first()
    if not config:
        raise HTTPException(status_code=404, detail="Config not found")

    if config.name == "default":
        raise HTTPException(status_code=400, detail="Cannot delete default config")

    db.delete(config)
    db.commit()


@router.get("/log-patterns")
def get_default_log_patterns():
    """Get default log patterns for critical error detection."""
    return {
        "critical_patterns": [
            {"pattern": "ECC|UE|CE", "description": "ECC errors (Uncorrectable/Correctable)"},
            {"pattern": "RAS|ras_error", "description": "RAS events"},
            {"pattern": "xgmi|XGMI.*error", "description": "XGMI link failures"},
            {"pattern": "GPU hang|gpu reset", "description": "GPU hangs and resets"},
            {"pattern": "amdgpu.*timeout", "description": "AMD GPU driver timeouts"},
            {"pattern": "Hardware Error|MCE", "description": "Hardware/Machine Check Errors"},
            {"pattern": "machine check", "description": "Machine check exceptions"},
            {"pattern": "thermal throttle|temperature", "description": "Thermal issues"},
            {"pattern": "overheat", "description": "Overheating warnings"},
            {"pattern": "PCIe.*error|pcie.*fail", "description": "PCIe errors"},
        ],
        "severity_keywords": {
            "critical": ["fatal", "panic", "crash", "hung", "reset"],
            "error": ["error", "fail", "timeout", "uncorrect"],
            "warning": ["warn", "throttle", "correct"],
        },
    }
