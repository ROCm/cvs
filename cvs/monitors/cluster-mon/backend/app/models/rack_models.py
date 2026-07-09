"""
Pydantic response models for the rack API endpoints.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from app.core.rack_config import RackSettings, RackTraySSHConfig


# ---------------------------------------------------------------------------
# Config endpoint models
# ---------------------------------------------------------------------------


class RackConfigResponse(BaseModel):
    """Response for GET /api/rack/config."""

    settings: RackSettings
    compute_tray_count: int
    switch_tray_count: int


class RackConfigUpdateRequest(BaseModel):
    """Request body for POST /api/rack/config."""

    compute_trays: List[str] = []
    switch_trays: List[str] = []
    compute_ssh: RackTraySSHConfig = RackTraySSHConfig()
    switch_ssh: RackTraySSHConfig = RackTraySSHConfig()
    poll_interval: int = 300
    # Passwords: optional, stored in memory only
    compute_password: Optional[str] = None
    switch_password: Optional[str] = None


# ---------------------------------------------------------------------------
# IFoE data models
# ---------------------------------------------------------------------------


class IFoEDataResponse(BaseModel):
    """Response for GET /api/rack/ifoe."""

    compute_devices: Dict[str, List[Dict[str, Any]]] = {}
    compute_ports: Dict[str, List[Dict[str, Any]]] = {}
    # Port statistics per tray: {host: {mac: [...], fec: [...], ifcp: [...], pfc: [...]}}
    compute_port_stats: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    switch_vlan: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    switch_mac: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    topology: List[Dict[str, Any]] = []
    # Switch platform/system overview per host
    switch_overview: Dict[str, Any] = {}
    # Switch interface metrics/counters per host
    switch_metrics: Dict[str, Any] = {}
    last_updated: Optional[str] = None
    errors: Dict[str, str] = {}
    state: str = "no_data"


class RefreshResponse(BaseModel):
    """Response for POST /api/rack/ifoe/refresh."""

    message: str
    triggered: bool
