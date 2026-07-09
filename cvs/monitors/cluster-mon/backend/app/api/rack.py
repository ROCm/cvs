"""
FastAPI router for rack-level IFoE monitoring endpoints.

Endpoints:
  GET  /api/rack/config         → current rack settings
  POST /api/rack/config         → save rack settings + store passwords in memory
  POST /api/rack/config/reload  → restart RackCollector with new settings
  GET  /api/rack/ifoe           → latest collected IFoE data
  POST /api/rack/ifoe/refresh   → trigger immediate collection
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from app.core.rack_config import RackSettings, load_rack_config, save_rack_config
from app.models.rack_models import (
    IFoEDataResponse,
    RackConfigResponse,
    RackConfigUpdateRequest,
    RefreshResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_app_state(request: Request) -> Any:
    """Retrieve the global AppState from the ASGI app."""
    # app_state is a module-level singleton in main.py — import lazily to
    # avoid circular imports at module load time.
    from app.main import app_state  # type: ignore[import]

    return app_state


# ---------------------------------------------------------------------------
# Config endpoints
# ---------------------------------------------------------------------------


@router.get("/config", response_model=RackConfigResponse)
async def get_rack_config(request: Request):
    """Return current rack.yaml settings (no passwords)."""
    app_state = _get_app_state(request)
    settings: RackSettings = getattr(app_state, "rack_settings", None) or load_rack_config()
    return RackConfigResponse(
        settings=settings,
        compute_tray_count=len(settings.compute_trays),
        switch_tray_count=len(settings.switch_trays),
    )


@router.post("/config")
async def update_rack_config(request: Request, body: RackConfigUpdateRequest):
    """
    Save rack configuration.

    Passwords are stored in app_state.rack_passwords only — never persisted.
    """
    app_state = _get_app_state(request)

    new_settings = RackSettings(
        compute_trays=body.compute_trays,
        switch_trays=body.switch_trays,
        compute_ssh=body.compute_ssh,
        switch_ssh=body.switch_ssh,
        poll_interval=body.poll_interval,
    )

    # Persist (sans passwords)
    try:
        save_rack_config(new_settings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save rack.yaml: {e}") from e

    # Store in memory
    app_state.rack_settings = new_settings

    # Store passwords in memory (never written to disk)
    if not hasattr(app_state, "rack_passwords"):
        app_state.rack_passwords = {}
    if body.compute_password is not None:
        app_state.rack_passwords["compute"] = body.compute_password
    if body.switch_password is not None:
        app_state.rack_passwords["switch"] = body.switch_password

    logger.info(
        f"Rack config updated: {len(new_settings.compute_trays)} compute trays, "
        f"{len(new_settings.switch_trays)} switch trays"
    )
    return {
        "success": True,
        "compute_tray_count": len(new_settings.compute_trays),
        "switch_tray_count": len(new_settings.switch_trays),
    }


@router.post("/config/reload")
async def reload_rack_config(request: Request):
    """
    Reload rack settings and restart the RackCollector.

    If the collector is already running it is cancelled and restarted so that
    the new settings (tray lists, poll interval) take effect immediately.
    """
    app_state = _get_app_state(request)

    from app.main import _start_collector_task  # type: ignore[import]
    from app.collectors.rack_collector import RackCollector

    rack_cfg = getattr(app_state, "rack_settings", None) or load_rack_config()
    app_state.rack_settings = rack_cfg

    # Cancel existing rack collector task if running
    existing_task = app_state.collector_tasks.get("rack")
    if existing_task and not existing_task.done():
        existing_task.cancel()
        try:
            await existing_task
        except Exception:
            pass

    # Only start if there are trays to collect from
    if rack_cfg.compute_trays or rack_cfg.switch_trays:
        collector = RackCollector()
        app_state.collectors["rack"] = collector
        app_state.collector_tasks["rack"] = _start_collector_task(collector)
        logger.info("RackCollector (re)started after config reload")
        return {"success": True, "message": "RackCollector restarted", "collecting": True}
    else:
        logger.info("RackCollector not started — no trays configured")
        return {"success": True, "message": "No trays configured — collector not started", "collecting": False}


# ---------------------------------------------------------------------------
# IFoE data endpoints
# ---------------------------------------------------------------------------


@router.get("/ifoe", response_model=IFoEDataResponse)
async def get_ifoe_data(request: Request):
    """Return the latest IFoE collection result."""
    app_state = _get_app_state(request)

    # Check collector_results first (preferred)
    collector_result = app_state.collector_results.get("rack")
    if collector_result and collector_result.data:
        data = collector_result.data
        return IFoEDataResponse(
            compute_devices=data.get("compute_devices", {}),
            compute_ports=data.get("compute_ports", {}),
            compute_port_stats=data.get("compute_port_stats", {}),
            switch_vlan=data.get("switch_vlan", {}),
            switch_mac=data.get("switch_mac", {}),
            topology=data.get("topology", []),
            switch_overview=data.get("switch_overview", {}),
            switch_metrics=data.get("switch_metrics", {}),
            last_updated=data.get("last_updated"),
            errors=data.get("errors", {}),
            state=str(collector_result.state),
        )

    # Fallback to cached rack data
    latest = getattr(app_state, "latest_rack_data", None)
    if latest:
        return IFoEDataResponse(
            compute_devices=latest.get("compute_devices", {}),
            compute_ports=latest.get("compute_ports", {}),
            compute_port_stats=latest.get("compute_port_stats", {}),
            switch_vlan=latest.get("switch_vlan", {}),
            switch_mac=latest.get("switch_mac", {}),
            topology=latest.get("topology", []),
            last_updated=latest.get("last_updated"),
            errors=latest.get("errors", {}),
            state="ok",
        )

    return IFoEDataResponse(state="no_data")


@router.post("/ifoe/refresh", response_model=RefreshResponse)
async def refresh_ifoe_data(request: Request):
    """
    Trigger an immediate collection cycle.

    If the RackCollector is running, signals its refresh event so it
    starts a new collection cycle without waiting for poll_interval.
    """
    app_state = _get_app_state(request)

    from app.collectors.rack_collector import RackCollector

    collector = app_state.collectors.get("rack")
    if isinstance(collector, RackCollector):
        collector.trigger_refresh()
        return RefreshResponse(message="Refresh triggered", triggered=True)

    # Collector not running — try to start it
    rack_cfg = getattr(app_state, "rack_settings", None) or load_rack_config()
    if rack_cfg.compute_trays or rack_cfg.switch_trays:
        from app.main import _start_collector_task  # type: ignore[import]

        new_collector = RackCollector()
        app_state.collectors["rack"] = new_collector
        app_state.collector_tasks["rack"] = _start_collector_task(new_collector)
        return RefreshResponse(message="RackCollector started and refresh triggered", triggered=True)

    return RefreshResponse(message="No trays configured — nothing to refresh", triggered=False)
