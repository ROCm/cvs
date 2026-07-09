"""
Rack-level configuration models for IFoE monitoring.

Stores compute tray and switch tray IPs with separate SSH credentials.
Passwords are never persisted to rack.yaml — they live in AppState.rack_passwords only.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import yaml
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

# Config file path relative to the repo root (same pattern as cluster.yaml)
_CONFIG_PATHS = [
    "config/rack.yaml",
    "/app/config/rack.yaml",
]


class RackTraySSHConfig(BaseModel):
    username: str = "admin"
    auth_method: str = "key"  # "key" or "password"
    key_file: Optional[str] = None
    timeout: int = 30

    @field_validator("auth_method")
    @classmethod
    def _validate_auth_method(cls, v: str) -> str:
        if v not in ("key", "password"):
            raise ValueError("auth_method must be 'key' or 'password'")
        return v


class RackSettings(BaseModel):
    compute_trays: List[str] = []  # IPs/hostnames, max 18
    switch_trays: List[str] = []  # IPs/hostnames, max 6
    compute_ssh: RackTraySSHConfig = RackTraySSHConfig()
    switch_ssh: RackTraySSHConfig = RackTraySSHConfig()
    poll_interval: int = 300

    @field_validator("compute_trays")
    @classmethod
    def _validate_compute_trays(cls, v: List[str]) -> List[str]:
        if len(v) > 18:
            raise ValueError("compute_trays max is 18")
        return [t.strip() for t in v if t.strip()]

    @field_validator("switch_trays")
    @classmethod
    def _validate_switch_trays(cls, v: List[str]) -> List[str]:
        if len(v) > 6:
            raise ValueError("switch_trays max is 6")
        return [t.strip() for t in v if t.strip()]


def _find_config_path() -> Optional[str]:
    """Locate rack.yaml by checking known paths."""
    for p in _CONFIG_PATHS:
        expanded = os.path.expanduser(p)
        if os.path.exists(expanded):
            return expanded
    return None


def load_rack_config() -> RackSettings:
    """Load rack config from rack.yaml. Returns defaults if file not found."""
    path = _find_config_path()
    if path is None:
        logger.info("rack.yaml not found — using default RackSettings")
        return RackSettings()
    try:
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh) or {}
        settings = RackSettings(**raw)
        logger.info(
            f"Loaded rack config from {path}: "
            f"{len(settings.compute_trays)} compute trays, "
            f"{len(settings.switch_trays)} switch trays"
        )
        return settings
    except Exception as e:
        logger.error(f"Failed to load rack config from {path}: {e}")
        return RackSettings()


def save_rack_config(settings: RackSettings) -> None:
    """Persist rack config to rack.yaml. Passwords are NOT included."""
    # Determine write path: prefer first writable candidate
    write_path: Optional[str] = None
    for p in _CONFIG_PATHS:
        expanded = os.path.expanduser(p)
        parent = os.path.dirname(expanded) or "."
        if os.path.exists(parent) and os.access(parent, os.W_OK):
            write_path = expanded
            break

    if write_path is None:
        raise RuntimeError(f"Cannot find writable directory for rack.yaml. Checked: {_CONFIG_PATHS}")

    # Serialize — exclude passwords (never persisted)
    data = settings.model_dump()
    # Strip any accidentally included password fields
    for section in ("compute_ssh", "switch_ssh"):
        data[section].pop("password", None)

    os.makedirs(os.path.dirname(write_path) or ".", exist_ok=True)
    with open(write_path, "w") as fh:
        yaml.safe_dump(data, fh, default_flow_style=False)
    logger.info(f"Saved rack config to {write_path}")
