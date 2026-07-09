"""
Node group configuration — three SSH-managed host groups.

  gpu_nodes        — regular GPU nodes AND/OR compute trays (IFoE-capable)
  scale_up_switches — SONiC switches forming the IFoE scale-up fabric
  scale_out_switches — SONiC switches forming the scale-out (inter-rack) fabric

Each group has independent SSH credentials (username + key OR password) and
an optional jump host. All groups use the Go SSH daemon for execution.
Passwords are never persisted to disk.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import yaml
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

_CONFIG_PATHS = [
    "config/node_groups.yaml",
    "/app/config/node_groups.yaml",
]


class GroupSSHConfig(BaseModel):
    username: str = "root"
    auth_method: str = "key"
    key_file: str = "~/.ssh/id_rsa"
    timeout: int = 30

    @field_validator("auth_method")
    @classmethod
    def _check(cls, v: str) -> str:
        if v not in ("key", "password"):
            raise ValueError("auth_method must be 'key' or 'password'")
        return v


class GroupJumpHost(BaseModel):
    enabled: bool = False
    host: str = ""
    username: str = "root"
    auth_method: str = "key"
    key_file: str = "~/.ssh/id_rsa"


class NodeGroup(BaseModel):
    hosts: List[str] = []
    ssh: GroupSSHConfig = GroupSSHConfig()
    jump_host: GroupJumpHost = GroupJumpHost()

    @field_validator("hosts")
    @classmethod
    def _strip(cls, v: List[str]) -> List[str]:
        return [h.strip() for h in v if h.strip()]


class NodeGroupsSettings(BaseModel):
    gpu_nodes: NodeGroup = NodeGroup()
    scale_up_switches: NodeGroup = NodeGroup()
    scale_out_switches: NodeGroup = NodeGroup()
    poll_interval: int = 300


def _find_config() -> Optional[str]:
    for p in _CONFIG_PATHS:
        if os.path.exists(os.path.expanduser(p)):
            return os.path.expanduser(p)
    return None


def load_node_groups() -> NodeGroupsSettings:
    path = _find_config()
    if path is None:
        return NodeGroupsSettings()
    try:
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        s = NodeGroupsSettings(**raw)
        logger.info(
            f"Loaded node_groups.yaml: gpu={len(s.gpu_nodes.hosts)}, "
            f"scale_up={len(s.scale_up_switches.hosts)}, "
            f"scale_out={len(s.scale_out_switches.hosts)}"
        )
        return s
    except Exception as e:
        logger.error(f"Failed to load node_groups.yaml: {e}")
        return NodeGroupsSettings()


def save_node_groups(s: NodeGroupsSettings) -> None:
    write_path: Optional[str] = None
    for p in _CONFIG_PATHS:
        expanded = os.path.expanduser(p)
        parent = os.path.dirname(expanded) or "."
        if os.path.isdir(parent) and os.access(parent, os.W_OK):
            write_path = expanded
            break
    if write_path is None:
        raise RuntimeError(f"No writable path for node_groups.yaml in {_CONFIG_PATHS}")

    data = s.model_dump()
    for group in ("gpu_nodes", "scale_up_switches", "scale_out_switches"):
        data[group]["ssh"].pop("password", None)
        data[group]["jump_host"].pop("password", None)

    os.makedirs(os.path.dirname(write_path) or ".", exist_ok=True)
    with open(write_path, "w") as fh:
        yaml.safe_dump(data, fh, default_flow_style=False)
    logger.info(f"Saved node_groups.yaml to {write_path}")
