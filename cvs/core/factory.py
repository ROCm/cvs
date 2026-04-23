"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from cvs.core.launchers.factory import build_launchers
from cvs.core.orchestrator import Orchestrator
from cvs.core.runtimes.factory import build_runtime
from cvs.core.transports.pssh import PsshTransport

if TYPE_CHECKING:
    from cvs.core.config import OrchestratorConfig
    from cvs.core.transports.base import Transport


def build_transport(transport_cfg: dict, log: logging.Logger) -> "Transport":
    """Build a Transport from a parsed cluster.json transport block.

    Today the only transport is pssh. Future transports (slurm, k8s, local)
    add a class + a branch here. Unknown transport names raise
    OrchestratorConfigError.
    """
    from cvs.core.errors import OrchestratorConfigError

    name = (transport_cfg or {}).get("name", "pssh").lower()
    if name == "pssh":
        node_dict = transport_cfg.get("node_dict") or {}
        if isinstance(node_dict, dict):
            hosts = list(node_dict.keys())
        else:
            hosts = [n.get("mgmt_ip") for n in node_dict]
        head_node_dict = transport_cfg.get("head_node_dict") or {}
        head_node = head_node_dict.get("name") if isinstance(head_node_dict, dict) else None
        if head_node is None:
            head_node = hosts[0] if hosts else None
        return PsshTransport(
            hosts=hosts,
            head_node=head_node,
            username=transport_cfg["username"],
            priv_key_file=transport_cfg["priv_key_file"],
            password=transport_cfg.get("password"),
            env_vars=transport_cfg.get("env_vars") or {},
            log=log,
            stop_on_errors=transport_cfg.get("stop_on_errors", False),
        )
    raise OrchestratorConfigError(f"transport '{name}' is not implemented")


def create_orchestrator(
    cfg: "OrchestratorConfig", log: Optional[logging.Logger] = None
) -> Orchestrator:
    """Build a fully-wired Orchestrator from an OrchestratorConfig.

    This is the one entry point new code uses. The exec_plugin CLI and the
    RCCL pytest fixtures both call it.
    """
    log = log or logging.getLogger("cvs")
    transport = build_transport(cfg.transport, log)
    runtime = build_runtime(cfg.runtime)
    launchers = build_launchers(cfg.launchers)
    return Orchestrator(transport, runtime, launchers, log=log)
