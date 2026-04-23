'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

from __future__ import annotations

from typing import TYPE_CHECKING

from .docker import DockerRuntime
from .hostshell import HostShellRuntime

if TYPE_CHECKING:
    from .base import Runtime


def build_runtime(runtime_cfg: dict) -> "Runtime":
    """Build a Runtime instance from a parsed cluster.json runtime block.

    runtime_cfg shape::

        {"name": "hostshell" | "docker", "config": {...}}

    Unknown runtime names raise OrchestratorConfigError. There are no stub
    runtimes that pretend to work; future runtimes (enroot, apptainer, podman)
    add a real class and a branch here.
    """
    from cvs.core.errors import OrchestratorConfigError

    name = (runtime_cfg or {}).get("name", "hostshell").lower()
    config = (runtime_cfg or {}).get("config", {})

    if name == "hostshell":
        return HostShellRuntime.parse_config(config)
    if name == "docker":
        return DockerRuntime.parse_config(config)
    raise OrchestratorConfigError(f"runtime '{name}' is not implemented")
