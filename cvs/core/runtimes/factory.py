'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

from __future__ import annotations

from typing import TYPE_CHECKING

from .docker import DockerRuntime
from .enroot import EnrootRuntime
from .hostshell import HostShellRuntime

if TYPE_CHECKING:
    from .base import Runtime


class RuntimeFactory:
    """Legacy factory kept for the existing ContainerOrchestrator path.

    The new dispatch is build_runtime(cfg) below; this class is removed when
    the legacy orchestrators get deleted.
    """

    @staticmethod
    def create(runtime_name, log, orchestrator):
        """Create a container runtime instance."""
        runtime_name = runtime_name.lower()

        if runtime_name == 'docker':
            return DockerRuntime(log, orchestrator)
        elif runtime_name == 'enroot':
            return EnrootRuntime(log, orchestrator)
        else:
            raise ValueError(f"Unsupported container runtime: {runtime_name}")


# -----------------------------------------------------------------------------
# New Runtime factory used by the single Orchestrator class.
# -----------------------------------------------------------------------------
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
