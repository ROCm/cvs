"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

from typing import Optional

from cvs.core.transports.base import Transport


class HostShellRuntime:
    """Trivial Runtime: commands run directly in the host bash shell.

    Replaces what the friend's branch called BaremetalOrchestrator. The whole
    "container vs baremetal" class distinction collapses into "what Runtime is
    composed into the single Orchestrator class": HostShellRuntime here, or
    DockerRuntime for container mode.

    wrap_cmd is identity. setup and teardown are no-ops because the host shell
    is always already there. capabilities is empty by default; future host-only
    runtimes (e.g. running through a wrapper script) could extend it.
    """

    name = "hostshell"

    def __init__(
        self,
        workload_ssh_port: int = 22,
        workload_hostfile_path: str = "/tmp/mpi_hosts.txt",
        capabilities: Optional[set[str]] = None,
    ):
        self._workload_ssh_port = workload_ssh_port
        self._workload_hostfile_path = workload_hostfile_path
        self.capabilities: set[str] = set(capabilities) if capabilities else set()

    @classmethod
    def parse_config(cls, config: Optional[dict]) -> "HostShellRuntime":
        """Build a HostShellRuntime from its runtime.config dict.

        HostShellRuntime has no mandatory config; an empty / None config is fine
        and corresponds to legacy main-shape cluster.json with no runtime block.
        """
        config = config or {}
        return cls(
            workload_ssh_port=int(config.get("workload_ssh_port", 22)),
            workload_hostfile_path=config.get("workload_hostfile_path", "/tmp/mpi_hosts.txt"),
            capabilities=set(config.get("capabilities", [])),
        )

    def setup(self, transport: Transport) -> None:
        return None

    def teardown(self, transport: Transport) -> None:
        return None

    def wrap_cmd(self, cmd: str) -> str:
        return cmd

    def workload_ssh_port(self) -> int:
        return self._workload_ssh_port

    def workload_hostfile_path(self) -> str:
        return self._workload_hostfile_path
