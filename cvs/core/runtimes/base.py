'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

from __future__ import annotations

from typing import Protocol, runtime_checkable

from cvs.core.transports.base import Transport


@runtime_checkable
class Runtime(Protocol):
    """How a command executes on ONE host: per-host execution context.

    Variation across runtimes:
      * HostShellRuntime  - identity wrap; commands run in the host bash shell
      * DockerRuntime     - wrap_cmd produces `docker exec ... bash -lc <cmd>`
      * future ApptainerRuntime / EnrootRuntime - their own wrap shapes

    Lifecycle (per-host bring-up/tear-down) is the runtime's responsibility:
    HostShell does nothing; Docker runs `docker run -d ... sleep infinity`.

    workload_ssh_port and workload_hostfile_path are the bits a WorkloadLauncher
    needs to know -- e.g. MpiLauncher uses them to build the right mpirun string
    without ever knowing what runtime it is launching into.

    capabilities is a free-form set of opt-in tags that lifecycle Phases
    consult via Phase.applies_to(). E.g. MultinodeSshPhase opts in via
    "in_namespace_sshd" so a future runtime that runs sshd inside its
    container inherits the phase for free.
    """

    name: str
    capabilities: set[str]

    def setup(self, transport: Transport) -> None: ...
    def teardown(self, transport: Transport) -> None: ...
    def wrap_cmd(self, cmd: str) -> str: ...
    def workload_ssh_port(self) -> int: ...
    def workload_hostfile_path(self) -> str: ...
