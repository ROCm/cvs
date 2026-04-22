"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

from cvs.core.scope import ExecResult

if TYPE_CHECKING:
    from cvs.core.orchestrator import Orchestrator


@runtime_checkable
class WorkloadLauncher(Protocol):
    """How a logical workload expands into N processes across the cluster.

    MPI is one launcher (head node fans out via mpirun + SSH). Future launchers
    could be torchrun (per-node self-launch), srun (slurm), Ray, etc.

    A launcher consumes the Orchestrator's exec API and asks the Runtime for
    its workload-launch facts (ssh port, hostfile path) so the launcher does
    not need to know what runtime it is launching into.
    """

    name: str

    def launch(
        self,
        orch: "Orchestrator",
        cmd: str,
        hosts: list[str],
        env: dict[str, str],
        ranks_per_host: int,
        extra_args: Optional[list[str]] = None,
    ) -> dict[str, ExecResult]: ...
