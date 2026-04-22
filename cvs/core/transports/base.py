"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from cvs.core.scope import ExecResult, ExecScope


@runtime_checkable
class Transport(Protocol):
    """How commands reach N hosts.

    The transport hides the details of "fan out a shell command across the cluster"
    behind one method. Today's only implementation is PsshTransport (parallel SSH);
    future transports could be local subprocess, slurm srun, or kubernetes exec.

    A Transport never wraps the command. Wrapping (e.g. docker exec ...) is the
    Runtime's responsibility. Transports are pure dispatchers.
    """

    hosts: list[str]
    head_node: str
    env_prefix: str

    def exec(
        self,
        cmd: str,
        scope: ExecScope,
        *,
        subset: Optional[list[str]] = None,
        timeout: Optional[int] = None,
    ) -> dict[str, ExecResult]: ...

    def scp(
        self,
        src: str,
        dst: str,
        scope: ExecScope,
        *,
        subset: Optional[list[str]] = None,
    ) -> None: ...
