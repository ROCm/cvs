"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

import logging
from typing import Optional

from cvs.core.launchers.base import WorkloadLauncher
from cvs.core.runtimes.base import Runtime
from cvs.core.scope import ExecResult, ExecScope, ExecTarget
from cvs.core.transports.base import Transport


class Orchestrator:
    """The single Orchestrator class. Variation lives in the composed parts.

    Composed of:
      * transport - how to reach N hosts
      * runtime   - per-host execution context (host shell, docker exec, ...)
      * launchers - {name: WorkloadLauncher}; today {"mpi": MpiLauncher}

    There is no inheritance hierarchy. "Container vs baremetal" is just
    "what Runtime is composed in." The whole point of this class is that
    swapping Transport / Runtime / Launcher requires no Orchestrator subclass.

    Surface:
      * setup()   - delegate to runtime.setup; once the lifecycle Phase
                    abstraction lands in a follow-up commit, this also runs
                    PREPARE_PIPELINE.
      * cleanup() - mirror of setup.
      * exec()    - the only command-execution entry point. Callers pick the
                    scope (ALL/HEAD/SUBSET) and target (HOST = bare host shell,
                    RUNTIME = goes through runtime.wrap_cmd).

    Notably absent: orch.all / orch.head properties (use scope=ALL/HEAD with
    target=HOST), distribute_using_mpi (use orch.launchers["mpi"].launch).
    """

    def __init__(
        self,
        transport: Transport,
        runtime: Runtime,
        launchers: dict[str, WorkloadLauncher],
        log: Optional[logging.Logger] = None,
    ):
        self.transport = transport
        self.runtime = runtime
        self.launchers = launchers
        self.log = log or logging.getLogger("cvs")
        self.hosts = transport.hosts
        self.head_node = transport.head_node
        # Per-pipeline artifact storage (in-memory only this PR; persistence
        # to /tmp/cvs/<pipeline>/<host>.json is the host-prep PR's concern).
        self.artifacts: dict[str, dict[str, dict]] = {}

    def setup(self) -> None:
        # Lazy import so this module doesn't pull in lifecycle on import (which
        # in turn imports the Orchestrator type for its Phase Protocol hint).
        from cvs.core.lifecycle import PREPARE_PIPELINE

        self.runtime.setup(self.transport)
        self.artifacts["prepare"] = PREPARE_PIPELINE.run(self)

    def cleanup(self) -> None:
        from cvs.core.lifecycle import PREPARE_PIPELINE

        try:
            PREPARE_PIPELINE.rollback(self, self.artifacts.get("prepare", {}))
        finally:
            self.runtime.teardown(self.transport)

    def exec(
        self,
        cmd: str,
        *,
        scope: ExecScope = ExecScope.ALL,
        target: ExecTarget = ExecTarget.RUNTIME,
        subset: Optional[list[str]] = None,
        timeout: Optional[int] = None,
    ) -> dict[str, ExecResult]:
        body = cmd if target is ExecTarget.HOST else self.runtime.wrap_cmd(cmd)
        return self.transport.exec(body, scope, subset=subset, timeout=timeout)
