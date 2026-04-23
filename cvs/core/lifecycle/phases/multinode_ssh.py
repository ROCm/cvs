"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from cvs.core.lifecycle.base import PhaseError, Severity
from cvs.core.scope import ExecScope, ExecTarget

if TYPE_CHECKING:
    from cvs.core.orchestrator import Orchestrator


# Six commands ported verbatim from the friend's branch
# (cvs/core/orchestrators/container.py::ContainerOrchestrator.setup_sshd):
# bring up an in-namespace sshd on port 2224 in every container so MPI can
# SSH peer-to-peer between containers.
SETUP_CMDS: list[str] = [
    "mkdir -p /root/.ssh",
    "bash -c 'cp -r /host_ssh/* /root/.ssh/'",  # bash -c so the glob expands inside the runtime
    "chown -R root:root /root/.ssh",
    "bash -c 'chmod 700 /root/.ssh && chmod 600 /root/.ssh/*'",
    "mkdir -p /run/sshd",
    "/usr/sbin/sshd -p2224",
]

# Validate the daemon is up after the start command before declaring success.
SSHD_CHECK_CMD = "pgrep -f 'sshd.*2224' > /dev/null 2>&1"

# Stop the in-namespace sshd processes on rollback.
SSHD_STOP_CMD = "pkill -f 'sshd.*2224' || true"

# Brief sleep to let sshd come up before we check; matches friend's branch.
SSHD_STARTUP_SLEEP_S = 2


class MultinodeSshPhase:
    """Bring up an in-namespace sshd on every host's runtime so MPI can ssh
    peer-to-peer between runtimes.

    applies_to(): runtime advertises 'in_namespace_sshd' in its capabilities.
    DockerRuntime opts in. HostShellRuntime does not. A future runtime that
    runs sshd inside its containers (apptainer, podman) inherits this phase
    by adding the same capability tag, no isinstance checks needed.
    """

    name = "multinode_ssh"
    severity = Severity.HARD_FAIL

    def applies_to(self, orch: "Orchestrator") -> bool:
        return "in_namespace_sshd" in orch.runtime.capabilities

    def run(self, orch: "Orchestrator", artifact: dict) -> None:
        for cmd in SETUP_CMDS:
            results = orch.exec(
                cmd, scope=ExecScope.ALL, target=ExecTarget.RUNTIME, timeout=10
            )
            failed = {h: r for h, r in results.items() if r.exit_code != 0}
            if failed:
                raise PhaseError(
                    f"setup command {cmd!r} failed on hosts: "
                    + ", ".join(
                        f"{h} (exit={r.exit_code}): {r.output.strip()[:120]}"
                        for h, r in failed.items()
                    )
                )

        # Give sshd a moment to start before we probe.
        time.sleep(SSHD_STARTUP_SLEEP_S)

        # Validate the daemon is actually accepting on the expected port.
        check = orch.exec(
            SSHD_CHECK_CMD, scope=ExecScope.ALL, target=ExecTarget.RUNTIME, timeout=10
        )
        not_running = [h for h, r in check.items() if r.exit_code != 0]
        if not_running:
            raise PhaseError(f"sshd:2224 not running on hosts: {not_running}")

        artifact["port"] = 2224
        artifact["hosts"] = list(orch.hosts)

    def undo(self, orch: "Orchestrator", artifact: dict) -> None:
        orch.exec(
            SSHD_STOP_CMD, scope=ExecScope.ALL, target=ExecTarget.RUNTIME, timeout=10
        )
