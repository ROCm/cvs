"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from cvs.core.orchestrator import Orchestrator


class Severity(Enum):
    """Phase failure severity. HARD_FAIL aborts the pipeline and rolls back
    completed phases; SOFT_FAIL records the failure in the artifact and lets
    the pipeline keep going."""

    HARD_FAIL = "hard"
    SOFT_FAIL = "soft"


class PhaseError(Exception):
    """Raised by Phase.run when the phase failed in a way that should be
    surfaced to the pipeline runner. The runner classifies it via the phase's
    severity (re-raises on HARD_FAIL after rollback; records on SOFT_FAIL)."""


@runtime_checkable
class Phase(Protocol):
    """Discrete cluster-prep work that runs after the runtime is up but before
    workloads can run.

    A Phase is the right home for "do some commands on every host that aren't
    workload-specific": e.g. set up in-namespace sshd (this PR), check exclusivity,
    sanitize host CPU governor, run a noise-floor probe (follow-up PR).

    Phases are reusable across runtimes via applies_to(): the same MultinodeSshPhase
    runs against any Runtime whose .capabilities contains "in_namespace_sshd",
    so a future apptainer/podman runtime inherits it for free with no
    isinstance checks.
    """

    name: str
    severity: Severity

    def applies_to(self, orch: "Orchestrator") -> bool:
        """Cheap check: should this phase run for this orchestrator? Phases
        that don't apply are recorded as 'skipped' in the artifact and not
        executed."""
        ...

    def run(self, orch: "Orchestrator", artifact: dict) -> None:
        """Do the work. Stores any structured result in `artifact` (an empty
        dict the runner provides). Raises PhaseError on failure."""
        ...

    def undo(self, orch: "Orchestrator", artifact: dict) -> None:
        """Reverse the effect of run(). Called by the pipeline rollback path
        for phases that completed (status == 'ok'). May be a no-op."""
        ...
