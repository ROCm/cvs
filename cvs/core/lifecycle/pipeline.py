"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Phase, PhaseError, Severity

if TYPE_CHECKING:
    from cvs.core.orchestrator import Orchestrator


class Pipeline:
    """Ordered list of Phases run together with shared artifact storage.

    run(orch) walks phases in order:
      * applies_to == False -> record {"status": "skipped"}, continue
      * run() returns       -> record {"status": "ok"} (plus whatever the
                               phase wrote into its artifact dict)
      * run() raises PhaseError + Severity.SOFT_FAIL -> record
                               {"status": "failed", "error": str(e)},
                               continue
      * run() raises PhaseError + Severity.HARD_FAIL -> record the failure,
                               call rollback() to undo every previously-ok
                               phase in reverse order, then re-raise

    rollback(orch, artifacts) walks phases in REVERSE order and calls
    phase.undo() for any phase whose artifact says status == 'ok'. Errors
    in undo() are logged but do not stop the rollback.

    Artifacts are an in-memory dict on the orchestrator instance; persistence
    to /tmp/cvs/<pipeline>/<host>.json is the host-prep follow-up's concern.
    """

    def __init__(self, name: str, phases: list[Phase]):
        self.name = name
        self.phases = list(phases)

    def run(self, orch: "Orchestrator") -> dict[str, dict]:
        artifacts: dict[str, dict] = {}
        for phase in self.phases:
            if not phase.applies_to(orch):
                artifacts[phase.name] = {"status": "skipped"}
                continue
            artifact: dict = {}
            artifacts[phase.name] = artifact
            try:
                phase.run(orch, artifact)
                artifact["status"] = "ok"
            except PhaseError as e:
                artifact["status"] = "failed"
                artifact["error"] = str(e)
                if phase.severity is Severity.HARD_FAIL:
                    orch.log.error(
                        f"pipeline '{self.name}' aborted: phase '{phase.name}' failed: {e}"
                    )
                    self.rollback(orch, artifacts)
                    raise
                orch.log.warning(
                    f"pipeline '{self.name}' phase '{phase.name}' soft-failed: {e}"
                )
        return artifacts

    def rollback(self, orch: "Orchestrator", artifacts: dict[str, dict]) -> None:
        for phase in reversed(self.phases):
            entry = artifacts.get(phase.name) or {}
            if entry.get("status") != "ok":
                continue
            try:
                phase.undo(orch, entry)
            except Exception as e:  # noqa: BLE001 - rollback never raises
                orch.log.warning(
                    f"pipeline '{self.name}' undo of phase '{phase.name}' failed: {e}"
                )
