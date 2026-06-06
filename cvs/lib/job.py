"""DTNI v1 Job — 7-phase driver, slim verdict.json output.

Phases: prepare, launch, await, parse, verify, teardown (always).
On failure: WorkloadError raised, phase tagged, verdict.json records
{failed_phase, message, verdicts:[]}.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cvs.lib.errors import WorkloadError
from cvs.lib.run_context import RunContext
from cvs.lib.verdict import all_passed, evaluate_all


@dataclass
class JobResult:
    passed: bool
    failed_phase: str | None
    message: str | None
    verdicts: list[dict]
    verdict_dict: dict[str, Any] = field(default_factory=dict)

    @property
    def failure(self) -> str | None:
        if self.passed:
            return None
        if self.failed_phase:
            return f"phase={self.failed_phase}: {self.message}"
        return f"thresholds failed: {[v for v in self.verdicts if not v['passed']]}"


class Job:
    def __init__(self, adapter, ctx: RunContext) -> None:
        self.adapter = adapter
        self.ctx = ctx

    def _phase(self, name: str, fn) -> None:
        try:
            fn()
        except WorkloadError as exc:
            if exc.phase is None:
                exc.phase = name
            raise
        except Exception as exc:
            raise WorkloadError(f"{name} failed: {exc}", phase=name) from exc

    def run(self) -> JobResult:
        ctx = self.ctx
        failed_phase: str | None = None
        message: str | None = None
        verdicts: list[dict] = []

        try:
            self._phase("prepare", lambda: self.adapter.prepare(ctx))
            self._phase("launch", lambda: self.adapter.launch(ctx))
            self._phase("await", lambda: self.adapter.await_completion(ctx))
            self._phase("parse", lambda: self.adapter.parse(ctx))
            # verify is inline (no adapter override in v1)
            verdicts = evaluate_all(ctx.thresholds, ctx.result.scalars)
            if not all_passed(verdicts):
                raise WorkloadError(
                    f"{sum(1 for v in verdicts if not v['passed'])}/{len(verdicts)} thresholds failed",
                    phase="verify",
                )
        except WorkloadError as exc:
            failed_phase = exc.phase
            message = exc.message
        finally:
            try:
                self.adapter.teardown(ctx)
            except Exception as exc:
                # teardown errors don't override a real failure
                if failed_phase is None:
                    failed_phase = "teardown"
                    message = f"teardown failed: {exc}"

        passed = failed_phase is None and all_passed(verdicts)
        verdict_dict = {
            "run_id": ctx.run_id,
            "workload": ctx.workload_name,
            "arch": ctx.arch,
            "framework": ctx.framework,
            "workload_hash": ctx.workload_hash,
            "failed_phase": failed_phase,
            "message": message,
            "verdicts": verdicts,
        }
        return JobResult(passed, failed_phase, message, verdicts, verdict_dict)
