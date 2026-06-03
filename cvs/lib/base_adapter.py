"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

import abc
import time
from typing import Any, List

from cvs.lib.adapter_protocol import Progress
from cvs.lib.config.thresholds import ThresholdVerdict
from cvs.lib.failure_taxonomy import LivenessFailure, SafetyViolation, VerificationFailure


class BaseWorkloadAdapter(abc.ABC):
    """Concrete defaults most adapters inherit.

    Subclasses must implement the workload-specific steps (``launch``,
    ``progress_predicate``, ``parse``). The polling ``await_completion``, the
    threshold-driven ``verify``, the no-op ``prepare``, and the
    forensics-capturing ``teardown`` are provided here so an adapter does not
    re-implement them (the copy-paste this whole refactor removes).

    The ``ctx`` parameter is the ``RunContext`` introduced by G5b; it is typed
    as ``Any`` here because G5a deliberately predates the executor/staging seam.
    """

    framework: str = "base"

    # Tunable by subclasses or via config; await_completion honors these.
    poll_interval_s: float = 5.0
    completion_timeout_s: float = 3600.0

    def prepare(self, ctx: Any) -> None:  # noqa: ARG002 - default no-op
        return None

    @abc.abstractmethod
    def launch(self, ctx: Any) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def progress_predicate(self, ctx: Any) -> Progress:
        raise NotImplementedError

    @abc.abstractmethod
    def parse(self, ctx: Any) -> None:
        raise NotImplementedError

    def await_completion(self, ctx: Any) -> None:
        """Poll ``progress_predicate`` until DONE; classify BROKEN/timeout."""
        deadline = time.monotonic() + self.completion_timeout_s
        while True:
            state = self.progress_predicate(ctx)
            if state is Progress.DONE:
                return
            if state is Progress.BROKEN:
                ctx.events.emit("safety.violated", run_id=ctx.run_id)
                raise SafetyViolation("progress predicate broke mid-run")
            if time.monotonic() >= deadline:
                raise LivenessFailure(f"await_completion timed out after {self.completion_timeout_s}s")
            time.sleep(self.poll_interval_s)

    def verify(self, ctx: Any) -> List[ThresholdVerdict]:
        """Evaluate every configured threshold against ``ctx.result``.

        Verdicts are recorded on the context regardless of outcome (so the
        manifest captures passing thresholds too); a single failing threshold
        raises ``VerificationFailure`` -- classification happens at this raise
        site, never by post-hoc inspection.
        """
        verdicts = [threshold.evaluate(ctx.result) for threshold in ctx.config.thresholds]
        ctx.scratch["threshold_verdicts"] = verdicts
        failed = [v for v in verdicts if not v.passed]
        if failed:
            ctx.events.emit("verify.failed", failed=len(failed), total=len(verdicts))
            raise VerificationFailure(
                f"{len(failed)}/{len(verdicts)} thresholds failed",
                detail={"verdicts": [v.model_dump() for v in verdicts]},
            )
        ctx.events.emit("verify.passed", total=len(verdicts))
        return verdicts

    def teardown(self, ctx: Any) -> None:
        """Capture forensics from every registered container, then remove them.

        Always safe to call (best-effort): runs in the ``Job``'s ``finally`` so
        a crash never leaks containers. Captured artifacts are written under the
        run's ``logs/`` directory.
        """
        logs_dir = ctx.layout.logs_dir
        logs_dir.mkdir(parents=True, exist_ok=True)
        for handle in ctx.containers:
            try:
                artifacts = handle.capture()
                for name, text in artifacts.items():
                    (logs_dir / f"{handle.name}.{name}").write_text(text)
                    ctx.logs[f"{handle.name}.{name}"] = text
            except Exception:  # noqa: BLE001 - teardown is best-effort
                pass
            finally:
                handle.remove()
