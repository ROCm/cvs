"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

import enum
from typing import Any, List, Protocol, runtime_checkable


class Progress(enum.Enum):
    """Tri-state returned by ``progress_predicate`` and consumed by ``await_completion``.

    - ``RUNNING`` -> still healthy; keep polling (timeout => liveness_failure).
    - ``DONE``    -> workload finished cleanly; stop polling.
    - ``BROKEN``  -> predicate broke mid-run => safety_violation.
    """

    RUNNING = "running"
    DONE = "done"
    BROKEN = "broken"


@runtime_checkable
class WorkloadAdapter(Protocol):
    """The v1 adapter contract: the seven lifecycle methods every workload implements.

    This Protocol shape is v1's commitment. It is deliberately *not* claimed to
    be a closed contract for every conceivable future workload; it is the
    surface the ``Job`` driver depends on today.

    The ``ctx`` parameter is the ``RunContext`` introduced by G5b alongside the
    ``Job`` driver and staging seam; it is typed here as ``Any`` to keep G5a
    free of a forward import on a module that does not exist yet.
    """

    framework: str

    def prepare(self, ctx: Any) -> None: ...

    def launch(self, ctx: Any) -> None: ...

    def progress_predicate(self, ctx: Any) -> Progress: ...

    def await_completion(self, ctx: Any) -> None: ...

    def parse(self, ctx: Any) -> None: ...

    def verify(self, ctx: Any) -> List: ...

    def teardown(self, ctx: Any) -> None: ...
