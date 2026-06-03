"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# CLOSED event vocabulary. Adding a new event name is a deliberate schema change
# reviewed in a PR -- NOT an ad-hoc ``log.info`` call. EventWriter.emit raises on
# any name outside this set, which is what keeps events.jsonl analyzable across
# runs (a fixed vocabulary instead of free-text log lines).
EVENT_VOCAB = frozenset(
    {
        "prepare.start",
        "prepare.done",
        "launch.container_up",
        "launch.role_ready",
        "seed.logged",
        "arrival.start",
        "arrival.end",
        "accuracy.start",
        "accuracy.end",
        "step",
        "request",
        "safety.violated",
        "pattern.matched",
        "parse.done",
        "verify.passed",
        "verify.failed",
        "teardown.start",
        "teardown.done",
    }
)


class UnknownEventError(ValueError):
    """Raised when emitting an event name outside the closed vocabulary."""


class EventWriter:
    """Append-only writer for ``events.jsonl`` enforcing the closed vocabulary."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path is not None else None
        self.records: List[dict] = []
        self._fh = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")

    def emit(self, name: str, **fields) -> dict:
        if name not in EVENT_VOCAB:
            raise UnknownEventError(
                f"event {name!r} is not in the closed vocabulary; add it to EVENT_VOCAB in a reviewed change"
            )
        # Fail loud rather than silently dropping a durable write: a file-backed
        # writer used after close() would otherwise look like it persisted the event.
        if self.path is not None and self._fh is None:
            raise ValueError("EventWriter.emit called after close(); the events.jsonl handle is closed")
        record = {"ts": datetime.now(timezone.utc).isoformat(), "event": name}
        record.update(fields)
        if self._fh is not None:
            self._fh.write(json.dumps(record, default=str) + "\n")
            self._fh.flush()
        else:
            # Only buffer in memory when there is no file to stream to. High-frequency
            # events ("step", "request") would otherwise grow self.records without bound.
            self.records.append(record)
        return record

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "EventWriter":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False
