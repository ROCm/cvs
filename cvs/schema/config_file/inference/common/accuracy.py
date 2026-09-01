"""Accuracy-evaluation config schema shared across inference suites."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from pydantic import model_validator

from cvs.schema.base import _Forbid


class AccuracyTask(_Forbid):
    id: str
    task: str
    num_fewshot: int = 0
    metadata: Dict[str, Any] = {}
    include_path: str = ""
    num_concurrent: int = 8
    apply_chat_template: bool = False
    gen_kwargs: Dict[str, Any] = {}


class AccuracyConfig(_Forbid):
    tasks: List[AccuracyTask] = []

    @model_validator(mode="after")
    def _check_unique_task_ids(self):
        counts = Counter(t.id for t in self.tasks)
        dupes = sorted(i for i, n in counts.items() if n > 1)
        if dupes:
            rendered = ", ".join(repr(d) for d in dupes)
            raise ValueError(f"duplicate task id(s): {rendered}")
        return self
