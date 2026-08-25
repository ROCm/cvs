'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Long-context accuracy cell selection (ACC-12 NIAH) for inference suites.
Threshold/gating values live in threshold.json under ``long_context_accuracy``.
'''

from __future__ import annotations

from typing import List

from pydantic import model_validator

from cvs.lib.utils.config_loader import _Forbid


class LongContextAccCell(_Forbid):
    id: str
    isl: int
    osl: int = 32
    num_prompts: int = 8
    seed: int = 42


class LongContextAccuracyConfig(_Forbid):
    cells: List[LongContextAccCell] = []

    @model_validator(mode="after")
    def _check_unique_cell_ids(self):
        from collections import Counter

        counts = Counter(c.id for c in self.cells)
        dupes = sorted(i for i, n in counts.items() if n > 1)
        if dupes:
            rendered = ", ".join(repr(d) for d in dupes)
            raise ValueError(f"duplicate long-context cell id(s): {rendered}")
        return self
