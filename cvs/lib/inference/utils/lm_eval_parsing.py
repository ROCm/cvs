'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Pure JSON -> {scalar: float} projector for lm-eval-harness `results.json`
payloads (see cvs/lib/inference/utils/AGENTS.md for the broader
accuracy-evaluation design). Auto-discovers every numeric metric rather than
requiring a per-task registry, so group tasks (e.g. RULER's per-seq-length
metrics) and custom tasks fall out of the same walk with no special-casing.
'''

from __future__ import annotations

import math
from typing import Any, Dict


def _is_real_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return not math.isnan(value)


def project(payload: Dict[str, Any]) -> Dict[str, float]:
    out = {}
    for lm_task_name, metrics in payload.get("results", {}).items():
        for metric_key, value in metrics.items():
            if metric_key == "alias" or not _is_real_number(value):
                continue
            key = f"{lm_task_name}.{metric_key}".replace(",", "__")
            out[key] = float(value)
    return out
