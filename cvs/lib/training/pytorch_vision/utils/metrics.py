"""Metric contract for PyTorch Vision training artifacts."""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple


METRICS: Tuple[Tuple[str, str], ...] = (
    ("images_per_sec", "images/s"),
    ("images_per_sec_per_gpu", "images/s/GPU"),
    ("step_time_ms_mean", "ms"),
    ("step_time_ms_p50", "ms"),
    ("step_time_ms_p95", "ms"),
    ("peak_memory_allocated_mb", "MB"),
    ("peak_memory_reserved_mb", "MB"),
    ("loss_initial", "-"),
    ("loss_final", "-"),
)

METRIC_UNITS = dict(METRICS)
GATED_METRICS = {
    "images_per_sec",
    "images_per_sec_per_gpu",
    "step_time_ms_p95",
    "peak_memory_allocated_mb",
    "peak_memory_reserved_mb",
}


def to_training_metrics(raw: Dict[str, Any]) -> Dict[str, float]:
    """Validate a rank-zero artifact and return namespaced numeric metrics."""
    metrics = raw.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("vision result artifact is missing a metrics object")

    out = {}
    missing = []
    for name, _unit in METRICS:
        value = metrics.get(name)
        if value is None:
            missing.append(name)
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"vision metric {name!r} is not numeric: {value!r}") from exc
        if not math.isfinite(numeric):
            raise ValueError(f"vision metric {name!r} is not finite: {value!r}")
        out[f"training.{name}"] = numeric

    if missing:
        raise ValueError(f"vision result artifact is missing metrics: {missing}")
    return out
