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

RESULTS_COLUMNS = (
    ("Model", None),
    ("GPU", None),
    ("Workload", None),
    ("Resolution", None),
    ("Precision", None),
    ("Batch/GPU", None),
    ("Host", None),
    ("Images/s", "training.images_per_sec"),
    ("Images/s/GPU", "training.images_per_sec_per_gpu"),
    ("Mean step (ms)", "training.step_time_ms_mean"),
    ("P50 step (ms)", "training.step_time_ms_p50"),
    ("P95 step (ms)", "training.step_time_ms_p95"),
    ("Peak allocated (MB)", "training.peak_memory_allocated_mb"),
    ("Peak reserved (MB)", "training.peak_memory_reserved_mb"),
    ("Initial loss", "training.loss_initial"),
    ("Final loss", "training.loss_final"),
)

METRIC_TIERS = {
    "throughput": (
        "images_per_sec",
        "images_per_sec_per_gpu",
    ),
    "latency": (
        "step_time_ms_mean",
        "step_time_ms_p50",
        "step_time_ms_p95",
    ),
    "memory": (
        "peak_memory_allocated_mb",
        "peak_memory_reserved_mb",
    ),
}
METRIC_TIER_ORDER = tuple(METRIC_TIERS) + ("record",)
_TIERED_METRICS = {metric for names in METRIC_TIERS.values() for metric in names}
RECORD_METRICS = tuple(name for name, _unit in METRICS if name not in _TIERED_METRICS)


def tier_metric_specs(thresholds_cell: dict, tier: str) -> Dict[str, dict]:
    """Return full metric-name threshold specs for one run-deck tier."""
    names = RECORD_METRICS if tier == "record" else METRIC_TIERS.get(tier, ())
    specs = {}
    for name in names:
        full_name = f"training.{name}"
        spec = thresholds_cell.get(full_name)
        if spec is not None:
            specs[full_name] = spec
    return specs


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
