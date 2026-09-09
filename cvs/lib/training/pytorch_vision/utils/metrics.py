"""Metric contract for PyTorch Vision training artifacts."""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple


ARTIFACT_METRICS: Tuple[Tuple[str, str], ...] = (
    ("images_per_sec", "images/s"),
    ("images_per_sec_per_gpu", "images/s/GPU"),
    ("tflops_per_sec_per_gpu", "TFLOPS/s/GPU"),
    ("mfu_pct", "%"),
    ("step_time_ms_mean", "ms"),
    ("step_time_ms_p50", "ms"),
    ("step_time_ms_p95", "ms"),
    ("peak_memory_allocated_mb", "MB"),
    ("peak_memory_reserved_mb", "MB"),
    ("device_memory_used_mb_observed", "MB"),
    ("checkpoint_save_seconds", "s"),
    ("checkpoint_load_seconds", "s"),
    ("checkpoint_state_match", "bool"),
    ("checkpoint_loss_delta", "-"),
    ("checkpoint_resume_model_max_abs_delta", "-"),
    ("checkpoint_resume_optimizer_max_abs_delta", "-"),
    ("loss_initial", "-"),
    ("loss_final", "-"),
)
OPTIONAL_ARTIFACT_METRICS: Tuple[Tuple[str, str], ...] = (("data_loader_images_per_sec", "images/s"),)
DERIVED_METRICS: Tuple[Tuple[str, str], ...] = (("gradient_accumulation_overhead_pct", "%"),)
METRICS = ARTIFACT_METRICS + OPTIONAL_ARTIFACT_METRICS + DERIVED_METRICS

METRIC_UNITS = dict(METRICS)
GATED_METRICS = {
    "images_per_sec",
    "images_per_sec_per_gpu",
    "step_time_ms_p95",
    "peak_memory_allocated_mb",
    "peak_memory_reserved_mb",
    "device_memory_used_mb_observed",
    "checkpoint_state_match",
    "checkpoint_loss_delta",
    "checkpoint_resume_model_max_abs_delta",
    "checkpoint_resume_optimizer_max_abs_delta",
    "gradient_accumulation_overhead_pct",
}

RESULTS_COLUMNS = (
    ("Model", None),
    ("GPU", None),
    ("Workload", None),
    ("Resolution", None),
    ("GA", None),
    ("MBS/GPU", None),
    ("Host", None),
    ("Images/s", "training.images_per_sec"),
    ("Images/s/GPU", "training.images_per_sec_per_gpu"),
    ("TFLOPS/s/GPU", "training.tflops_per_sec_per_gpu"),
    ("MFU (%)", "training.mfu_pct"),
    ("Mean step (ms)", "training.step_time_ms_mean"),
    ("P50 step (ms)", "training.step_time_ms_p50"),
    ("P95 step (ms)", "training.step_time_ms_p95"),
    ("Peak allocated (MB)", "training.peak_memory_allocated_mb"),
    ("Peak reserved (MB)", "training.peak_memory_reserved_mb"),
    ("Observed device used (MB)", "training.device_memory_used_mb_observed"),
    ("Checkpoint save (s)", "training.checkpoint_save_seconds"),
    ("Checkpoint load (s)", "training.checkpoint_load_seconds"),
    ("Checkpoint state match", "training.checkpoint_state_match"),
    ("Checkpoint loss delta", "training.checkpoint_loss_delta"),
    ("Resume model max abs delta", "training.checkpoint_resume_model_max_abs_delta"),
    ("Resume optimizer max abs delta", "training.checkpoint_resume_optimizer_max_abs_delta"),
    ("GA overhead (%)", "training.gradient_accumulation_overhead_pct"),
    ("Initial loss", "training.loss_initial"),
    ("Final loss", "training.loss_final"),
    ("Data loader images/s", "training.data_loader_images_per_sec"),
)

METRIC_TIERS = {
    "throughput": (
        "images_per_sec",
        "images_per_sec_per_gpu",
        "tflops_per_sec_per_gpu",
        "mfu_pct",
    ),
    "latency": (
        "step_time_ms_mean",
        "step_time_ms_p50",
        "step_time_ms_p95",
    ),
    "memory": (
        "peak_memory_allocated_mb",
        "peak_memory_reserved_mb",
        "device_memory_used_mb_observed",
    ),
    "checkpoint": (
        "checkpoint_save_seconds",
        "checkpoint_load_seconds",
        "checkpoint_state_match",
        "checkpoint_loss_delta",
        "checkpoint_resume_model_max_abs_delta",
        "checkpoint_resume_optimizer_max_abs_delta",
    ),
    "overhead": ("gradient_accumulation_overhead_pct",),
    "data": ("data_loader_images_per_sec",),
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


def gradient_accumulation_overhead_pct(baseline_step_ms: float, accumulated_step_ms: float) -> float:
    """Return GA optimizer-step overhead while effective global batch is fixed."""
    baseline = float(baseline_step_ms)
    accumulated = float(accumulated_step_ms)
    if not math.isfinite(baseline) or baseline <= 0:
        raise ValueError(f"baseline step time must be finite and positive, got {baseline_step_ms!r}")
    if not math.isfinite(accumulated) or accumulated <= 0:
        raise ValueError(f"accumulated step time must be finite and positive, got {accumulated_step_ms!r}")
    return (accumulated / baseline - 1.0) * 100.0


def to_training_metrics(raw: Dict[str, Any]) -> Dict[str, float]:
    """Validate a rank-zero artifact and return namespaced numeric metrics."""
    metrics = raw.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("vision result artifact is missing a metrics object")

    out = {}
    missing = []
    for name, _unit in ARTIFACT_METRICS:
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
    for name, _unit in OPTIONAL_ARTIFACT_METRICS:
        value = metrics.get(name)
        if value is None:
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"vision metric {name!r} is not finite: {value!r}")
        out[f"training.{name}"] = numeric
    return out
