"""Metric contract for PyTorch Vision training artifacts."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple


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
OPTIONAL_ARTIFACT_METRICS: Tuple[Tuple[str, str], ...] = (
    ("data_loader_images_per_sec", "images/s"),
    ("top1_accuracy_pct", "%"),
    ("top5_accuracy_pct", "%"),
    ("eval_loss", "-"),
    ("eval_sample_count", "images"),
    ("eval_completed", "bool"),
    ("loss_curve_points", "count"),
    ("loss_curve_decreased", "bool"),
    ("convergence_step", "step"),
    ("convergence_time_seconds", "s"),
    ("continuous_peak_device_memory_mb", "MB"),
    ("gpu_compute_util_pct", "%"),
    ("gpu_bandwidth_util_pct", "%"),
    ("gpu_energy_delta_j", "J"),
    ("energy_tracking_available", "bool"),
    ("energy_kwh", "kWh"),
    ("average_power_w", "W"),
    ("images_per_kwh", "images/kWh"),
    ("emissions_kg_co2eq", "kgCO2eq"),
    ("codecarbon_tracking_active", "bool"),
)
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
    ("Top-1 (%)", "training.top1_accuracy_pct"),
    ("Top-5 (%)", "training.top5_accuracy_pct"),
    ("Eval loss", "training.eval_loss"),
    ("Eval samples", "training.eval_sample_count"),
    ("Convergence step", "training.convergence_step"),
    ("Convergence time (s)", "training.convergence_time_seconds"),
    ("Continuous peak memory (MB)", "training.continuous_peak_device_memory_mb"),
    ("Compute util (%)", "training.gpu_compute_util_pct"),
    ("Bandwidth util (%)", "training.gpu_bandwidth_util_pct"),
    ("Energy tracking", "training.energy_tracking_available"),
    ("Energy (kWh)", "training.energy_kwh"),
    ("Images/kWh", "training.images_per_kwh"),
    ("Emissions (kgCO2eq)", "training.emissions_kg_co2eq"),
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
    "accuracy": (
        "top1_accuracy_pct",
        "top5_accuracy_pct",
        "eval_loss",
        "eval_sample_count",
        "eval_completed",
    ),
    "convergence": (
        "loss_curve_points",
        "loss_curve_decreased",
        "convergence_step",
        "convergence_time_seconds",
    ),
    "continuous-memory": (
        "continuous_peak_device_memory_mb",
        "gpu_compute_util_pct",
        "gpu_bandwidth_util_pct",
    ),
    "energy": (
        "gpu_energy_delta_j",
        "energy_tracking_available",
        "energy_kwh",
        "average_power_w",
        "images_per_kwh",
        "emissions_kg_co2eq",
        "codecarbon_tracking_active",
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


def gradient_accumulation_overhead_pct(baseline_step_ms: float, accumulated_step_ms: float) -> float:
    """Return GA optimizer-step overhead while effective global batch is fixed."""
    baseline = float(baseline_step_ms)
    accumulated = float(accumulated_step_ms)
    if not math.isfinite(baseline) or baseline <= 0:
        raise ValueError(f"baseline step time must be finite and positive, got {baseline_step_ms!r}")
    if not math.isfinite(accumulated) or accumulated <= 0:
        raise ValueError(f"accumulated step time must be finite and positive, got {accumulated_step_ms!r}")
    return (accumulated / baseline - 1.0) * 100.0


def accuracy_from_counts(top1_correct: int, top5_correct: int, sample_count: int) -> Tuple[float, float]:
    """Convert globally summed correct/sample counts into percentages."""
    if sample_count <= 0:
        raise ValueError("distributed accuracy requires a positive sample count")
    if not 0 <= top1_correct <= top5_correct <= sample_count:
        raise ValueError("distributed accuracy counts must satisfy 0 <= top1 <= top5 <= samples")
    return 100.0 * top1_correct / sample_count, 100.0 * top5_correct / sample_count


def loss_curve_decreased(losses: Sequence[float], minimum_points: int = 2) -> bool:
    """Return whether the least-squares loss slope is finite and negative."""
    if len(losses) < minimum_points:
        return False
    values = [float(value) for value in losses]
    if not all(math.isfinite(value) for value in values):
        return False
    x_mean = (len(values) - 1) / 2.0
    y_mean = sum(values) / len(values)
    numerator = sum((index - x_mean) * (value - y_mean) for index, value in enumerate(values))
    denominator = sum((index - x_mean) ** 2 for index in range(len(values)))
    return denominator > 0 and numerator / denominator < 0


def convergence_point(
    evaluations: Iterable[Dict[str, float]],
    target_top1_pct: Optional[float] = None,
    target_eval_loss: Optional[float] = None,
) -> Optional[Tuple[int, float]]:
    """Return the first (step, elapsed seconds) satisfying every configured target."""
    for item in evaluations:
        top1_ok = target_top1_pct is None or item.get("top1_accuracy_pct", -math.inf) >= target_top1_pct
        loss_ok = target_eval_loss is None or item.get("eval_loss", math.inf) <= target_eval_loss
        if top1_ok and loss_ok:
            return int(item["step"]), float(item["time_seconds"])
    return None


def codecarbon_tracking_active(payload: Dict[str, Any], expected_gpus: int) -> bool:
    """Validate explicit CodeCarbon/AMDSMI activation evidence."""
    return (
        payload.get("version") == "3.2.4"
        and payload.get("tracker") == "amdsmi"
        and int(payload.get("gpu_count") or 0) >= expected_gpus
        and payload.get("active") is True
    )


def parse_codecarbon_metrics(payload: Dict[str, Any], expected_gpus: int) -> Dict[str, float]:
    """Parse CodeCarbon output without treating unavailable tracking as zero usage."""
    if not codecarbon_tracking_active(payload, expected_gpus):
        reason = payload.get("reason") or "AMDSMI tracking is not active"
        raise ValueError(f"CodeCarbon metrics unavailable: {reason}")
    emissions = float(payload["emissions_kg_co2eq"])
    if not math.isfinite(emissions) or emissions < 0:
        raise ValueError(f"invalid CodeCarbon emissions value: {emissions!r}")
    metrics = {
        "training.codecarbon_tracking_active": 1.0,
        "training.emissions_kg_co2eq": emissions,
    }
    energy_kwh = payload.get("energy_kwh")
    if energy_kwh is not None:
        energy_kwh = float(energy_kwh)
        if not math.isfinite(energy_kwh) or energy_kwh <= 0:
            raise ValueError(f"invalid CodeCarbon energy value: {energy_kwh!r}")
        metrics["training.energy_kwh"] = energy_kwh
    return metrics


def required_metrics_for_run(run_mode: str, data_mode: str, codecarbon_enabled: bool = False) -> set[str]:
    """Return metrics whose absence is a structural failure for this run."""
    required = set(GATED_METRICS)
    required.update(
        {
            "continuous_peak_device_memory_mb",
            "gpu_compute_util_pct",
            "gpu_bandwidth_util_pct",
            "energy_tracking_available",
        }
    )
    if data_mode == "rocal":
        required.add("data_loader_images_per_sec")
    if run_mode != "perf":
        required.update(
            {
                "top1_accuracy_pct",
                "top5_accuracy_pct",
                "eval_loss",
                "eval_sample_count",
                "eval_completed",
            }
        )
    if run_mode not in {"smoke", "perf"}:
        required.update(
            {
                "loss_curve_points",
                "loss_curve_decreased",
            }
        )
    if codecarbon_enabled:
        required.add("codecarbon_tracking_active")
    return required


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
    if "eval_completed" in metrics and out["training.eval_completed"] != 1.0:
        raise ValueError("vision evaluation did not complete")
    if "eval_sample_count" in metrics and out["training.eval_sample_count"] <= 0:
        raise ValueError("vision evaluation sample count must be positive")
    if "codecarbon_tracking_active" in metrics and out["training.codecarbon_tracking_active"] != 1.0:
        raise ValueError("vision artifact claims inactive CodeCarbon tracking")
    return out
