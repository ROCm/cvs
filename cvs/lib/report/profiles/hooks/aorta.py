'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Aorta Run Deck data and run-card helpers.
'''

from cvs.lib.report.rundeck.config_builder import provenance_link_rows


def _milliseconds(value):
    if value is None:
        return None
    return round(float(value) / 1000.0, 3)


def _percent(value):
    if value is None:
        return None
    return round(float(value) * 100.0, 2)


def _display(value, suffix="", digits=2):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


def build_aorta_series(result, parser_name):
    """Convert an aggregated Aorta result into per-rank trace series."""
    ranks = {}
    for metric in sorted(result.per_rank_metrics, key=lambda item: item.rank):
        ranks[str(metric.rank)] = {
            "rank": metric.rank,
            "node": metric.node or "—",
            "local_rank": metric.local_rank if metric.local_rank is not None else "—",
            "parser": parser_name,
            "total_time_ms": _milliseconds(metric.total_time_us),
            "compute_time_ms": _milliseconds(metric.compute_time_us),
            "communication_time_ms": _milliseconds(metric.communication_time_us),
            "memory_time_ms": _milliseconds(metric.memory_time_us),
            "idle_time_ms": _milliseconds(metric.idle_time_us),
            "compute_ratio_pct": _percent(metric.compute_ratio),
            "communication_ratio_pct": _percent(metric.comm_ratio),
            "overlap_ratio_pct": _percent(metric.compute_comm_overlap),
            "peak_memory_gb": metric.peak_memory_gb,
            "allocated_memory_gb": metric.allocated_memory_gb,
            "compute_kernel_count": metric.compute_kernel_count,
            "communication_kernel_count": metric.comm_kernel_count,
        }
    return {f"{parser_name} per-rank trace": ranks} if ranks else {}


def update_aorta_run_summary(variant, result, parser_name, run_result=None):
    """Update mutable run-card state after parser aggregation."""
    variant.update(
        {
            "parser": parser_name,
            "parsed_ranks": len(result.per_rank_metrics),
            "avg_iteration_time_ms": result.avg_iteration_time_ms,
            "std_iteration_time_ms": _milliseconds(result.std_iteration_time_us),
            "min_iteration_time_ms": _milliseconds(result.min_iteration_time_us),
            "max_iteration_time_ms": _milliseconds(result.max_iteration_time_us),
            "avg_compute_ratio_pct": _percent(result.avg_compute_ratio),
            "avg_communication_ratio_pct": _percent(result.avg_comm_ratio),
            "avg_overlap_ratio_pct": _percent(result.avg_overlap_ratio),
            "samples_per_second": result.samples_per_second,
            "tokens_per_second": result.tokens_per_second,
        }
    )
    if run_result is not None:
        variant.update(
            {
                "status": run_result.status.value,
                "duration_seconds": run_result.duration_seconds,
                "launch_mode": run_result.metadata.get("launch_mode", "—"),
                "artifacts": sorted(run_result.artifacts),
            }
        )


def _thresholds_display(thresholds):
    if not thresholds:
        return "—"
    return ", ".join(f"{name}={value}" for name, value in thresholds.items())


def aorta_run_card_display(variant, provenance):
    """Build an Aorta-specific run card from suite state."""
    variant = variant or {}
    nodes = variant.get("num_nodes")
    gpus_per_node = variant.get("gpus_per_node")
    total_gpus = variant.get("total_gpus")
    cluster = f"{nodes} nodes · {gpus_per_node} GPUs/node · {total_gpus} GPUs"
    channels = f"{variant.get('nccl_channels', '—')} NCCL · {variant.get('compute_channels', '—')} compute"
    artifacts = variant.get("artifacts") or []
    rows = [
        ("Status", variant.get("status") or "—", False),
        ("Parser", variant.get("parser") or "—", False),
        ("Parsed ranks", str(variant.get("parsed_ranks", "—")), False),
        ("Cluster", cluster, False),
        ("Channels", channels, False),
        ("RCCL branch", variant.get("rccl_branch") or "—", False),
        ("Launch mode", variant.get("launch_mode") or "—", False),
        ("Base config", variant.get("base_config") or "—", False),
        ("Container image", variant.get("image") or "—", False),
        ("Avg iteration", _display(variant.get("avg_iteration_time_ms"), " ms"), False),
        ("Rank time std dev", _display(variant.get("std_iteration_time_ms"), " ms", 3), False),
        ("Compute", _display(variant.get("avg_compute_ratio_pct"), "%"), False),
        ("Exposed communication", _display(variant.get("avg_communication_ratio_pct"), "%"), False),
        ("Compute/communication overlap", _display(variant.get("avg_overlap_ratio_pct"), "%"), False),
        ("Duration", _display(variant.get("duration_seconds"), " s", 1), False),
        ("Artifacts", ", ".join(artifacts) if artifacts else "—", False),
        ("Qualification thresholds", _thresholds_display(variant.get("thresholds")), False),
    ]
    if variant.get("samples_per_second") is not None:
        rows.append(("Throughput", _display(variant["samples_per_second"], " samples/s"), False))
    if variant.get("tokens_per_second") is not None:
        rows.append(("Throughput", _display(variant["tokens_per_second"], " tokens/s"), False))
    rows.extend(provenance_link_rows(provenance))
    return rows


__all__ = [
    "aorta_run_card_display",
    "build_aorta_series",
    "update_aorta_run_summary",
]
