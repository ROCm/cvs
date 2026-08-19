'''
Copyright 2025 Advanced Micro Devices Inc.
All rights reserved.

Opt-in amd-smi GPU polling during atom inference cells (INF-7).
'''

from __future__ import annotations

from typing import Any, Mapping


def capture_gpu_snap(orch) -> dict:
    try:
        from cvs.lib.utils.gpu import capture_gpu_metrics

        return capture_gpu_metrics(orch) or {}
    except Exception:
        return {}


def gpu_results_from_poll(poll_readings: list, *, load_s=None, load_mb=None) -> dict[str, Any]:
    from cvs.lib.utils.gpu import agg_readings

    agg = agg_readings(poll_readings)
    out = {
        "gpu.peak_gpu_memory_mb": agg.get("peak_gpu_memory_mb"),
        "gpu.gpu_bandwidth_util_pct": agg.get("gpu_bandwidth_util_pct"),
        "gpu.gpu_compute_util_pct": agg.get("gpu_compute_util_pct"),
    }
    if load_s is not None:
        out["gpu.model_load_s"] = load_s
    if load_mb is not None:
        out["gpu.model_load_memory_mb"] = load_mb
    return out


def merge_gpu_into_results(results: Mapping[Any, dict], gpu_results: Mapping[str, Any]) -> None:
    if not gpu_results:
        return
    for host_actuals in results.values():
        if isinstance(host_actuals, dict):
            host_actuals.update(gpu_results)
