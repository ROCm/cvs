'''Convert Megatron ``training_results_dict`` to per-node ``training_res_dict`` shape.'''

from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Sequence


def _last_scalar(values: Any) -> Optional[str]:
    if values is None:
        return None
    if isinstance(values, list):
        return str(values[-1]) if values else None
    return str(values)


def _count_nan_iterations(raw: Mapping[str, Any]) -> int:
    total = 0
    for key, values in raw.items():
        if not isinstance(values, list):
            continue
        for val in values:
            if re.search(r"nan|inf", str(val), re.I):
                total += 1
    return total


def megatron_to_training_res_dict(
    raw: Mapping[str, Any],
    host_list: Sequence[str],
) -> dict[str, dict]:
    """Map Megatron parse output (metric → list) to node-keyed metrics for reports."""
    if not raw:
        return {}

    mem_key = "mem_usage" if "mem_usage" in raw else "mem_usages"

    def node_metrics() -> dict:
        return {
            "throughput_per_gpu": _last_scalar(raw.get("throughput_per_gpu")),
            "tokens_per_gpu": _last_scalar(raw.get("tokens_per_gpu")),
            "elapsed_time_per_iteration": _last_scalar(raw.get("elapsed_time_per_iteration")),
            "nan_iterations": _count_nan_iterations(raw),
            "mem_usages": _last_scalar(raw.get(mem_key)),
        }

    if not host_list:
        return {"aggregate": node_metrics()}

    if len(host_list) == 1:
        return {str(host_list[0]): node_metrics()}

    # Distributed: parsed log is often from the last node; attach same snapshot per host.
    metrics = node_metrics()
    return {str(host): dict(metrics) for host in host_list}
