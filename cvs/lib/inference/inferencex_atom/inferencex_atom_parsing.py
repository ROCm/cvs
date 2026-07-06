'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

InferenceX ATOM metric vocabulary and parsers.

ATOM ``benchmark_serving`` emits the same JSON scalar keys as stock vLLM bench,
so base parsing reuses :func:`vllm_parsing.to_client_metrics`. W1 IX gates
(``per_gpu_throughput``, ``output_tput_per_gpu``, tail percentiles) live here —
not in the vLLM single-node ``GATED_METRICS`` set (vLLM parity is a separate track).
'''

from __future__ import annotations

from cvs.lib.inference.utils.vllm_parsing import (
    CLIENT_METRICS as _VLLM_CLIENT_METRICS,
    GATED_METRICS as _VLLM_GATED_METRICS,
    _safe_div,
    to_client_metrics as _vllm_to_client_metrics,
)

_METRIC_INSERT = ("output_tput_per_gpu", "tok/s")
_idx = next(
    (i for i, (n, _) in enumerate(_VLLM_CLIENT_METRICS) if n == "per_gpu_throughput"),
    None,
)
CLIENT_METRICS = list(_VLLM_CLIENT_METRICS)
CLIENT_METRICS.insert(
    (_idx + 1) if _idx is not None else len(CLIENT_METRICS),
    _METRIC_INSERT,
)
CLIENT_METRIC_UNITS = dict(CLIENT_METRICS)

# IX W1 perf gates: vLLM baseline set plus per-GPU throughput derivations.
GATED_METRICS = frozenset(_VLLM_GATED_METRICS) | {
    "per_gpu_throughput",
    "output_tput_per_gpu",
}

# W1 metric tiers for ``test_cell_metrics`` (one parent row per cell × tier).
METRIC_TIERS: dict[str, tuple[str, ...]] = {
    "throughput": (
        "total_token_throughput",
        "output_throughput",
        "per_gpu_throughput",
        "output_tput_per_gpu",
    ),
    "ttft": (
        "mean_ttft_ms",
        "p99_ttft_ms",
    ),
    "tpot": (
        "mean_tpot_ms",
        "p99_tpot_ms",
    ),
    "health": (
        "success_rate",
        "failed",
    ),
}

METRIC_TIER_ORDER: tuple[str, ...] = tuple(METRIC_TIERS.keys()) + ("record",)

_tiered = {m for names in METRIC_TIERS.values() for m in names}
RECORD_METRICS: tuple[str, ...] = tuple(short for short, _unit in CLIENT_METRICS if short not in _tiered)

ENFORCED_METRICS = frozenset(_tiered)


def scaling_efficiency_pct(actual_output_throughput, *, baseline_single_node, nnodes):
    """Linear scaling efficiency: actual / (single-node baseline × nnodes)."""
    denom = _safe_div(baseline_single_node, 1)
    if denom is None or int(nnodes) < 1:
        return None
    ideal = denom * int(nnodes)
    if ideal <= 0:
        return None
    return _safe_div(actual_output_throughput, ideal)


def to_client_metrics(raw, *, tp, isl, scaling_baseline_output_throughput=None, nnodes=1):
    """Map an ATOM ``results.json`` dict to the ``client.*`` namespace for IX."""
    m = _vllm_to_client_metrics(raw, tp=tp, isl=isl)
    m["client.output_tput_per_gpu"] = _safe_div(raw.get("output_throughput"), tp)
    if scaling_baseline_output_throughput is not None:
        eff = scaling_efficiency_pct(
            raw.get("output_throughput"),
            baseline_single_node=scaling_baseline_output_throughput,
            nnodes=nnodes,
        )
        if eff is not None:
            m["scaling.efficiency_pct"] = eff * 100.0
    return m


def tier_metric_specs(thresholds_cell: dict, tier: str) -> dict[str, dict]:
    """Return ``client.*`` threshold specs for one tier in a sweep cell."""
    if tier == "record":
        names = RECORD_METRICS
    else:
        names = METRIC_TIERS.get(tier, ())
    specs = {}
    for short in names:
        full = f"client.{short}"
        spec = thresholds_cell.get(full)
        if spec is not None:
            specs[full] = spec
    return specs
