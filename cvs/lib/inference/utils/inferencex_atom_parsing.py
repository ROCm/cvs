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
CLIENT_METRICS: list[tuple[str, str]] = []
_inserted = False
for _name, _unit in _VLLM_CLIENT_METRICS:
    CLIENT_METRICS.append((_name, _unit))
    if _name == "per_gpu_throughput" and not _inserted:
        CLIENT_METRICS.append(_METRIC_INSERT)
        _inserted = True
if not _inserted:
    CLIENT_METRICS.append(_METRIC_INSERT)
CLIENT_METRIC_UNITS = dict(CLIENT_METRICS)

# IX W1 perf gates: vLLM baseline set plus per-GPU throughput derivations.
GATED_METRICS = frozenset(_VLLM_GATED_METRICS) | {
    "per_gpu_throughput",
    "output_tput_per_gpu",
}


def to_client_metrics(raw, *, tp, isl):
    """Map an ATOM ``results.json`` dict to the ``client.*`` namespace for IX."""
    m = _vllm_to_client_metrics(raw, tp=tp, isl=isl)
    m["client.output_tput_per_gpu"] = _safe_div(raw.get("output_throughput"), tp)
    return m
