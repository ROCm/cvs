'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

vLLM metric hooks for Run Deck profiles (main-safe fallback when suite utils absent).
'''

from __future__ import annotations

try:
    from cvs.lib.inference.utils.vllm_parsing import (
        CLIENT_METRIC_UNITS,
        METRIC_TIER_ORDER,
        tier_metric_specs,
    )
except ImportError:
    CLIENT_METRIC_UNITS = {
        "request_throughput": "req/s",
        "output_throughput": "tok/s",
        "total_token_throughput": "tok/s",
        "mean_ttft_ms": "ms",
        "p95_ttft_ms": "ms",
        "mean_tpot_ms": "ms",
        "p95_tpot_ms": "ms",
        "p99_itl_ms": "ms",
        "goodput": "req/s",
        "success_rate": "ratio",
        "failed": "count",
    }
    METRIC_TIER_ORDER = ("throughput", "ttft", "tpot", "latency", "health", "record")
    _TIERS = {
        "throughput": ("total_token_throughput", "output_throughput", "request_throughput", "goodput"),
        "ttft": ("mean_ttft_ms", "p95_ttft_ms"),
        "tpot": ("mean_tpot_ms", "p95_tpot_ms"),
        "latency": ("p99_itl_ms",),
        "health": ("success_rate", "failed"),
    }
    _TIERED = {m for names in _TIERS.values() for m in names}
    _RECORD = tuple(k for k in CLIENT_METRIC_UNITS if k not in _TIERED)

    def tier_metric_specs(thresholds_cell: dict, tier: str) -> dict[str, dict]:
        names = _RECORD if tier == "record" else _TIERS.get(tier, ())
        specs = {}
        for short in names:
            full = f"client.{short}"
            spec = thresholds_cell.get(full)
            if spec is not None:
                specs[full] = spec
        return specs

__all__ = ["CLIENT_METRIC_UNITS", "METRIC_TIER_ORDER", "tier_metric_specs"]
