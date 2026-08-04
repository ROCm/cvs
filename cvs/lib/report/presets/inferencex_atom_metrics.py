'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

InferenceX ATOM metric hooks for Run Deck profiles.

These live in the report layer so ``profiles/inferencex_atom_single.json`` works
on ``main`` before the full ``inferencex_atom`` suite merges from ``dev/dtni``.
When the suite package is present, definitions are re-exported from there.
'''

from __future__ import annotations

try:
    from cvs.lib.inference.inferencex_atom.inferencex_atom_parsing import (
        CLIENT_METRIC_UNITS,
        METRIC_TIER_ORDER,
        tier_metric_specs,
    )
except ImportError:
    # Minimal inline definitions for main-branch CI and demo generation.
    CLIENT_METRIC_UNITS = {
        "output_throughput": "tok/s",
        "total_token_throughput": "tok/s",
        "per_gpu_throughput": "tok/s/gpu",
        "output_tput_per_gpu": "tok/s/gpu",
        "mean_ttft_ms": "ms",
        "p99_ttft_ms": "ms",
        "mean_tpot_ms": "ms",
        "p99_tpot_ms": "ms",
        "p99_itl_ms": "ms",
        "success_rate": "ratio",
        "failed": "count",
    }

    METRIC_TIER_ORDER = ("throughput", "ttft", "tpot", "health", "record")

    _METRIC_TIERS = {
        "throughput": (
            "total_token_throughput",
            "output_throughput",
            "per_gpu_throughput",
            "output_tput_per_gpu",
        ),
        "ttft": ("mean_ttft_ms", "p99_ttft_ms"),
        "tpot": ("mean_tpot_ms", "p99_tpot_ms"),
        "health": ("success_rate", "failed"),
    }
    _TIERED = {m for names in _METRIC_TIERS.values() for m in names}
    _RECORD = tuple(k for k in CLIENT_METRIC_UNITS if k not in _TIERED)

    def tier_metric_specs(thresholds_cell: dict, tier: str) -> dict[str, dict]:
        if tier == "record":
            names = _RECORD
        else:
            names = _METRIC_TIERS.get(tier, ())
        specs = {}
        for short in names:
            full = f"client.{short}"
            spec = thresholds_cell.get(full)
            if spec is not None:
                specs[full] = spec
        return specs


__all__ = ["CLIENT_METRIC_UNITS", "METRIC_TIER_ORDER", "tier_metric_specs"]
