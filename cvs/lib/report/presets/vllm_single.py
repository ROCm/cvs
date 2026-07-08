'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Auto-loaded when running ``cvs run vllm_single`` (stem matches filename).
'''

from __future__ import annotations

from typing import Any, List, Tuple

from cvs.lib.inference.inference_suite_results_table import VLLM_SINGLE_RESULTS_COLUMNS
from cvs.lib.inference.utils.vllm_parsing import CLIENT_METRIC_UNITS, CLIENT_METRICS
from cvs.lib.report.chart_presets import DEFAULT_PERF_CHART_SERIES
from cvs.lib.report.presets.builder import (
    make_inference_report_config,
    provenance_link_rows,
    thresholds_run_card_row,
)

_VLLM_METRIC_TIERS: dict[str, tuple[str, ...]] = {
    "throughput": ("total_token_throughput", "output_throughput"),
    "ttft": ("mean_ttft_ms", "p99_ttft_ms"),
    "tpot": ("mean_tpot_ms", "p99_tpot_ms"),
    "itl": ("mean_itl_ms", "p99_itl_ms"),
    "e2el": ("mean_e2el_ms", "p99_e2el_ms"),
    "health": ("success_rate", "failed"),
}

_VLLM_TIER_ORDER: tuple[str, ...] = tuple(_VLLM_METRIC_TIERS.keys()) + ("record",)

_tiered = {m for names in _VLLM_METRIC_TIERS.values() for m in names}
_VLLM_RECORD_METRICS: tuple[str, ...] = tuple(
    short for short, _unit in CLIENT_METRICS if short not in _tiered
)

_VLLM_SESSION_LIFECYCLE: tuple[str, ...] = (
    "container_launch",
    "topology_discovery",
    "model_fetch",
    "server_ready",
    "client_complete",
    "teardown",
)


def _vllm_tier_metric_specs(thresholds_cell: dict, tier: str) -> dict[str, dict]:
    if tier == "record":
        names = _VLLM_RECORD_METRICS
    else:
        names = _VLLM_METRIC_TIERS.get(tier, ())
    specs = {}
    for short in names:
        full = f"client.{short}"
        spec = thresholds_cell.get(full)
        if spec is not None:
            specs[full] = spec
    return specs


def _vllm_run_card_display(variant: Any, provenance: dict) -> List[Tuple[str, str, bool]]:
    rows: List[Tuple[str, str, bool]] = [
        ("Model", getattr(getattr(variant, "model", None), "id", "\u2014"), False),
        ("GPU", getattr(variant, "gpu_arch", "\u2014"), False),
        ("Framework", "vllm", False),
        thresholds_run_card_row(variant),
    ]
    params = getattr(variant, "params", None)
    if params is not None and hasattr(params, "tensor_parallelism"):
        rows.append(("TP", str(params.tensor_parallelism), False))
    rows.extend(provenance_link_rows(provenance))
    return rows


VLLM_SINGLE_REPORT_CONFIG = make_inference_report_config(
    suite_id="vllm",
    report_basename="vllm_run_deck",
    title="vLLM Run Deck",
    subtitle="CVS vLLM \u00b7 lab performance summary",
    footer="CVS vllm \u00b7 render-only \u00b7 does not affect gates",
    link_name="vLLM Run Deck",
    results_columns=VLLM_SINGLE_RESULTS_COLUMNS,
    metric_units=CLIENT_METRIC_UNITS,
    tier_metric_specs=_vllm_tier_metric_specs,
    metric_tier_order=_VLLM_TIER_ORDER,
    inference_test_substring="test_vllm_inference",
    chart_series=DEFAULT_PERF_CHART_SERIES,
    row_card_test_names=("test_metric",),
    run_card_display_builder=_vllm_run_card_display,
    session_lifecycle_labels=_VLLM_SESSION_LIFECYCLE,
)
