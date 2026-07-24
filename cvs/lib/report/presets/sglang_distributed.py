'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Auto-loaded when running ``pytest .../sglang_distributed.py`` (stem matches filename).

Wires the unified multi-node SGLang suite into the generic inference report engine.
Render-only: does not change pass/fail or threshold enforcement.
'''

from __future__ import annotations

from typing import Any, List, Tuple

from cvs.lib.inference.sglang.sglang_common import as_node_list, resolve_server_node_list
from cvs.lib.inference.sglang.sglang_parsing import (
    METRIC_TIER_ORDER,
    SGLANG_CHART_SERIES,
    SGLANG_METRIC_UNITS,
    SGLANG_RESULTS_COLUMNS,
    tier_metric_specs,
)
from cvs.lib.report.presets.builder import (
    make_inference_report_config,
    provenance_link_rows,
    thresholds_run_card_row,
)

SGLANG_DISTRIBUTED_SESSION_LIFECYCLE_LABELS = (
    "container_launch",
    "rms_norm",
    "server_launch",
    "server_ready",
    "smoke_endpoints",
    "lm_eval_hellaswag",
    "lm_eval_gsm8k",
    "gpu_topology",
    "teardown",
)


def _format_nodes(raw: Any) -> str:
    if not raw:
        return "\u2014"
    hosts = as_node_list(raw)
    return ", ".join(hosts) if hosts else "\u2014"


def _sglang_distributed_run_card(variant: Any, provenance: dict) -> List[Tuple[str, str, bool]]:
    bp = getattr(variant, "benchmark_params", None) or {}
    inf = getattr(variant, "inference", None) or {}
    try:
        server_nodes = resolve_server_node_list(inf)
    except ValueError:
        server_nodes = []
    rows: List[Tuple[str, str, bool]] = [
        ("Model", variant.model.id, False),
        ("GPU", variant.gpu_arch, False),
        ("Server nodes", ", ".join(server_nodes) if server_nodes else "\u2014", False),
        ("nnodes", str(inf.get("nnodes", len(server_nodes) or "-")), False),
        ("Benchmark node", _format_nodes(inf.get("benchmark_serv_node")), False),
        ("TP", str(bp.get("tensor_parallelism", "-")), False),
        ("PP", str(bp.get("pipeline_parallelism", "-")), False),
        thresholds_run_card_row(variant),
    ]
    rows.extend(provenance_link_rows(provenance))
    return rows


SGLANG_DISTRIBUTED_REPORT_CONFIG = make_inference_report_config(
    suite_id="sglang_distributed",
    report_basename="sglang_distributed_run_deck",
    title="SGLang Distributed Run Deck",
    subtitle="SGLang \u00b7 unified multi-node lab performance summary",
    footer="CVS sglang_distributed \u00b7 render-only \u00b7 does not affect gates",
    link_name="SGLang Distributed Run Deck",
    results_columns=SGLANG_RESULTS_COLUMNS,
    metric_units=SGLANG_METRIC_UNITS,
    tier_metric_specs=tier_metric_specs,
    metric_tier_order=METRIC_TIER_ORDER,
    metric_prefix="",
    cell_highlights=(
        ("output_throughput_per_sec", "Output tok/s"),
        ("mean_ttft_ms", "Mean TTFT (ms)"),
        ("mean_tpot_ms", "Mean TPOT (ms)"),
        ("goodput", "Goodput"),
        ("mfu", "MFU"),
    ),
    chart_series=SGLANG_CHART_SERIES,
    sweep_throughput_metric="output_throughput_per_sec",
    sweep_ttft_metric="mean_ttft_ms",
    headline_metric="output_throughput_per_sec",
    inference_test_substring="test_run_performance_benchmark_test",
    row_card_test_names=("test_run_performance_benchmark_test",),
    session_lifecycle_labels=SGLANG_DISTRIBUTED_SESSION_LIFECYCLE_LABELS,
    cell_lifecycle_labels=(),
    run_card_display_builder=_sglang_distributed_run_card,
)
