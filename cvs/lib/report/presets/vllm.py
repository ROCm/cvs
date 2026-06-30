'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Inference report preset for the unified vLLM suite.

Auto-loaded when running ``cvs run vllm`` (the pytest module stem ``vllm``
matches this filename). Root ``cvs/conftest.py`` registers any
``InferenceReportConfig`` defined here via
``try_auto_register_inference_suite_report``; no suite conftest changes needed.

Render-only: produces the HTML/JSON dashboard + interactive viewer + CI summary
when ``--html`` is set. Does not change pass/fail or threshold enforcement.
'''

from __future__ import annotations

import dataclasses

from cvs.lib.inference.utils.vllm_parsing import (
    CLIENT_METRIC_UNITS,
    METRIC_TIER_ORDER,
    VLLM_RESULTS_COLUMNS,
    tier_metric_specs,
)
from cvs.lib.report.presets.builder import make_inference_report_config
from cvs.lib.report.types import ReportChartSeries

# Explicit highlights/charts: the builder would auto-derive all three charts from
# the first results columns (all throughput) -- a poor mix. Give the cell cards
# and concurrency charts a throughput + latency spread instead.
_CELL_HIGHLIGHTS = (
    ("output_throughput", "Output tok/s"),
    ("total_token_throughput", "Total tok/s"),
    ("mean_ttft_ms", "Mean TTFT (ms)"),
    ("mean_tpot_ms", "Mean TPOT (ms)"),
    ("p99_ttft_ms", "P99 TTFT (ms)"),
)
_CHART_SERIES = (
    ReportChartSeries("output_throughput", "Output tok/s", "tok/s"),
    ReportChartSeries("mean_ttft_ms", "Mean TTFT", "ms", invert=True),
    ReportChartSeries("mean_tpot_ms", "Mean TPOT", "ms", invert=True),
)

_BASE = make_inference_report_config(
    suite_id="vllm",
    report_basename="vllm_run_deck",
    title="vLLM Run Deck",
    subtitle="CVS vLLM · lab performance summary",
    results_columns=VLLM_RESULTS_COLUMNS,
    metric_units=CLIENT_METRIC_UNITS,
    tier_metric_specs=tier_metric_specs,
    metric_tier_order=METRIC_TIER_ORDER,
    inference_test_substring="test_vllm_inference",
    row_card_test_names=("test_metric",),
    cell_highlights=_CELL_HIGHLIGHTS,
    chart_series=_CHART_SERIES,
)

# Override the lifecycle labels to match what the unified vLLM suite actually
# records (vllm.py): container_launch, topology_discovery, model_fetch,
# server_ready, teardown. It does NOT record sshd_setup or client_complete, so
# the builder defaults would leave empty slots / omit topology_discovery.
# (make_inference_report_config sets these itself, so override post-build via
# dataclasses.replace rather than passing them as colliding kwargs.)
VLLM_REPORT_CONFIG = dataclasses.replace(
    _BASE,
    session_lifecycle_labels=(
        "container_launch",
        "topology_discovery",
        "model_fetch",
        "server_ready",
        "teardown",
    ),
    cell_lifecycle_labels=("server_ready",),
)
