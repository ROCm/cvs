'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Auto-loaded when running ``cvs run vllm`` (stem matches this filename -- see
``cvs.lib.report.auto_register``). Wires the unified single-node/distributed
vLLM suite (``cvs/tests/inference/vllm/vllm.py``) into the generic inference
report engine. Render-only: does not change pass/fail or threshold enforcement.
'''

from __future__ import annotations

from cvs.lib.inference.utils.vllm_parsing import (
    CLIENT_METRIC_UNITS,
    METRIC_TIER_ORDER,
    VLLM_RESULTS_COLUMNS,
    tier_metric_specs,
)
from cvs.lib.report.chart_presets import DEFAULT_PERF_CHART_SERIES
from cvs.lib.report.presets.builder import make_inference_report_config

# The suite records these lifecycle stages (cvs/tests/inference/vllm/vllm.py);
# it does NOT record "sshd_setup" or "client_complete" (the builder defaults
# assume both). Using the builder defaults here would leave two permanently
# empty timeline slots and silently drop the real "topology_discovery" stage.
VLLM_SESSION_LIFECYCLE_LABELS = (
    "container_launch",
    "topology_discovery",
    "model_fetch",
    "server_ready",
    "teardown",
)
VLLM_CELL_LIFECYCLE_LABELS = ("server_ready",)

VLLM_REPORT_CONFIG = make_inference_report_config(
    suite_id="vllm",
    report_basename="vllm_run_deck",
    title="vLLM Run Deck",
    subtitle="vLLM · single-node & PP-distributed lab performance summary",
    footer="CVS vllm · render-only · does not affect gates",
    link_name="vLLM Run Deck",
    results_columns=VLLM_RESULTS_COLUMNS,
    metric_units=CLIENT_METRIC_UNITS,
    tier_metric_specs=tier_metric_specs,
    metric_tier_order=METRIC_TIER_ORDER,
    chart_series=DEFAULT_PERF_CHART_SERIES,
    inference_test_substring="test_vllm_inference",
    row_card_test_names=("test_metric",),
    session_lifecycle_labels=VLLM_SESSION_LIFECYCLE_LABELS,
    cell_lifecycle_labels=VLLM_CELL_LIFECYCLE_LABELS,
)
