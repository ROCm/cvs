'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Auto-loaded when running ``cvs run inferencex_atom_vllm_single``.
'''

from cvs.lib.inference.inference_suite_results_table import INFERENCEX_ATOM_RESULTS_COLUMNS
from cvs.lib.inference.utils.inferencex_atom_parsing import (
    CLIENT_METRIC_UNITS,
    COMPARE_VLLM_METRIC_UNITS,
    SCALING_METRIC_UNITS,
    VLLM_METRIC_TIER_ORDER,
    tier_metric_specs,
)
from cvs.lib.report.presets.inferencex_atom import (
    _INFERENCEX_ATOM_CHART_SERIES,
    _atom_run_card_display,
    _CELL_LIFECYCLE,
    _SESSION_LIFECYCLE,
)
from cvs.lib.report.types import InferenceReportConfig, ReportChartSeries

_VLLM_PARITY_CHART_SERIES = (
    *_INFERENCEX_ATOM_CHART_SERIES,
    ReportChartSeries(
        "output_throughput_ratio",
        "vLLM / ATOM output tput",
        "ratio",
        metric_key="compare.vllm.output_throughput_ratio",
    ),
)

_VLLM_RESULTS_COLUMNS = INFERENCEX_ATOM_RESULTS_COLUMNS + (
    ("vLLM/ATOM tput", "compare.vllm.output_throughput_ratio"),
    ("vLLM/ATOM TTFT", "compare.vllm.mean_ttft_ms_ratio"),
)

INFERENCEX_ATOM_VLLM_SINGLE_REPORT_CONFIG = InferenceReportConfig(
    suite_id="inferencex_atom_vllm",
    report_basename="inferencex_atom_vllm_run_deck",
    title="IX Run Deck",
    subtitle="InferenceX vLLM parity \u00b7 W1 reference sweep",
    footer="CVS inferencex_atom_vllm_single \u00b7 render-only \u00b7 does not affect gates",
    link_name="IX vLLM Run Deck",
    results_columns=_VLLM_RESULTS_COLUMNS,
    metric_tier_order=VLLM_METRIC_TIER_ORDER,
    tier_metric_specs=tier_metric_specs,
    metric_units={**CLIENT_METRIC_UNITS, **SCALING_METRIC_UNITS, **COMPARE_VLLM_METRIC_UNITS},
    metric_prefix="client.",
    cell_highlights=(
        ("output_throughput", "Output tok/s"),
        ("mean_ttft_ms", "Mean TTFT (ms)"),
        ("mean_tpot_ms", "Mean TPOT (ms)"),
        ("compare.vllm.output_throughput_ratio", "vLLM/ATOM tput"),
        ("compare.vllm.mean_ttft_ms_ratio", "vLLM/ATOM TTFT"),
    ),
    chart_series=_VLLM_PARITY_CHART_SERIES,
    inference_test_substring="test_inferencex_atom_vllm_inference",
    session_lifecycle_labels=_SESSION_LIFECYCLE,
    cell_lifecycle_labels=_CELL_LIFECYCLE,
    row_card_extras=False,
    row_card_test_names=("test_cell_metrics",),
    interactive_viewer=True,
    viewer_cell_threshold=16,
    run_card_display_builder=_atom_run_card_display,
)
