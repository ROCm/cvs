'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Per-suite inference report presets. Import from here in suite ``conftest.py``; register
via ``cvs.lib.report.inference_wiring.configure_inference_suite_report``.
'''

from __future__ import annotations

from typing import Any, List, Tuple

from cvs.lib.inference.inference_suite_results_table import (
    INFERENCEX_ATOM_RESULTS_COLUMNS,
    VLLM_SINGLE_RESULTS_COLUMNS,
)
from cvs.lib.inference.utils.inferencex_atom_parsing import (
    CLIENT_METRIC_UNITS,
    METRIC_TIER_ORDER,
    tier_metric_specs,
)
from cvs.lib.inference.utils.vllm_parsing import CLIENT_METRIC_UNITS as VLLM_METRIC_UNITS
from cvs.lib.report.types import InferenceReportConfig, ReportChartSeries

_SESSION_LIFECYCLE = (
    "container_launch",
    "sshd_setup",
    "model_fetch",
    "server_ready",
    "client_complete",
    "teardown",
)

_CELL_LIFECYCLE = ("server_ready", "client_complete")


def _atom_run_card_display(variant: Any, provenance: dict) -> List[Tuple[str, str, bool]]:
    rc = variant.run_card
    enforce = bool(variant.enforce_thresholds)
    rows: List[Tuple[str, str, bool]] = [
        ("Model", variant.model.id, False),
        ("GPU", variant.gpu_arch, False),
        ("Driver", variant.params.driver, False),
        ("IX recipe", variant.ix_recipe_id or "\u2014", False),
        ("Image pin", rc.atom_image_pin or "\u2014", False),
        ("TP", str(variant.params.tensor_parallelism), False),
        ("Thresholds", "enforced" if enforce else "record-only", False),
    ]
    if rc.upstream_run_url:
        rows.append(("Upstream", rc.upstream_run_url, True))
    if provenance.get("pytest_html_path"):
        rows.append(("Pytest report", provenance["pytest_html_path"], True))
    if provenance.get("log_file_path"):
        rows.append(("Run log", provenance["log_file_path"], True))
    if rc.notes:
        rows.append(("Notes", rc.notes, False))
    return rows


INFERENCEX_ATOM_REPORT_CONFIG = InferenceReportConfig(
    suite_id="inferencex_atom",
    report_basename="inferencex_atom_report",
    title="IX Suite Report",
    subtitle="InferenceX ATOM \u00b7 lab performance summary",
    footer="CVS inferencex_atom_single \u00b7 render-only \u00b7 does not affect gates",
    link_name="IX Suite Report",
    embed_summary="IX Suite Report",
    results_columns=INFERENCEX_ATOM_RESULTS_COLUMNS,
    metric_tier_order=METRIC_TIER_ORDER,
    tier_metric_specs=tier_metric_specs,
    metric_units=CLIENT_METRIC_UNITS,
    metric_prefix="client.",
    cell_highlights=(
        ("output_throughput", "Output tok/s"),
        ("mean_ttft_ms", "Mean TTFT (ms)"),
        ("mean_tpot_ms", "Mean TPOT (ms)"),
        ("p99_ttft_ms", "P99 TTFT (ms)"),
        ("p95_tpot_ms", "P95 TPOT (ms)"),
    ),
    chart_series=(
        ReportChartSeries("output_throughput", "Output tok/s", "tok/s"),
        ReportChartSeries("mean_ttft_ms", "Mean TTFT", "ms", invert=True),
        ReportChartSeries("mean_tpot_ms", "Mean TPOT", "ms", invert=True),
    ),
    inference_test_substring="test_inferencex_atom_inference",
    session_lifecycle_labels=_SESSION_LIFECYCLE,
    cell_lifecycle_labels=_CELL_LIFECYCLE,
    parity_reference_framework_id="atom",
    viewer_cell_threshold=16,
    run_card_display_builder=_atom_run_card_display,
)


def _vllm_run_card_display(variant: Any, provenance: dict) -> List[Tuple[str, str, bool]]:
    rows: List[Tuple[str, str, bool]] = [
        ("Model", variant.model.id, False),
        ("GPU", variant.gpu_arch, False),
        ("Framework", variant.framework, False),
        ("TP", str(variant.params.tensor_parallelism), False),
        (
            "Thresholds",
            "enforced" if variant.enforce_thresholds else "record-only",
            False,
        ),
    ]
    if provenance.get("pytest_html_path"):
        rows.append(("Pytest report", provenance["pytest_html_path"], True))
    if provenance.get("log_file_path"):
        rows.append(("Run log", provenance["log_file_path"], True))
    return rows


# Wire with: test_report = make_report_test(VLLM_SINGLE_REPORT_CONFIG)
VLLM_SINGLE_REPORT_CONFIG = InferenceReportConfig(
    suite_id="vllm_single",
    report_basename="vllm_single_report",
    title="Inference Suite Report",
    subtitle="vLLM single-node \u00b7 lab performance summary",
    footer="CVS vllm_single \u00b7 render-only \u00b7 does not affect gates",
    link_name="Suite Report",
    embed_summary="Suite Report",
    results_columns=VLLM_SINGLE_RESULTS_COLUMNS,
    metric_tier_order=("throughput", "latency", "health", "record"),
    tier_metric_specs=lambda _cell, tier: {},
    metric_units=VLLM_METRIC_UNITS,
    metric_prefix="client.",
    cell_highlights=(
        ("output_throughput", "Output tok/s"),
        ("mean_ttft_ms", "Mean TTFT (ms)"),
        ("mean_tpot_ms", "Mean TPOT (ms)"),
    ),
    chart_series=(
        ReportChartSeries("output_throughput", "Output tok/s", "tok/s"),
        ReportChartSeries("mean_ttft_ms", "Mean TTFT", "ms", invert=True),
        ReportChartSeries("mean_tpot_ms", "Mean TPOT", "ms", invert=True),
    ),
    inference_test_substring="test_vllm_inference",
    session_lifecycle_labels=_SESSION_LIFECYCLE,
    cell_lifecycle_labels=_CELL_LIFECYCLE,
    run_card_display_builder=_vllm_run_card_display,
)
