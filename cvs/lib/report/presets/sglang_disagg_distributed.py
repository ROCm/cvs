'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Report preset for ``cvs run sglang_disagg_distributed``.

Register explicitly from ``cvs/tests/inference/sglang/conftest.py`` via
``configure_inference_suite_report(config, SGLANG_DISAGG_REPORT_CONFIG)``.
'''

from __future__ import annotations

from typing import Any, List, Tuple

from cvs.lib.report.presets.builder import make_inference_report_config
from cvs.lib.report.types import InferenceReportConfig, ReportChartSeries

SGLANG_DISAGG_RESULTS_COLUMNS = (
    # Fixed columns (from 6-tuple key + host; None = not read from metrics dict)
    ("Model", None),
    ("GPU", None),
    ("ISL", None),
    ("OSL", None),
    ("Policy", None),
    ("Conc", None),
    ("Host", None),
    # Throughput / efficiency
    ("Req/s", "request_throughput_per_sec"),
    ("Output tok/s", "output_throughput_per_sec"),
    ("Output tok/s/GPU", "output_throughput_per_gpu_per_sec"),
    ("Goodput", "goodput"),
    ("MFU", "mfu"),
    # Latency
    ("Mean TTFT (ms)", "mean_ttft_ms"),
    ("Mean TPOT (ms)", "mean_tpot_ms"),
    ("P99 ITL (ms)", "p99_itl_ms"),
    ("Mean E2E latency (ms)", "mean_e2e_latency_ms"),
)

SGLANG_METRIC_UNITS = {
    "mean_ttft_ms": "ms",
    "mean_tpot_ms": "ms",
    "p99_itl_ms": "ms",
    "mean_e2e_latency_ms": "ms",
    "request_throughput_per_sec": "req/s",
    "output_throughput_per_sec": "tok/s",
    "output_throughput_per_gpu_per_sec": "tok/s/GPU",
    "goodput": "",
    "mfu": "",
}


def _sglang_run_card(variant: Any, provenance: dict) -> List[Tuple[str, str, bool]]:
    rows = [
        ("Model", str(getattr(variant, "model", "—")), False),
        ("Variant", getattr(variant, "variant_key", "—"), False),
        (
            "Thresholds",
            "enforced" if getattr(variant, "enforce_thresholds", False) else "record-only",
            False,
        ),
    ]
    if provenance.get("pytest_html_path"):
        rows.append(("Pytest report", provenance["pytest_html_path"], True))
    if provenance.get("log_file_path"):
        rows.append(("Run log", provenance["log_file_path"], True))
    return rows


def _tier_metric_specs(_cell: dict, tier: str) -> dict:
    """Record-only for now; add gate specs when cell_key/thresholds align."""
    return {}


SGLANG_DISAGG_REPORT_CONFIG = InferenceReportConfig(
    suite_id="sglang_disagg_distributed",
    report_basename="sglang_disagg_report",
    title="SGLang Disagg Run Deck",
    subtitle="SGLang disaggregated PD · lab performance summary",
    footer="CVS sglang_disagg_distributed · render-only · does not affect gates",
    link_name="SGLang Disagg Report",
    results_columns=SGLANG_DISAGG_RESULTS_COLUMNS,
    metric_tier_order=("throughput", "latency", "health", "record"),
    tier_metric_specs=_tier_metric_specs,
    metric_units=SGLANG_METRIC_UNITS,
    metric_prefix="",
    cell_highlights=(
        ("output_throughput_per_sec", "Output tok/s"),
        ("output_throughput_per_gpu_per_sec", "Output tok/s/GPU"),
        ("request_throughput_per_sec", "Req/s"),
        ("mean_ttft_ms", "Mean TTFT (ms)"),
        ("mean_tpot_ms", "Mean TPOT (ms)"),
        ("p99_itl_ms", "P99 ITL (ms)"),
        ("mean_e2e_latency_ms", "Mean E2E (ms)"),
        ("goodput", "Goodput"),
        ("mfu", "MFU"),
    ),
    chart_series=(
        ReportChartSeries("output_throughput_per_sec", "Output tok/s", "tok/s"),
        ReportChartSeries("output_throughput_per_gpu_per_sec", "Output tok/s/GPU", "tok/s/GPU"),
        ReportChartSeries("request_throughput_per_sec", "Req/s", "req/s"),
        ReportChartSeries("goodput", "Goodput", ""),
        ReportChartSeries("mfu", "MFU", ""),
        ReportChartSeries("mean_ttft_ms", "Mean TTFT", "ms", invert=True),
        ReportChartSeries("mean_tpot_ms", "Mean TPOT", "ms", invert=True),
        ReportChartSeries("p99_itl_ms", "P99 ITL", "ms", invert=True),
        ReportChartSeries("mean_e2e_latency_ms", "Mean E2E latency", "ms", invert=True),
    ),
    inference_test_substring="test_run_performance_benchmark",
    sweep_throughput_metric="output_throughput_per_gpu_per_sec",
    sweep_ttft_metric="mean_ttft_ms",
    headline_metric="output_throughput_per_gpu_per_sec",
    session_lifecycle_labels=(
        "stale_cleanup", "container_launch", "ibv_setup", "rms_norm",
        "prefill_launch", "decode_launch", "server_ready", "proxy_router_launch",
        "smoke_endpoints", "bench_serv_random", "gpu_topology", "teardown",
    ),
    row_card_extras=False,
    run_card_display_builder=_sglang_run_card,
)