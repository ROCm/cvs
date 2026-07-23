'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Pure parsers and metric vocabulary for SGLang benchmark reports.

SGLang bench artifacts (log regex parsing in ``sglang_single_lib`` / ``sglang_disagg_lib``)
use bare metric keys (``mean_ttft_ms``, ``output_throughput_per_sec``) — not the
``client.*`` namespace vLLM uses. Report presets set ``metric_prefix=""`` accordingly.
'''

from __future__ import annotations

from cvs.lib.report.types import ReportChartSeries

SGLANG_METRIC_UNITS: dict[str, str] = {
    "request_throughput_per_sec": "req/s",
    "output_throughput_per_sec": "tok/s",
    "output_throughput_per_gpu_per_sec": "tok/s/GPU",
    "mean_ttft_ms": "ms",
    "median_ttft_ms": "ms",
    "p99_ttft_ms": "ms",
    "mean_tpot_ms": "ms",
    "median_tpot_ms": "ms",
    "p99_tpot_ms": "ms",
    "p99_itl_ms": "ms",
    "mean_e2e_latency_ms": "ms",
    "median_e2e_latency_ms": "ms",
    "p99_e2e_latency_ms": "ms",
    "goodput": "ratio",
    "mfu": "ratio",
}

SGLANG_RESULTS_COLUMNS = (
    ("Model", None),
    ("GPU", None),
    ("ISL", None),
    ("OSL", None),
    ("Policy", None),
    ("Conc", None),
    ("Host", None),
    ("Req/s", "request_throughput_per_sec"),
    ("Output tok/s", "output_throughput_per_sec"),
    ("Mean TTFT (ms)", "mean_ttft_ms"),
    ("Mean TPOT (ms)", "mean_tpot_ms"),
    ("P99 ITL (ms)", "p99_itl_ms"),
    ("Mean E2E latency (ms)", "mean_e2e_latency_ms"),
    ("Goodput", "goodput"),
    ("MFU (estimated)", "mfu"),
)

METRIC_TIERS: dict[str, tuple[str, ...]] = {
    "throughput": (
        "output_throughput_per_sec",
        "request_throughput_per_sec",
        "output_throughput_per_gpu_per_sec",
    ),
    "latency": (
        "mean_ttft_ms",
        "mean_tpot_ms",
        "p99_ttft_ms",
        "p99_tpot_ms",
        "p99_itl_ms",
        "mean_e2e_latency_ms",
    ),
    "health": (
        "goodput",
        "mfu",
    ),
}

METRIC_TIER_ORDER: tuple[str, ...] = tuple(METRIC_TIERS.keys()) + ("record",)

_tiered = {m for names in METRIC_TIERS.values() for m in names}
RECORD_METRICS: tuple[str, ...] = tuple(
    short for short in SGLANG_METRIC_UNITS if short not in _tiered
)

SGLANG_CHART_SERIES: tuple[ReportChartSeries, ...] = (
    ReportChartSeries("output_throughput_per_sec", "Output tok/s", "tok/s"),
    ReportChartSeries("request_throughput_per_sec", "Req/s", "req/s"),
    ReportChartSeries("mean_ttft_ms", "Mean TTFT", "ms", invert=True),
    ReportChartSeries("mean_tpot_ms", "Mean TPOT", "ms", invert=True),
    ReportChartSeries("p99_ttft_ms", "P99 TTFT", "ms", invert=True),
    ReportChartSeries("p99_tpot_ms", "P99 TPOT", "ms", invert=True),
)


def tier_metric_specs(thresholds_cell: dict, tier: str) -> dict[str, dict]:
    """Return threshold specs for one tier in a sweep cell (bare metric keys)."""
    names = RECORD_METRICS if tier == "record" else METRIC_TIERS.get(tier, ())
    specs: dict[str, dict] = {}
    for name in names:
        spec = thresholds_cell.get(name)
        if spec is not None:
            specs[name] = spec
    return specs
