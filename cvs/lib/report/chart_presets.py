'''Shared sweep chart presets for inference suite reports.'''

from __future__ import annotations

from cvs.lib.report.types import ReportChartSeries

DEFAULT_PERF_CHART_SERIES: tuple[ReportChartSeries, ...] = (
    ReportChartSeries("output_throughput", "Output tok/s", "tok/s"),
    ReportChartSeries("total_token_throughput", "Total tok/s", "tok/s"),
    ReportChartSeries("mean_ttft_ms", "Mean TTFT", "ms", invert=True),
    ReportChartSeries("mean_tpot_ms", "Mean TPOT", "ms", invert=True),
    ReportChartSeries("p99_ttft_ms", "P99 TTFT", "ms", invert=True),
    ReportChartSeries("p99_tpot_ms", "P99 TPOT", "ms", invert=True),
    ReportChartSeries("p99_itl_ms", "P99 ITL", "ms", invert=True),
)
