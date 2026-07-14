'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Shared types for CVS inference suite reports.
'''

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Tuple

TierMetricSpecsFn = Callable[[dict, str], dict[str, dict]]
RunCardDisplayFn = Callable[[Any, dict], List[Tuple[str, str, bool]]]

DEFAULT_SESSION_LIFECYCLE_LABELS: tuple[str, ...] = (
    "container_launch",
    "sshd_setup",
    "model_fetch",
    "server_ready",
    "client_complete",
    "teardown",
)
DEFAULT_CELL_LIFECYCLE_LABELS: tuple[str, ...] = ("server_ready", "client_complete")


@dataclass(frozen=True)
class ReportChartSeries:
    """One concurrency chart; ``metric_suffix`` is the short name after ``metric_prefix``."""

    metric_suffix: str
    title: str
    unit: str
    invert: bool = False


@dataclass(frozen=True)
class InferenceReportConfig:
    """Per-suite binding for the generic inference suite report renderer."""

    suite_id: str
    report_basename: str
    title: str
    subtitle: str
    footer: str
    link_name: str
    results_columns: tuple
    metric_tier_order: tuple[str, ...]
    tier_metric_specs: TierMetricSpecsFn
    metric_units: dict[str, str]
    metric_prefix: str = "client."
    cell_highlights: tuple[tuple[str, str], ...] = ()
    chart_series: tuple[ReportChartSeries, ...] = ()
    record_tier: str = "record"
    inference_test_substring: str = "test_inference"
    session_lifecycle_labels: tuple[str, ...] = DEFAULT_SESSION_LIFECYCLE_LABELS
    cell_lifecycle_labels: tuple[str, ...] = DEFAULT_CELL_LIFECYCLE_LABELS
    sweep_throughput_metric: str = "client.output_throughput"
    sweep_ttft_metric: str = "client.mean_ttft_ms"
    headline_metric: str = "client.output_throughput"
    row_card_extras: bool = True
    row_card_test_names: tuple[str, ...] = ("test_metric", "test_cell_metrics")
    interactive_viewer: bool = True
    viewer_cell_threshold: int = 24
    prev_run_json: str = ""
    run_card_display_builder: RunCardDisplayFn = field(default=lambda _variant, _prov: [("Suite", "inference", False)])

    @property
    def gated_tiers(self) -> tuple[str, ...]:
        return tuple(t for t in self.metric_tier_order if t != self.record_tier)

    def full_metric(self, short: str) -> str:
        if short.startswith(f"{self.metric_prefix}"):
            return short
        return f"{self.metric_prefix}{short}"
