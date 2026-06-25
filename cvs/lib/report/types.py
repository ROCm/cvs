'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Shared types for CVS suite reports (inference, training, parity).
'''

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Tuple

TierMetricSpecsFn = Callable[[dict, str], dict[str, dict]]
RunCardDisplayFn = Callable[[Any, dict], List[Tuple[str, str, bool]]]


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
    embed_summary: str
    results_columns: tuple
    metric_tier_order: tuple[str, ...]
    tier_metric_specs: TierMetricSpecsFn
    metric_units: dict[str, str]
    metric_prefix: str = "client."
    cell_highlights: tuple[tuple[str, str], ...] = ()
    chart_series: tuple[ReportChartSeries, ...] = ()
    record_tier: str = "record"
    inference_test_substring: str = "test_inference"
    session_lifecycle_labels: tuple[str, ...] = (
        "container_launch",
        "sshd_setup",
        "model_fetch",
        "server_ready",
        "client_complete",
        "teardown",
    )
    cell_lifecycle_labels: tuple[str, ...] = ("server_ready", "client_complete")
    sweep_throughput_metric: str = "client.output_throughput"
    sweep_ttft_metric: str = "client.mean_ttft_ms"
    headline_metric: str = "client.output_throughput"
    parity_reference_framework_id: str = ""
    parity_compare_jsons: tuple[tuple[str, str], ...] = ()
    parity_metrics: tuple = ()
    scaling_baseline_json: str = ""
    prev_run_json: str = ""
    row_card_extras: bool = True
    row_card_test_names: tuple[str, ...] = ("test_metric", "test_cell_metrics")
    interactive_viewer: bool = True
    viewer_cell_threshold: int = 24
    run_card_display_builder: RunCardDisplayFn = field(
        default=lambda _variant, _prov: [("Suite", "inference", False)]
    )

    @property
    def gated_tiers(self) -> tuple[str, ...]:
        return tuple(t for t in self.metric_tier_order if t != self.record_tier)

    def full_metric(self, short: str) -> str:
        if short.startswith(f"{self.metric_prefix}"):
            return short
        return f"{self.metric_prefix}{short}"


@dataclass(frozen=True)
class TrainingReportConfig:
    """Per-suite binding for training suite reports (Megatron / JAX)."""

    suite_id: str
    report_basename: str
    title: str
    subtitle: str
    footer: str
    link_name: str
    node_metric_columns: tuple[tuple[str, str], ...] = (
        ("throughput_per_gpu", "Throughput / GPU"),
        ("tokens_per_gpu", "Tokens / GPU"),
        ("elapsed_time_per_iteration", "s / iter"),
        ("nan_iterations", "NaN iters"),
        ("mem_usages", "Mem usage"),
    )
    parity_baseline_json: str = ""
    parity_reference_label: str = "Baseline"
    parity_candidate_label: str = "This run"
    parity_metric_key: str = "throughput_per_gpu"


@dataclass(frozen=True)
class InferenceParitySource:
    """One framework's inference suite report JSON sidecar."""

    framework_id: str
    label: str
    json_path: str


@dataclass(frozen=True)
class InferenceParityMetric:
    """One ``compare.*`` ratio between a reference and comparator framework."""

    metric: str
    compare_framework_id: str
    ratio_key: str
    reference_framework_id: str = ""
    title: str = ""


@dataclass(frozen=True)
class InferenceParityConfig:
    """Binding for a standalone inference framework parity report (M4)."""

    report_basename: str = "inference_parity_report"
    title: str = "Inference framework parity"
    subtitle: str = "Aligned sweep cells across framework report JSON sidecars"
    footer: str = "CVS inference parity · render-only"
    link_name: str = "Inference parity report"
    reference_framework_id: str = ""
    sources: tuple[InferenceParitySource, ...] = ()
    metrics: tuple[InferenceParityMetric, ...] = ()
