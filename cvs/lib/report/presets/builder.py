'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Build an ``InferenceReportConfig`` with CVS-wide defaults so suite owners only
supply columns, metrics, tiers, and the workload test name substring.
'''

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

from cvs.lib.report.chart_presets import DEFAULT_PERF_CHART_SERIES
from cvs.lib.report.types import InferenceReportConfig, ReportChartSeries, TierMetricSpecsFn

_SESSION_LIFECYCLE = (
    "container_launch",
    "sshd_setup",
    "model_fetch",
    "server_ready",
    "client_complete",
    "teardown",
)

_CELL_LIFECYCLE = ("server_ready", "client_complete")


def _default_run_card(variant: Any, provenance: dict) -> List[Tuple[str, str, bool]]:
    rows: List[Tuple[str, str, bool]] = [
        ("Model", getattr(getattr(variant, "model", None), "id", "\u2014"), False),
        ("GPU", getattr(variant, "gpu_arch", "\u2014"), False),
        (
            "Thresholds",
            "enforced" if getattr(variant, "enforce_thresholds", False) else "record-only",
            False,
        ),
    ]
    framework = getattr(variant, "framework", None)
    if framework:
        rows.insert(2, ("Framework", framework, False))
    params = getattr(variant, "params", None)
    if params is not None and hasattr(params, "tensor_parallelism"):
        rows.append(("TP", str(params.tensor_parallelism), False))
    if provenance.get("pytest_html_path"):
        rows.append(("Pytest report", provenance["pytest_html_path"], True))
    if provenance.get("log_file_path"):
        rows.append(("Run log", provenance["log_file_path"], True))
    return rows


def _highlights_from_columns(
    results_columns: tuple,
    metric_units: dict[str, str],
    *,
    limit: int = 5,
) -> tuple[tuple[str, str], ...]:
    out: list[tuple[str, str]] = []
    for label, key in results_columns:
        if not key or not str(key).startswith("client."):
            continue
        short = str(key).split(".", 1)[-1]
        if short in metric_units or short.endswith("_ms") or "throughput" in short:
            out.append((short, label))
        if len(out) >= limit:
            break
    if out:
        return tuple(out)
    return (("output_throughput", "Output tok/s"),)


def _charts_from_highlights(
    highlights: tuple[tuple[str, str], ...],
    metric_units: dict[str, str],
) -> tuple[ReportChartSeries, ...]:
    del highlights, metric_units  # builder default uses the shared perf sweep set
    return DEFAULT_PERF_CHART_SERIES


def make_inference_report_config(
    *,
    suite_id: str,
    results_columns: tuple,
    metric_units: dict[str, str],
    tier_metric_specs: TierMetricSpecsFn,
    metric_tier_order: tuple[str, ...] = ("throughput", "latency", "health", "record"),
    inference_test_substring: Optional[str] = None,
    report_basename: Optional[str] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    footer: Optional[str] = None,
    link_name: Optional[str] = None,
    cell_highlights: Optional[tuple[tuple[str, str], ...]] = None,
    chart_series: Optional[tuple[ReportChartSeries, ...]] = None,
    row_card_test_names: tuple[str, ...] = ("test_metric", "test_cell_metrics"),
    run_card_display_builder: Optional[Callable[..., List[Tuple[str, str, bool]]]] = None,
    **kwargs,
) -> InferenceReportConfig:
    """Return a report preset with sensible defaults for any ``cvs run`` inference suite."""
    basename = report_basename or f"{suite_id}_report"
    display_title = title or f"{suite_id.replace('_', ' ').title()} Report"
    highlights = cell_highlights or _highlights_from_columns(results_columns, metric_units)
    charts = chart_series or _charts_from_highlights(highlights, metric_units)
    workload = inference_test_substring or f"test_{suite_id}"
    session_labels = kwargs.pop("session_lifecycle_labels", _SESSION_LIFECYCLE)
    cell_labels = kwargs.pop("cell_lifecycle_labels", _CELL_LIFECYCLE)

    return InferenceReportConfig(
        suite_id=suite_id,
        report_basename=basename,
        title=display_title,
        subtitle=subtitle or f"{suite_id} \u00b7 lab performance summary",
        footer=footer or f"CVS {suite_id} \u00b7 render-only \u00b7 does not affect gates",
        link_name=link_name or display_title,
        results_columns=results_columns,
        metric_tier_order=metric_tier_order,
        tier_metric_specs=tier_metric_specs,
        metric_units=metric_units,
        cell_highlights=highlights,
        chart_series=charts,
        inference_test_substring=workload,
        session_lifecycle_labels=session_labels,
        cell_lifecycle_labels=cell_labels,
        row_card_extras=True,
        row_card_test_names=row_card_test_names,
        interactive_viewer=True,
        run_card_display_builder=run_card_display_builder or _default_run_card,
        **kwargs,
    )
