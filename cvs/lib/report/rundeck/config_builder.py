'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Build an ``InferenceReportConfig`` with CVS-wide defaults so suite owners only
supply columns, metrics, tiers, and the workload test name substring.
'''

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

from cvs.lib.report.chart_presets import DEFAULT_PERF_CHART_SERIES
from cvs.lib.report.types import (
    DEFAULT_CELL_LIFECYCLE_LABELS,
    DEFAULT_SESSION_LIFECYCLE_LABELS,
    InferenceReportConfig,
    ReportChartSeries,
    TierMetricSpecsFn,
)


def provenance_link_rows(provenance: dict) -> List[Tuple[str, str, bool]]:
    rows: List[Tuple[str, str, bool]] = []
    pytest_href = provenance.get("pytest_html_href") or provenance.get("pytest_html_path")
    if pytest_href:
        rows.append(("Pytest report", str(pytest_href), True))
    log_href = provenance.get("log_file_href") or provenance.get("log_file_path")
    if log_href:
        rows.append(("Run log", str(log_href), True))
    return rows


def thresholds_run_card_row(variant: Any) -> Tuple[str, str, bool]:
    enforce = bool(getattr(variant, "enforce_thresholds", False))
    return ("Thresholds", "enforced" if enforce else "record-only", False)


def _default_run_card(variant: Any, provenance: dict) -> List[Tuple[str, str, bool]]:
    rows: List[Tuple[str, str, bool]] = [
        ("Model", getattr(getattr(variant, "model", None), "id", "\u2014"), False),
        ("GPU", getattr(variant, "gpu_arch", "\u2014"), False),
        thresholds_run_card_row(variant),
    ]
    framework = getattr(variant, "framework", None)
    if framework:
        rows.insert(2, ("Framework", framework, False))
    params = getattr(variant, "params", None)
    if params is not None and hasattr(params, "tensor_parallelism"):
        rows.append(("TP", str(params.tensor_parallelism), False))
    rows.extend(provenance_link_rows(provenance))
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
    charts = chart_series or DEFAULT_PERF_CHART_SERIES
    workload = inference_test_substring or f"test_{suite_id}"
    session_labels = kwargs.pop("session_lifecycle_labels", DEFAULT_SESSION_LIFECYCLE_LABELS)
    cell_labels = kwargs.pop("cell_lifecycle_labels", DEFAULT_CELL_LIFECYCLE_LABELS)
    row_card_extras = kwargs.pop("row_card_extras", True)
    interactive_viewer = kwargs.pop("interactive_viewer", True)
    viewer_cell_threshold = kwargs.pop("viewer_cell_threshold", 24)

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
        row_card_extras=row_card_extras,
        row_card_test_names=row_card_test_names,
        interactive_viewer=interactive_viewer,
        viewer_cell_threshold=viewer_cell_threshold,
        run_card_display_builder=run_card_display_builder or _default_run_card,
        **kwargs,
    )
