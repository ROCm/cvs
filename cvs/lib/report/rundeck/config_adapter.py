'''
Copyright 2025 Advanced Micro Devices Inc.
All rights reserved.

Resolve deck profile hooks and build ``InferenceReportConfig`` from JSON profiles.
'''

from __future__ import annotations

import importlib
from typing import Any, Callable

from cvs.lib.report.rundeck.config_builder import make_inference_report_config
from cvs.lib.report.profile import DeckProfile
from cvs.lib.report.types import InferenceReportConfig, ReportChartSeries


def import_object(spec: str) -> Any:
    """Import ``module.path:attr`` from a profile hook or constant reference."""
    if ":" not in spec:
        raise ValueError(f"Hook spec must be module:attr, got {spec!r}")
    module_name, attr_name = spec.split(":", 1)
    return getattr(importlib.import_module(module_name), attr_name)


def import_callable(spec: str) -> Callable[..., Any]:
    obj = import_object(spec)
    if not callable(obj):
        raise TypeError(f"Hook {spec!r} is not callable")
    return obj


def _parse_results_columns(raw: list) -> tuple:
    out = []
    for entry in raw:
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            label, key = entry
            out.append((label, key))
        elif isinstance(entry, dict):
            out.append((entry["label"], entry.get("key")))
    return tuple(out)


def _parse_chart_series(raw: list) -> tuple[ReportChartSeries, ...]:
    series = []
    for entry in raw:
        if isinstance(entry, dict):
            series.append(
                ReportChartSeries(
                    entry["metric_suffix"],
                    entry["title"],
                    entry["unit"],
                    invert=bool(entry.get("invert", False)),
                )
            )
    return tuple(series)


def _parse_cell_highlights(raw: list) -> tuple[tuple[str, str], ...]:
    out = []
    for entry in raw:
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            out.append((entry[0], entry[1]))
        elif isinstance(entry, dict):
            out.append((entry["suffix"], entry["label"]))
    return tuple(out)


def build_inference_config_from_profile(profile: dict[str, Any]) -> InferenceReportConfig:
    """Materialize a legacy ``InferenceReportConfig`` from a JSON deck profile."""
    builder = profile.get("dataset_builder", "sweep")
    if builder in ("series", "matrix"):
        return make_inference_report_config(
            suite_id=profile.get("suite_id") or profile.get("profile_id", "suite"),
            report_basename=profile.get("report_basename") or f"{profile.get('suite_id', 'suite')}_run_deck",
            title=profile.get("title") or "Run Deck",
            subtitle=profile.get("subtitle") or "",
            footer=profile.get("footer") or "",
            link_name=profile.get("link_name") or profile.get("title") or "Run Deck",
            results_columns=(("Collective", None), ("Size", None), ("Bus BW", "bus_bw")),
            metric_units={"bus_bw": "GB/s", "alg_bw": "GB/s"},
            tier_metric_specs=lambda _c, _t: {},
            interactive_viewer=bool(profile.get("interactive_viewer", False)),
        )

    hooks = profile.get("hooks") or {}
    sweep = profile.get("sweep") or {}

    tier_metric_specs = (
        import_callable(hooks.get("tier_metric_specs") or sweep.get("tier_metric_specs_hook", ""))
        if hooks.get("tier_metric_specs") or sweep.get("tier_metric_specs_hook")
        else None
    )

    if tier_metric_specs is None:
        raise ValueError("JSON sweep profile requires hooks.tier_metric_specs")

    metric_units_spec = hooks.get("metric_units") or sweep.get("metric_units_hook")
    if metric_units_spec:
        metric_units = import_object(str(metric_units_spec))
    else:
        metric_units = sweep.get("metric_units") or {}
    if callable(metric_units):
        metric_units = metric_units()

    results_columns = _parse_results_columns(sweep.get("results_table_columns") or profile.get("results_columns") or [])
    chart_series = _parse_chart_series(sweep.get("chart_series") or profile.get("chart_series") or [])
    cell_highlights = _parse_cell_highlights(sweep.get("cell_highlights") or profile.get("cell_highlights") or [])

    run_card_builder = None
    if hooks.get("run_card_display"):
        run_card_builder = import_callable(hooks["run_card_display"])

    launch_builder = None
    if hooks.get("launch_provenance"):
        launch_builder = import_callable(hooks["launch_provenance"])

    lifecycle = profile.get("lifecycle") or {}
    kwargs = {}
    if lifecycle.get("session_labels"):
        kwargs["session_lifecycle_labels"] = tuple(lifecycle["session_labels"])
    if lifecycle.get("cell_labels"):
        kwargs["cell_lifecycle_labels"] = tuple(lifecycle["cell_labels"])

    behavior = profile.get("behavior") or {}
    return make_inference_report_config(
        suite_id=profile.get("suite_id") or profile.get("profile_id", "suite"),
        report_basename=profile.get("report_basename") or f"{profile.get('suite_id', 'suite')}_run_deck",
        title=profile.get("title") or "Run Deck",
        subtitle=profile.get("subtitle") or "",
        footer=profile.get("footer") or "",
        link_name=profile.get("link_name") or profile.get("title") or "Run Deck",
        results_columns=results_columns,
        metric_units=metric_units,
        tier_metric_specs=tier_metric_specs,
        metric_tier_order=tuple(
            sweep.get("tier_order") or profile.get("tier_order") or ("throughput", "health", "record")
        ),
        metric_prefix=sweep.get("metric_prefix") or profile.get("metric_prefix") or "client.",
        cell_highlights=cell_highlights or None,
        chart_series=chart_series or None,
        inference_test_substring=behavior.get("inference_test_substring") or profile.get("inference_test_substring"),
        row_card_extras=behavior.get("row_card_extras", profile.get("row_card_extras", True)),
        row_card_test_names=tuple(
            behavior.get("row_card_test_names")
            or profile.get("row_card_test_names")
            or ("test_metric", "test_cell_metrics")
        ),
        interactive_viewer=profile.get("interactive_viewer", True),
        viewer_cell_threshold=int(profile.get("viewer_cell_threshold", 24)),
        prev_run_json=str(
            profile.get("prev_run", {}).get("json_path", "")
            if isinstance(profile.get("prev_run"), dict)
            else profile.get("prev_run_json", "")
        ),
        run_card_display_builder=run_card_builder,
        launch_provenance_builder=launch_builder,
        sweep_throughput_metric=sweep.get("throughput_metric") or "client.output_throughput",
        sweep_ttft_metric=sweep.get("ttft_metric") or "client.mean_ttft_ms",
        headline_metric=sweep.get("headline_metric") or sweep.get("throughput_metric") or "client.output_throughput",
        **kwargs,
    )


def resolve_report_config(profile: DeckProfile) -> InferenceReportConfig:
    if isinstance(profile, InferenceReportConfig):
        return profile
    if isinstance(profile, dict):
        return build_inference_config_from_profile(profile)
    raise TypeError(f"Unsupported deck profile type: {type(profile).__name__}")
