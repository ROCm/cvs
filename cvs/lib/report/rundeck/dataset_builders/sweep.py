'''
Copyright 2025 Advanced Micro Devices Inc.
All rights reserved.

Sweep dataset builder — lifts inference payload assembly into ``datasets.sweep.*``.
'''

from __future__ import annotations

import math
from typing import Any, Mapping

from cvs.lib.report.cell_build import (
    CellRecordBuilder,
    bar_pct,
    build_all_cells,
    margin_text,
    metric_pass,
    select_summary_cells,
)
from cvs.lib.report.inference_payload import (
    build_chart_series,
    build_results_table,
    build_sweep_summaries,
    overall_status,
    sweep_has_multi_shape_comparison,
)
from cvs.lib.report.profile import DeckProfile
from cvs.lib.report.render.gate_matrix import build_gate_matrix_rows
from cvs.lib.report.rundeck.config_adapter import resolve_report_config
from cvs.lib.report.rundeck.dataset_builders.registry import register_dataset_builder
from cvs.lib.report.types import InferenceReportConfig


def _results_dict(sources: Mapping[str, Any]) -> Mapping:
    return sources.get("results") or sources.get("cvs_results_dict") or sources.get("inf_res_dict") or {}


def _sweep_layout_cfg(profile):
    if isinstance(profile, dict):
        return profile.get("sweep") or {}
    return {}


def _metric_samples(value):
    items = list(value) if isinstance(value, (list, tuple)) else [value]
    samples = []
    for item in items:
        if item is None or item == "":
            continue
        try:
            samples.append(float(item))
        except (TypeError, ValueError):
            continue
    return samples


def _reduce_actual(value, spec):
    samples = _metric_samples(value)
    if not samples:
        while isinstance(value, (list, tuple)):
            if not value:
                return None
            value = value[-1]
        return value
    kind = str((spec or {}).get("kind") or "")
    if kind.startswith("min"):
        return min(samples)
    if kind.startswith("max"):
        return max(samples)
    return samples[-1]


def _safe_bar_pct(config, actual, spec):
    if spec is None or actual is None or "value" not in spec:
        return None
    if config.metric_verdict is not None and not (type(actual) in (int, float) and math.isfinite(actual)):
        return None
    try:
        return bar_pct(float(actual), spec)
    except (TypeError, ValueError, KeyError):
        return None


def _named_cell_dimensions(cell_id, variant_config, sweep_cfg):
    dimension_keys = list(sweep_cfg.get("dimension_keys") or [])
    aliases = sweep_cfg.get("label_aliases") or {}
    alias_to_key = {str(key).lower(): key for key in dimension_keys}
    alias_to_key.update({str(alias).lower(): key for key, alias in aliases.items()})

    dimensions = {}
    if cell_id != "default":
        for part in str(cell_id).split(","):
            if "=" not in part:
                continue
            name, value = part.split("=", 1)
            key = alias_to_key.get(name.strip().lower(), name.strip().lower())
            dimensions[key] = value.strip()

    combinations = getattr(getattr(variant_config, "sweep", None), "combinations", {}) or {}
    combo = combinations.get(cell_id)
    for name in dimension_keys:
        value = getattr(combo, name, None)
        if value is not None:
            dimensions.setdefault(name, str(value))
        dimensions.setdefault(name, "—")
    return dimensions


def _display_label(cell_id, dimensions, sweep_cfg):
    keys = list(sweep_cfg.get("label_parts") or sweep_cfg.get("dimension_keys") or [])
    if not keys:
        return str(cell_id)
    aliases = sweep_cfg.get("label_aliases") or {}
    return " · ".join(f"{aliases.get(key, key)}={dimensions.get(key, '—')}" for key in keys)


def _variant_metadata(variant_config):
    train_params = getattr(variant_config, "train_params", {}) or {}
    container = getattr(variant_config, "container", None)
    env = getattr(container, "env", {}) or {}
    return {
        "model": train_params.get("model_name") or train_params.get("model") or "—",
        "gpu": getattr(variant_config, "gpu_arch", "—"),
        "nnodes": str(env.get("NNODES") or train_params.get("nnodes") or "—"),
    }


def _named_cell_metrics(config, actuals, thresholds_cell, enforce):
    metrics = []
    for short, label in config.cell_highlights:
        full = config.full_metric(short)
        actual = actuals.get(full)
        spec = thresholds_cell.get(full)
        status = metric_pass(full, actual, spec, evaluator=config.metric_verdict) if enforce and spec else "record"
        metrics.append(
            {
                "label": label,
                "metric": full,
                "actual": actual,
                "unit": config.metric_units.get(short, ""),
                "spec": spec,
                "status": status,
                "bar_pct": _safe_bar_pct(config, actual, spec),
                "margin": margin_text(actual, spec) if spec else None,
            }
        )
    return metrics


def _extra_metric_label(config, metric):
    prefix = config.metric_prefix or ""
    bare = metric[len(prefix) :] if prefix and str(metric).startswith(prefix) else str(metric)
    return bare.replace("_", " ").title()


def _named_results_table(config, cells):
    columns = list(config.results_columns)
    known = {key for _label, key in columns if key}
    id_label = next((label for label, key in columns if key is None), "Cell")
    for cell in cells:
        for metric in cell["actuals"]:
            if metric not in known:
                columns.append((_extra_metric_label(config, metric), metric))
                known.add(metric)

    headers = [label for label, _key in columns]
    rows = []
    for cell in cells:
        row = []
        dimensions = cell["dimensions"]
        for label, key in columns:
            if key is None:
                value = cell["cell_id"] if label == id_label else "—"
            elif key in cell:
                value = cell[key]
            elif key in dimensions:
                value = dimensions[key]
            else:
                value = cell["actuals"].get(key, "—")
            row.append("—" if value is None else value)
        rows.append(row)
    return {"headers": headers, "rows": rows}


def _named_charts(config, cells):
    charts = {}
    chart_series = {}
    for chart in config.chart_series:
        full = config.full_metric(chart.metric_suffix)
        points = []
        for cell in cells:
            value = cell["actuals"].get(full)
            if value is None:
                continue
            try:
                points.append((cell["display_label"], float(value)))
            except (TypeError, ValueError):
                continue
        if points:
            entry = {"label": f"{chart.title} vs cells", "points": points}
            charts[chart.metric_suffix] = {"cells": [entry]}
            chart_series[chart.metric_suffix] = [entry]
    return charts, chart_series


def _build_named_cell_datasets(sources, config, profile):
    sweep_cfg = _sweep_layout_cfg(profile)
    variant_config = sources.get("variant")
    results = _results_dict(sources)
    enforce = bool(getattr(variant_config, "enforce_thresholds", False)) if variant_config is not None else False
    thresholds = getattr(variant_config, "thresholds", {}) or {} if variant_config is not None else {}
    metadata = _variant_metadata(variant_config)
    tier_builder = CellRecordBuilder(config)
    cells = []

    for cell_id, raw_metrics in results.items():
        crashed = raw_metrics is None
        if crashed:
            raw_metrics = {}
        elif not isinstance(raw_metrics, dict):
            continue
        thresholds_cell = thresholds.get(cell_id) or {}
        actuals = {}
        for metric, value in raw_metrics.items():
            if str(metric).startswith("_"):
                continue
            full = str(metric) if "." in str(metric) else config.full_metric(str(metric))
            actuals[full] = _reduce_actual(value, thresholds_cell.get(full))
        dimensions = _named_cell_dimensions(cell_id, variant_config, sweep_cfg)
        tiers = {
            tier: "na" if crashed else tier_builder.tier_status(actuals, thresholds_cell, tier, enforce)
            for tier in config.metric_tier_order
        }
        cells.append(
            {
                **metadata,
                **dimensions,
                "cell_id": str(cell_id),
                "display_label": _display_label(cell_id, dimensions, sweep_cfg),
                "dimensions": dimensions,
                "metrics": _named_cell_metrics(config, actuals, thresholds_cell, enforce),
                "tiers": tiers,
                "actuals": actuals,
                "cell_lifecycle": {},
                "pytest_inference_nodeid": "",
                "pytest_metrics_nodeid": "",
            }
        )

    charts, chart_series = _named_charts(config, cells)
    chart_config = [
        {
            "suffix": chart.metric_suffix,
            "title": chart.title,
            "unit": chart.unit,
            "metric": config.full_metric(chart.metric_suffix),
            "invert": chart.invert,
        }
        for chart in config.chart_series
    ]
    return {
        "cells": cells,
        "all_cells": cells,
        "charts": charts,
        "chart_series": chart_series,
        "chart_config": chart_config,
        "sweep_summaries": [],
        "gate_matrix": build_gate_matrix_rows(cells),
        "results_table": _named_results_table(config, cells),
        "multi_shape_comparison": False,
        "overall_status": overall_status(config, cells, enforce),
        "metric_tier_order": config.metric_tier_order,
        "headline_metric": config.headline_metric,
        "session_lifecycle_labels": config.session_lifecycle_labels,
        "cell_lifecycle_labels": config.cell_lifecycle_labels,
        "enforce": enforce,
        "config": config,
    }


@register_dataset_builder("sweep")
def build_sweep_datasets(sources: dict[str, Any], profile: DeckProfile) -> dict[str, Any]:
    config = resolve_report_config(profile)
    if isinstance(profile, dict) and (profile.get("sweep") or {}).get("layout") == "named_cells":
        return _build_named_cell_datasets(sources, config, profile)

    variant_config = sources.get("variant")
    lifecycle_report = sources.get("lifecycle_report") or {}
    if hasattr(sources.get("lifecycle"), "report"):
        lifecycle_report = sources["lifecycle"].report

    inf_res_dict = _results_dict(sources)
    enforce = bool(getattr(variant_config, "enforce_thresholds", False))
    cells = build_all_cells(
        config,
        variant_config=variant_config,
        inf_res_dict=inf_res_dict,
        lifecycle_report=lifecycle_report,
    )

    chart_series = build_chart_series(config, cells)
    chart_config = [
        {
            "suffix": ch.metric_suffix,
            "title": ch.title,
            "unit": ch.unit,
            "metric": config.full_metric(ch.metric_suffix),
            "invert": ch.invert,
        }
        for ch in config.chart_series
    ]

    return {
        "cells": cells,
        "all_cells": cells,
        "chart_series": chart_series,
        "chart_config": chart_config,
        "sweep_summaries": build_sweep_summaries(config, cells),
        "gate_matrix": build_gate_matrix_rows(cells),
        "results_table": build_results_table(config, inf_res_dict),
        "multi_shape_comparison": sweep_has_multi_shape_comparison(cells),
        "overall_status": overall_status(config, cells, enforce),
        "metric_tier_order": config.metric_tier_order,
        "headline_metric": config.headline_metric,
        "session_lifecycle_labels": config.session_lifecycle_labels,
        "cell_lifecycle_labels": config.cell_lifecycle_labels,
        "enforce": enforce,
        "config": config,
    }


def select_inline_cells(
    config: InferenceReportConfig,
    cells: list[dict],
    *,
    mode: str,
    inline_limit: int,
) -> list[dict]:
    if mode != "truncated":
        return cells
    return select_summary_cells(
        cells,
        inline_limit,
        gated_tiers=config.gated_tiers,
    )
