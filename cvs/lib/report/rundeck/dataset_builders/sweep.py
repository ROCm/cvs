'''
Copyright 2025 Advanced Micro Devices Inc.
All rights reserved.

Sweep dataset builder — lifts inference payload assembly into ``datasets.sweep.*``.
'''

from __future__ import annotations

from typing import Any, Mapping

from cvs.lib.report.cell_build import build_all_cells, select_summary_cells
from cvs.lib.report.inference_payload import (
    aggregate_lifecycle,
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


@register_dataset_builder("sweep")
def build_sweep_datasets(sources: dict[str, Any], profile: DeckProfile) -> dict[str, Any]:
    config = resolve_report_config(profile)
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
