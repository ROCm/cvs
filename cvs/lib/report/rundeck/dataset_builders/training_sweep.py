'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Training-sweep dataset builder — Megatron ``train_res_dict`` string combo keys.
'''

from cvs.lib.report.inference_payload import overall_status
from cvs.lib.report.render.gate_matrix import build_gate_matrix_rows
from cvs.lib.report.rundeck.config_adapter import resolve_report_config
from cvs.lib.report.rundeck.dataset_builders.registry import register_dataset_builder
from cvs.lib.report.training_cells import (
    build_training_cells,
    build_training_chart_series,
    build_training_results_table,
    build_training_summaries,
)


def _results_dict(sources):
    return (
        sources.get("results")
        or sources.get("cvs_results_dict")
        or sources.get("train_res_dict")
        or sources.get("inf_res_dict")
        or {}
    )


@register_dataset_builder("training_sweep")
def build_training_sweep_datasets(sources, profile):
    config = resolve_report_config(profile)
    variant_config = sources.get("variant")
    lifecycle_report = sources.get("lifecycle_report") or {}
    if hasattr(sources.get("lifecycle"), "report"):
        lifecycle_report = sources["lifecycle"].report

    train_res_dict = _results_dict(sources)
    enforce = bool(getattr(variant_config, "enforce_thresholds", False))
    cells = build_training_cells(config, variant_config, train_res_dict, lifecycle_report)
    chart_series = build_training_chart_series(config, cells)
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
        "sweep_summaries": build_training_summaries(config, cells),
        "gate_matrix": build_gate_matrix_rows(cells),
        "results_table": build_training_results_table(config, cells),
        "multi_shape_comparison": False,
        "overall_status": overall_status(config, cells, enforce),
        "metric_tier_order": config.metric_tier_order,
        "headline_metric": config.headline_metric,
        "session_lifecycle_labels": config.session_lifecycle_labels,
        "cell_lifecycle_labels": config.cell_lifecycle_labels,
        "enforce": enforce,
        "config": config,
    }
