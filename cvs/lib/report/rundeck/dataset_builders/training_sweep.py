'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Training sweep dataset builder for the Run Deck.

The inference ``sweep`` builder is shaped for benchmark topology: a 6-tuple
``(model, gpu, isl, osl, policy, conc)`` cell key charted vs concurrency. Training
runs have a different shape -- one result point per sweep cell (BS/PRECISION/SL),
convergence/stability/throughput metrics, and step-series (loss vs step) rather
than metric-vs-concurrency curves. This builder consumes ``training_res_dict``
natively and emits the SAME render contract (cells, gate_matrix, results_table,
overall_status, ...) so the generic cards render it without touching inference code.

``training_res_dict`` shape (filled by cvs/tests/training/jaxmaxtext/_common.py)::

    {"mode": "single|distributed",
     "sweeps": {"<sweep_name>": {"results": {"training.<m>": v}, "num_nodes": N,
                                 "step_metrics": [...], "eval_metrics": [...]}}}

Thresholds are keyed by the sweep name; metric keys are ``training.<short>``.
'''

from __future__ import annotations

from typing import Any, Mapping

from cvs.lib.report.cell_build import metric_pass
from cvs.lib.report.profile import DeckProfile
from cvs.lib.report.rundeck.config_adapter import resolve_report_config
from cvs.lib.report.rundeck.dataset_builders.registry import register_dataset_builder

_MISSING = "\u2014"


def _results_dict(sources: Mapping[str, Any]) -> Mapping:
    return sources.get("results") or sources.get("cvs_results_dict") or sources.get("inf_res_dict") or {}


def _tier_status(config, actuals, thresholds_cell, tier, enforce):
    """Tier verdict for one sweep cell (mirrors CellRecordBuilder.tier_status)."""
    if not enforce or tier == config.record_tier:
        return "record"
    specs = config.tier_metric_specs(thresholds_cell, tier)
    if not specs:
        return "na"
    saw_pass = False
    for metric, spec in specs.items():
        status = metric_pass(metric, actuals.get(metric), spec, evaluator=config.metric_verdict)
        if status == "fail":
            return "fail"
        if status == "na":
            return "na"
        saw_pass = True
    return "pass" if saw_pass else "na"


def _cell_metrics(config, actuals, thresholds_cell):
    metrics = []
    for short, label in config.cell_highlights or ():
        full = config.full_metric(short)
        metrics.append(
            {
                "label": label,
                "metric": full,
                "actual": actuals.get(full),
                "unit": config.metric_units.get(short, ""),
                "spec": thresholds_cell.get(full),
                "status": "record",
            }
        )
    return metrics


def _series_for_sweep(rec):
    """JSON-safe ``{tag: [[step, value], ...]}`` from a sweep's collected tb_scalars.

    Feeds the viewer's dynamic (Chart.js) per-tag line charts -- no PNGs.
    """
    out = {}
    for tag, points in (rec.get("tb_scalars") or {}).items():
        pts = [[s, v] for s, v in (points or []) if v is not None]
        if pts:
            out[tag] = pts
    return out


def _metric_bars(config, cells):
    """Per-metric bar data across sweeps for the deck's dynamic bar charts.

    Returns ``[{metric, label, unit, values: {sweep_id: number}}]`` in metric-units
    order; a metric appears only if at least one sweep produced a numeric value.
    """
    bars = []
    for short, unit in (config.metric_units or {}).items():
        if unit == "bool":
            continue  # 0/1 flags (e.g. loss_decreased) are not meaningful as bars
        full = config.full_metric(short)
        values = {}
        for cell in cells:
            value = cell["actuals"].get(full)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values[cell["cell_id"]] = value
        if values:
            bars.append({"metric": full, "label": short, "unit": unit, "values": values})
    return bars


def _overall_status(config, cells, enforce):
    if not cells:
        return "na"
    if not enforce:
        return "record"
    saw_pass = False
    for cell in cells:
        for tier in config.gated_tiers:
            status = cell["tiers"].get(tier)
            if status == "fail":
                return "fail"
            if status == "pass":
                saw_pass = True
    return "pass" if saw_pass else "na"


def _fixed_values(sweep_name, rec, variant_config):
    """Values for the null-key (non-metric) results columns, in column order.

    Convention for training decks: the leading null columns are the sweep label
    then the node count. Extra null columns render as em-dash.
    """
    model = getattr(getattr(variant_config, "model", None), "id", "") or ""
    return [sweep_name, model, rec.get("num_nodes") if rec.get("num_nodes") is not None else _MISSING]


def _results_table(config, sweeps, variant_config):
    headers = [label for label, _key in config.results_columns]
    rows = []
    for sweep_name in sorted(sweeps):
        rec = sweeps[sweep_name] or {}
        results = rec.get("results") or {}
        fixed = _fixed_values(sweep_name, rec, variant_config)
        fi = 0
        row = []
        for _label, key in config.results_columns:
            if key is None:
                row.append(fixed[fi] if fi < len(fixed) else _MISSING)
                fi += 1
            else:
                value = results.get(key)
                row.append(value if value is not None else _MISSING)
        rows.append(row)
    return {"headers": headers, "rows": rows}


@register_dataset_builder("training_sweep")
def build_training_datasets(sources: dict[str, Any], profile: DeckProfile) -> dict[str, Any]:
    config = resolve_report_config(profile)
    variant_config = sources.get("variant")
    res = _results_dict(sources)
    sweeps = (res or {}).get("sweeps") or {}
    thresholds = getattr(variant_config, "thresholds", {}) or {}
    enforce = bool(getattr(variant_config, "enforce_thresholds", False))
    gpu = getattr(variant_config, "gpu_arch", "") or ""
    model = getattr(getattr(variant_config, "model", None), "id", "") or ""
    tier_order = config.metric_tier_order

    cells = []
    gate_matrix = []
    training_series = {}
    for sweep_name in sorted(sweeps):
        rec = sweeps[sweep_name] or {}
        results = rec.get("results") or {}
        thresholds_cell = thresholds.get(sweep_name) or {}
        nodes = rec.get("num_nodes")
        tiers = {tier: _tier_status(config, results, thresholds_cell, tier, enforce) for tier in tier_order}
        series = _series_for_sweep(rec)
        if series:
            training_series[sweep_name] = series
        cells.append(
            {
                "model": model,
                "gpu": gpu,
                "isl": "",
                "osl": "",
                "policy": sweep_name,
                "concurrency": nodes if nodes is not None else "",
                "host": "",
                "show_host_in_label": False,
                "cell_id": sweep_name,
                "metrics": _cell_metrics(config, results, thresholds_cell),
                "tiers": tiers,
                "actuals": dict(results),
                "cell_lifecycle": {},
            }
        )
        gate_matrix.append(
            {
                "label": sweep_name,
                "cell_id": sweep_name,
                "concurrency": nodes if nodes is not None else "",
                "tiers": tiers,
            }
        )

    return {
        "cells": cells,
        "all_cells": cells,
        "chart_series": {},
        "chart_config": [],
        "sweep_summaries": [],
        "gate_matrix": gate_matrix,
        "results_table": _results_table(config, sweeps, variant_config),
        "training_series": training_series,
        "metric_bars": _metric_bars(config, cells),
        "multi_shape_comparison": False,
        "overall_status": _overall_status(config, cells, enforce),
        "metric_tier_order": tier_order,
        "headline_metric": config.headline_metric,
        "session_lifecycle_labels": config.session_lifecycle_labels,
        "cell_lifecycle_labels": config.cell_lifecycle_labels,
        "enforce": enforce,
        "config": config,
    }
