'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

JAX MaxText training sweep normalization for Run Deck.
'''

from cvs.lib.training.jaxmaxtext.utils.maxtext_parsing import TRAINING_METRICS
from cvs.lib.utils.verdict import ThresholdViolation, evaluate_all

_THROUGHPUT_METRICS = (
    "training.tflops_per_sec_per_gpu",
    "training.tokens_per_sec_per_gpu",
    "training.tokens_per_sec_total",
    "training.scaling_efficiency_pct",
)
_QUALITY_METRICS = (
    "training.final_loss",
    "training.loss_decreased",
    "training.eval_loss",
    "training.steps_to_target",
    "training.time_to_target_seconds",
)


def parse_sweep_key(name):
    values = {}
    for token in str(name).split(","):
        key, separator, value = token.partition("=")
        if separator:
            values[key.strip().upper()] = value.strip()
    return {
        "batch_size": values.get("BS", "\u2014"),
        "precision": values.get("PRECISION", "\u2014"),
        "sequence_length": values.get("SL", "\u2014"),
    }


def sweep_label(name):
    dimensions = parse_sweep_key(name)
    if "\u2014" in dimensions.values():
        return str(name)
    return (
        f"BS={dimensions['batch_size']} \u00b7 {dimensions['precision']} "
        f"\u00b7 SL={dimensions['sequence_length']}"
    )


def tier_metric_specs(thresholds_cell, tier):
    metrics = _THROUGHPUT_METRICS if tier == "throughput" else _QUALITY_METRICS if tier == "quality" else ()
    return {
        metric: thresholds_cell[metric]
        for metric in metrics
        if metric in thresholds_cell and thresholds_cell[metric].get("kind") != "info"
    }


def _sweeps(sources):
    results = sources.get("results") or sources.get("cvs_results_dict") or {}
    if not isinstance(results, dict):
        return {}
    sweeps = results.get("sweeps")
    return sweeps if isinstance(sweeps, dict) else {}


def _actuals(record):
    if not isinstance(record, dict):
        return {}
    results = record.get("results")
    return results if isinstance(results, dict) else {}


def _tier_status(actuals, specs, enforce):
    if not specs or not enforce:
        return "record"
    if any(actuals.get(metric) is None for metric in specs):
        return "na"
    try:
        evaluate_all(actuals, specs)
    except ThresholdViolation:
        return "fail"
    return "pass"


def _gate_matrix(sweeps, variant):
    thresholds = getattr(variant, "thresholds", {}) or {}
    if not thresholds:
        return []
    enforce = bool(getattr(variant, "enforce_thresholds", False))
    rows = []
    for name, record in sweeps.items():
        cell_thresholds = thresholds.get(name)
        if not isinstance(cell_thresholds, dict):
            continue
        actuals = _actuals(record)
        dimensions = parse_sweep_key(name)
        rows.append(
            {
                "label": sweep_label(name),
                "cell_id": name,
                "concurrency": dimensions["batch_size"],
                "tiers": {
                    tier: _tier_status(actuals, tier_metric_specs(cell_thresholds, tier), enforce)
                    for tier in ("throughput", "quality")
                },
            }
        )
    return rows


def _throughput_chart(sweeps):
    metric_series = (
        ("training.tokens_per_sec_per_gpu", "Tokens/s/GPU"),
        ("training.tokens_per_sec_total", "Total tokens/s"),
    )
    chart = {}
    for metric, label in metric_series:
        points = []
        for name, record in sweeps.items():
            value = _actuals(record).get(metric)
            if value is not None:
                points.append((sweep_label(name), value))
        if points:
            chart[metric] = [{"label": label, "points": points}]
    return chart


def _results_table(sweeps, variant):
    headers = ["Sweep", "BS", "Precision", "SL", "nnodes"]
    headers.extend(short.replace("_", " ") for short, _unit in TRAINING_METRICS)
    rows = []
    nnodes = getattr(variant, "nnodes", "\u2014")
    for name, record in sweeps.items():
        dimensions = parse_sweep_key(name)
        actuals = _actuals(record)
        row = [
            name,
            dimensions["batch_size"],
            dimensions["precision"],
            dimensions["sequence_length"],
            nnodes,
        ]
        row.extend(actuals.get(f"training.{short}", "\u2014") for short, _unit in TRAINING_METRICS)
        rows.append(row)
    return {"headers": headers, "rows": rows}


def _overall_status(gate_matrix, enforce):
    if not gate_matrix or not enforce:
        return "record"
    statuses = [status for row in gate_matrix for status in row["tiers"].values()]
    if "fail" in statuses:
        return "fail"
    return "pass" if "pass" in statuses else "na"


def build_jaxmaxtext_datasets(sources, profile):
    sweeps = _sweeps(sources)
    variant = sources.get("variant")
    gate_matrix = _gate_matrix(sweeps, variant)
    return {
        "charts": {"throughput": _throughput_chart(sweeps)},
        "gate_matrix": gate_matrix,
        "results_table": _results_table(sweeps, variant),
        "overall_status": _overall_status(
            gate_matrix,
            bool(getattr(variant, "enforce_thresholds", False)),
        ),
        "all_cells": [],
        "cells": [],
        "metric_tier_order": tuple((profile.get("sweep") or {}).get("tier_order") or ("throughput", "quality")),
    }
