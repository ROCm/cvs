'''
Copyright 2025 Advanced Micro Devices Inc.
All rights reserved.

Build profile-driven interactive viewer configuration for sweep Run Deck payloads.
'''

from __future__ import annotations

from typing import Any, Optional

from cvs.lib.report.types import InferenceReportConfig

# Results-table column labels → cell record fields (sweep cells).
LABEL_TO_FIELD: dict[str, str] = {
    "Model": "model",
    "GPU": "gpu",
    "ISL": "isl",
    "OSL": "osl",
    "Policy": "policy",
    "Conc": "concurrency",
    "Host": "host",
    "Cell": "cell_id",
}

FIELD_LABELS: dict[str, str] = {
    "cell_id": "Cell",
    "model": "Model",
    "gpu": "GPU",
    "isl": "ISL",
    "osl": "OSL",
    "policy": "Policy",
    "concurrency": "C",
    "host": "Host",
}


def _field_label(field: str) -> str:
    return FIELD_LABELS.get(field, field.replace("_", " ").title())


def _metric_meta(config: InferenceReportConfig, full_key: str, label: str, *, invert: bool = False) -> dict[str, Any]:
    short = full_key.split(".", 1)[-1] if "." in full_key else full_key
    unit = config.metric_units.get(short, "")
    higher = not invert
    if "ms" in label.lower() or "latency" in label.lower() or invert:
        higher = False
    if "throughput" in label.lower() or "tok/s" in unit or "req/s" in unit:
        higher = True
    return {"label": label, "unit": unit, "higher_better": higher}


def _metrics_from_config(config: InferenceReportConfig) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for ch in config.chart_series or ():
        full = config.full_metric(ch.metric_suffix)
        metrics[full] = _metric_meta(config, full, ch.title, invert=ch.invert)
    for label, key in config.results_columns:
        if not key or not str(key).startswith("client."):
            continue
        if key not in metrics:
            metrics[key] = _metric_meta(config, key, str(label))
    return metrics


def _table_columns_from_config(config: InferenceReportConfig) -> list[dict[str, Any]]:
    columns: list[dict[str, Any]] = [{"field": "cell_id", "label": "Cell"}]
    for label, key in config.results_columns:
        if key is None:
            field = LABEL_TO_FIELD.get(str(label))
            if field and field != "cell_id":
                columns.append({"field": field, "label": str(label)})
        elif str(key).startswith("client."):
            columns.append({"metric": str(key), "label": str(label)})
    if len(columns) <= 1:
        columns.extend(
            [
                {"field": "isl", "label": "ISL"},
                {"field": "osl", "label": "OSL"},
                {"field": "policy", "label": "Policy"},
                {"field": "host", "label": "Host"},
                {"field": "concurrency", "label": "C"},
            ]
        )
        for suffix, hl_label in (config.cell_highlights or ())[:3]:
            columns.append({"metric": config.full_metric(suffix), "label": str(hl_label)})
    columns.append({"computed": "status", "label": "Status"})
    return columns


def _parse_profile_columns(raw: list) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for entry in raw:
        if isinstance(entry, dict):
            if entry.get("field") or entry.get("metric") or entry.get("computed"):
                out.append(dict(entry))
            continue
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            label, key = entry
            if key is None:
                field = LABEL_TO_FIELD.get(str(label))
                if field:
                    out.append({"field": field, "label": str(label)})
            elif str(key).startswith("client."):
                out.append({"metric": str(key), "label": str(label)})
    return out


def build_viewer_config(
    profile: Optional[dict[str, Any]],
    config: InferenceReportConfig,
) -> dict[str, Any]:
    """Materialize ``viewer_config`` for the interactive sweep viewer."""
    profile = profile or {}
    viewer = profile.get("viewer") or {}
    sweep = profile.get("sweep") or {}

    group_by: list[str] = list(viewer.get("group_by") or sweep.get("group_by") or ("isl", "osl"))
    filter_fields: list[str] = list(viewer.get("filters") or group_by + ["policy", "host"])

    metrics = _metrics_from_config(config)
    extra_metrics = viewer.get("metrics") or {}
    for key, meta in extra_metrics.items():
        if isinstance(meta, dict):
            metrics[str(key)] = {
                "label": meta.get("label", key),
                "unit": meta.get("unit", ""),
                "higher_better": bool(meta.get("higher_better", True)),
            }

    heatmap_metrics: list[str] = list(
        viewer.get("heatmap_metrics") or [config.full_metric(ch.metric_suffix) for ch in (config.chart_series or ())]
    )
    heatmap_metrics = [m for m in heatmap_metrics if m in metrics]

    margin_metrics: list[str] = list(viewer.get("margin_metrics") or [])
    if not margin_metrics:
        candidates = [
            config.headline_metric,
            config.sweep_ttft_metric,
            config.full_metric("mean_tpot_ms"),
        ]
        margin_metrics = [m for m in candidates if m and m in metrics]

    table_columns = viewer.get("table_columns")
    if table_columns:
        parsed = _parse_profile_columns(table_columns) if isinstance(table_columns, list) else []
        table_columns = parsed or _table_columns_from_config(config)
    else:
        raw_cols = sweep.get("results_table_columns") or profile.get("results_columns")
        if raw_cols:
            table_columns = _parse_profile_columns(raw_cols)
            table_columns.append({"computed": "status", "label": "Status"})
        else:
            table_columns = _table_columns_from_config(config)

    interactivity_raw = viewer.get("interactivity") if isinstance(viewer.get("interactivity"), dict) else {}
    tpot_metric = interactivity_raw.get("tpot_metric") or config.full_metric("mean_tpot_ms")
    interactivity = {
        "enabled": bool(interactivity_raw.get("enabled", config.interactive_viewer)),
        "tpot_metric": tpot_metric,
        "title": interactivity_raw.get("title") or "Token Throughput per GPU vs. Interactivity",
        "hint": interactivity_raw.get(
            "hint",
            "Interactivity = 1000 / mean TPOT (ms) (tok/s/user) · Y = total token throughput per GPU",
        ),
    }

    dim_labels = {f: _field_label(f) for f in group_by}
    comparison_hint = viewer.get("comparison_hint") or (
        f"Compare {' / '.join(dim_labels.get(f, f.upper()) for f in group_by)} shapes at each concurrency"
    )

    return {
        "group_by": group_by,
        "group_labels": dim_labels,
        "filters": [{"field": f, "label": _field_label(f)} for f in filter_fields],
        "concurrency_field": viewer.get("concurrency_field", "concurrency"),
        "metrics": metrics,
        "heatmap_metrics": heatmap_metrics,
        "margin_metrics": margin_metrics,
        "default_heatmap_metric": viewer.get("default_heatmap_metric") or config.headline_metric,
        "default_margin_metric": viewer.get("default_margin_metric") or config.headline_metric,
        "table_columns": table_columns,
        "heatmap_row_fields": list(viewer.get("heatmap_row_fields") or group_by + ["policy"]),
        "comparison_hint": comparison_hint,
        "interactivity": interactivity,
    }
