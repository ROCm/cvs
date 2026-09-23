'''
Copyright 2025 Advanced Micro Devices Inc.
All rights reserved.

Series dataset builder for RCCL-style X→Y curves (message size vs bandwidth).
'''

from __future__ import annotations

from typing import Any

from cvs.lib.report.rundeck.dataset_builders.registry import register_dataset_builder


def _graph_to_series(graph_dict: dict, *, y_field: str = "bus_bw") -> dict[str, list[dict]]:
    """Convert RCCL ``convert_to_graph_dict`` output to chart series."""
    series_by_name: dict[str, list[dict]] = {}
    for collective, sizes in (graph_dict or {}).items():
        if not isinstance(sizes, dict):
            continue
        points = []
        for size_key in sorted(sizes.keys(), key=lambda s: int(s) if str(s).isdigit() else str(s)):
            entry = sizes[size_key]
            if not isinstance(entry, dict):
                continue
            y_val = entry.get(y_field)
            if y_val is None:
                continue
            try:
                points.append((int(size_key), float(y_val)))
            except (TypeError, ValueError):
                continue
        if points:
            series_by_name[str(collective)] = [{"label": str(collective), "points": points}]
    return series_by_name


def _table_columns(series_cfg):
    raw = series_cfg.get("table_columns") or []
    columns = []
    for item in raw:
        if isinstance(item, dict) and item.get("label") and item.get("field"):
            columns.append((item["label"], item["field"]))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            columns.append((item[0], item[1]))
    if columns:
        return columns
    return [
        ("Collective", "$series"),
        ("Message size", "$x"),
        ("Bus BW (GB/s)", "bus_bw"),
        ("Alg BW (GB/s)", "alg_bw"),
        ("Time (us)", "time"),
    ]


def _table_value(field, series_name, x_value, entry):
    if field == "$series":
        return series_name
    if field == "$x":
        return x_value
    return entry.get(field, "—")


@register_dataset_builder("series")
def build_series_datasets(sources: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    results = sources.get("results") or sources.get("cvs_results_dict") or {}
    series_cfg = profile.get("series") or {}
    y_fields = series_cfg.get("y_fields") or ["bus_bw", "alg_bw"]
    if isinstance(y_fields, str):
        y_fields = [y_fields]

    charts: dict[str, dict[str, list[dict]]] = {}
    for y_field in y_fields:
        charts[y_field] = _graph_to_series(results, y_field=y_field)

    table_rows = []
    table_columns = _table_columns(series_cfg)
    headers = [label for label, _field in table_columns]
    for collective, sizes in sorted((results or {}).items()):
        if not isinstance(sizes, dict):
            continue
        for size_key in sorted(sizes.keys(), key=lambda s: int(s) if str(s).isdigit() else str(s)):
            entry = sizes[size_key]
            if not isinstance(entry, dict):
                continue
            table_rows.append([_table_value(field, collective, size_key, entry) for _label, field in table_columns])

    return {
        "charts": charts,
        "results_table": {"headers": headers, "rows": table_rows},
        "x_field": series_cfg.get("x_field", "size"),
        "y_fields": y_fields,
    }
