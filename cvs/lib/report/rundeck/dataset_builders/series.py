'''
Copyright 2025 Advanced Micro Devices Inc.
All rights reserved.

Series dataset builder for RCCL-style X→Y curves (message size vs bandwidth).
'''

from __future__ import annotations

from typing import Any

from cvs.lib.report.rundeck.dataset_builders.registry import register_dataset_builder


def _record_charts(records, series_cfg):
    charts = {}
    for chart_cfg in series_cfg.get("charts") or []:
        chart_id = chart_cfg.get("id")
        if not chart_id:
            continue

        entries = {}
        points_field = chart_cfg.get("points_field")
        if points_field:
            for record in records:
                values = record.get(points_field) or []
                if not isinstance(values, (list, tuple)):
                    continue
                label = str(record.get(chart_cfg.get("label_field", "label")) or "run")
                points = []
                for index, value in enumerate(values, start=int(chart_cfg.get("x_start", 1))):
                    try:
                        points.append((index, float(value)))
                    except (TypeError, ValueError):
                        continue
                if points:
                    entries[label] = [{"label": label, "points": points}]
        else:
            x_field = chart_cfg.get("x_field")
            y_field = chart_cfg.get("y_field")
            label_field = chart_cfg.get("label_field", "label")
            grouped = {}
            for record in records:
                try:
                    point = (float(record.get(x_field)), float(record.get(y_field)))
                except (TypeError, ValueError):
                    continue
                label = str(record.get(label_field) or "run")
                grouped.setdefault(label, []).append(point)
            for label, points in grouped.items():
                entries[label] = [{"label": label, "points": sorted(points)}]
        charts[chart_id] = entries
    return charts


def _record_table(records, series_cfg):
    columns = series_cfg.get("table_columns") or []
    headers = []
    fields = []
    for column in columns:
        if isinstance(column, (list, tuple)) and len(column) == 2:
            headers.append(column[0])
            fields.append(column[1])
        elif isinstance(column, dict):
            headers.append(column.get("label", column.get("field", "")))
            fields.append(column.get("field"))

    rows = []
    for record in records:
        row = []
        for field in fields:
            value = record.get(field) if field else None
            row.append(value if value is not None else "—")
        rows.append(row)
    return {"headers": headers, "rows": rows}


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


@register_dataset_builder("series")
def build_series_datasets(sources: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    results = sources.get("results") or sources.get("cvs_results_dict") or {}
    series_cfg = profile.get("series") or {}
    if series_cfg.get("record_layout"):
        records = [record for record in results if isinstance(record, dict)] if isinstance(results, list) else []
        return {
            "charts": _record_charts(records, series_cfg),
            "results_table": _record_table(records, series_cfg),
            "records": records,
        }

    y_fields = series_cfg.get("y_fields") or ["bus_bw", "alg_bw"]
    if isinstance(y_fields, str):
        y_fields = [y_fields]

    charts: dict[str, dict[str, list[dict]]] = {}
    for y_field in y_fields:
        charts[y_field] = _graph_to_series(results, y_field=y_field)

    table_rows = []
    headers = ["Collective", "Message size", "Bus BW (GB/s)", "Alg BW (GB/s)", "Time (us)"]
    for collective, sizes in sorted((results or {}).items()):
        if not isinstance(sizes, dict):
            continue
        for size_key in sorted(sizes.keys(), key=lambda s: int(s) if str(s).isdigit() else str(s)):
            entry = sizes[size_key]
            if not isinstance(entry, dict):
                continue
            table_rows.append(
                [
                    collective,
                    size_key,
                    entry.get("bus_bw", "—"),
                    entry.get("alg_bw", "—"),
                    entry.get("time", "—"),
                ]
            )

    return {
        "charts": charts,
        "results_table": {"headers": headers, "rows": table_rows},
        "x_field": series_cfg.get("x_field", "size"),
        "y_fields": y_fields,
    }
