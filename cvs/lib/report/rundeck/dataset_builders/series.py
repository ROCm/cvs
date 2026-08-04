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
    headers = ["Collective", "Message size", "Bus BW (GB/s)", "Alg BW (GB/s)", "Time (us)"]
    for collective, sizes in sorted((results or {}).items()):
        if not isinstance(sizes, dict):
            continue
        for size_key in sorted(sizes.keys(), key=lambda s: int(s) if str(s).isdigit() else str(s)):
            entry = sizes[size_key]
            if not isinstance(entry, dict):
                continue
            table_rows.append([
                collective,
                size_key,
                entry.get("bus_bw", "—"),
                entry.get("alg_bw", "—"),
                entry.get("time", "—"),
            ])

    return {
        "charts": charts,
        "results_table": {"headers": headers, "rows": table_rows},
        "x_field": series_cfg.get("x_field", "size"),
        "y_fields": y_fields,
    }
