'''
Copyright 2025 Advanced Micro Devices Inc.
All rights reserved.

Matrix dataset builder for RCCL golden heatmap compare semantics.
'''

from __future__ import annotations

from typing import Any

from cvs.lib.report.rundeck.dataset_builders.registry import register_dataset_builder


def _flatten_matrix(data: dict, *, value_field: str = "bus_bw") -> dict[str, dict[str, float]]:
    """Nested collective→size→metrics to row→col matrix."""
    matrix: dict[str, dict[str, float]] = {}
    for collective, sizes in (data or {}).items():
        if not isinstance(sizes, dict):
            continue
        row = matrix.setdefault(str(collective), {})
        for size_key, entry in sizes.items():
            if isinstance(entry, dict) and entry.get(value_field) is not None:
                try:
                    row[str(size_key)] = float(entry[value_field])
                except (TypeError, ValueError):
                    continue
    return matrix


def _compare_matrices(current: dict, reference: dict) -> list[list[Any]]:
    rows = []
    for row_label in sorted(set(current.keys()) | set(reference.keys())):
        cur_row = current.get(row_label) or {}
        ref_row = reference.get(row_label) or {}
        for col in sorted(set(cur_row.keys()) | set(ref_row.keys()), key=lambda s: int(s) if str(s).isdigit() else str(s)):
            cur = cur_row.get(col)
            ref = ref_row.get(col)
            delta_pct = None
            if cur is not None and ref not in (None, 0):
                try:
                    delta_pct = 100.0 * (float(cur) - float(ref)) / float(ref)
                except (TypeError, ValueError, ZeroDivisionError):
                    delta_pct = None
            rows.append([row_label, col, cur, ref, delta_pct])
    return rows


@register_dataset_builder("matrix")
def build_matrix_datasets(sources: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    current = sources.get("results") or sources.get("cvs_results_dict") or {}
    reference = sources.get("reference") or sources.get("golden") or {}
    matrix_cfg = profile.get("matrix") or {}
    value_field = matrix_cfg.get("value_field", "bus_bw")

    current_matrix = _flatten_matrix(current, value_field=value_field)
    reference_matrix = _flatten_matrix(reference, value_field=value_field)

    return {
        "current": current_matrix,
        "reference": reference_matrix,
        "compare_rows": _compare_matrices(current_matrix, reference_matrix),
        "row_labels": matrix_cfg.get("row_labels") or sorted(current_matrix.keys()),
        "col_labels": matrix_cfg.get("col_labels") or [],
        "value_field": value_field,
    }
