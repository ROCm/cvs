'''Shared sweep shape grouping and metric extraction for report payloads.'''

from __future__ import annotations

from typing import Dict, List, Tuple


def shape_label(isl: str, osl: str) -> str:
    return f"ISL={isl} \u00b7 OSL={osl}"


def _sweep_sort_key(cell):
    if cell.get("named_cell"):
        return 1, str(cell.get("sweep_label", cell["concurrency"]))
    return 0, int(cell["concurrency"])


def group_cells_by_shape(cells: List[dict]) -> Dict[Tuple[str, str], List[dict]]:
    groups: Dict[Tuple[str, str], List[dict]] = {}
    for cell in cells:
        shape = ("", "") if cell.get("named_cell") else (str(cell["isl"]), str(cell["osl"]))
        groups.setdefault(shape, []).append(cell)
    for group in groups.values():
        group.sort(key=_sweep_sort_key)
    return groups


def metric_values_by_concurrency(group_cells: List[dict], metric: str) -> Dict[int, float]:
    values: Dict[int, float] = {}
    for cell in group_cells:
        val = (cell.get("actuals") or {}).get(metric)
        if val is None:
            continue
        try:
            axis_value = cell.get("sweep_label", cell["concurrency"])
            if not cell.get("named_cell"):
                axis_value = int(axis_value)
            values[axis_value] = float(val)
        except (TypeError, ValueError):
            continue
    return values
