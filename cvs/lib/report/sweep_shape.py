'''Shared sweep shape grouping and metric extraction for report payloads.'''

from __future__ import annotations

from typing import Dict, List, Tuple


def shape_label(isl: str, osl: str) -> str:
    return f"ISL={isl} \u00b7 OSL={osl}"


def group_cells_by_shape(cells: List[dict]) -> Dict[Tuple[str, str], List[dict]]:
    groups: Dict[Tuple[str, str], List[dict]] = {}
    for cell in cells:
        groups.setdefault((str(cell["isl"]), str(cell["osl"])), []).append(cell)
    for group in groups.values():
        group.sort(key=lambda c: int(c["concurrency"]))
    return groups


def metric_values_by_concurrency(group_cells: List[dict], metric: str) -> Dict[int, float]:
    values: Dict[int, float] = {}
    for cell in group_cells:
        val = (cell.get("actuals") or {}).get(metric)
        if val is None:
            continue
        try:
            values[int(cell["concurrency"])] = float(val)
        except (TypeError, ValueError):
            continue
    return values
