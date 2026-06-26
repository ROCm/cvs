'''JSON load and cell-index helpers for suite report sidecars.'''

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


def load_report_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def cell_id_host_key(cell: Mapping[str, Any]) -> tuple[str, str]:
    return (str(cell.get("cell_id", "")), str(cell.get("host", "")))


def index_cells_by_id_host(report_json: Mapping[str, Any]) -> dict[tuple[str, str], dict]:
    index: dict[tuple[str, str], dict] = {}
    for cell in report_json.get("cells") or []:
        if isinstance(cell, dict):
            index[cell_id_host_key(cell)] = cell
    return index


def _metric_values(cells: Iterable[dict], metric: str) -> list[float]:
    values: list[float] = []
    for cell in cells:
        raw = (cell.get("actuals") or {}).get(metric)
        if raw is None:
            continue
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            continue
    return values


def sum_metric_from_cells(cells: Iterable[dict], metric: str) -> Optional[float]:
    values = _metric_values(cells, metric)
    return sum(values) if values else None


def max_metric_from_cells(cells: Iterable[dict], metric: str) -> Optional[float]:
    values = _metric_values(cells, metric)
    return max(values) if values else None


def load_training_nodes(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    nodes = data.get("nodes") or {}
    if nodes:
        return nodes
    rows = data.get("node_rows") or []
    out: dict[str, dict] = {}
    for row in rows:
        node_id = row.get("node") or row.get("host")
        if node_id:
            out[str(node_id)] = row
    return out
