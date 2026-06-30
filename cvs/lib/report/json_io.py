'''JSON load and cell-index helpers for suite report sidecars.'''

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional


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
