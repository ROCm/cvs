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


def report_incompatibility(
    report_json,
    *,
    expected_schema_version=1,
    expected_suite_id="",
    expected_metric_contract=None,
):
    """Return why a contract-qualified report cannot be used as a baseline."""
    if expected_metric_contract is None:
        return ""
    if report_json.get("schema_version") != expected_schema_version:
        return f"schema_version {report_json.get('schema_version')!r} is incompatible with {expected_schema_version!r}"
    if report_json.get("suite_id") != expected_suite_id:
        return f"suite_id {report_json.get('suite_id')!r} is incompatible with {expected_suite_id!r}"
    if report_json.get("metric_contract") != expected_metric_contract:
        return (
            f"metric_contract {report_json.get('metric_contract')!r} is incompatible with {expected_metric_contract!r}"
        )
    return ""
