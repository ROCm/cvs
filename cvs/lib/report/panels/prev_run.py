'''Run-to-run comparison panel for inference suite reports (CI / sweep regression).'''

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from cvs.lib.report.compare import build_prev_run_compare_row
from cvs.lib.report.json_io import (
    cell_id_host_key,
    index_cells_by_id_host,
    load_report_json,
    report_incompatibility,
)
from cvs.lib.report.metrics import HEADLINE_THROUGHPUT_METRIC

PREV_RUN_ENV = "CVS_INFERENCE_PREV_REPORT_JSON"
DEFAULT_THRESHOLD_PCT = 5.0
_UNSET = object()


def resolve_prev_run_json_path(
    config_prev_run_json: str = "",
    *,
    report_basename: str = "",
    report_dir: Path | None = None,
) -> str:
    explicit = (config_prev_run_json or os.environ.get(PREV_RUN_ENV, "")).strip()
    if explicit:
        return explicit
    if report_basename and report_dir:
        sibling = Path(report_dir) / f"{report_basename}_prev.json"
        if sibling.is_file():
            return str(sibling)
    return ""


def build_prev_run_panel(
    cells: List[dict],
    baseline_json_path: Path,
    *,
    headline_metric: str = HEADLINE_THROUGHPUT_METRIC,
    threshold_pct: float = DEFAULT_THRESHOLD_PCT,
    expected_schema_version=1,
    expected_suite_id="",
    expected_metric_contract=None,
    baseline_payload=_UNSET,
) -> Optional[dict]:
    if not baseline_json_path.is_file():
        return None
    if baseline_payload is _UNSET:
        baseline_payload = load_report_json(baseline_json_path)
    if not isinstance(baseline_payload, dict):
        baseline_payload = {}
    incompatibility = report_incompatibility(
        baseline_payload,
        expected_schema_version=expected_schema_version,
        expected_suite_id=expected_suite_id,
        expected_metric_contract=expected_metric_contract,
    )
    if incompatibility:
        return {
            "baseline_json": str(baseline_json_path),
            "compatible": False,
            "incompatibility": incompatibility,
            "rows": [],
        }
    baseline = index_cells_by_id_host(baseline_payload)
    if not baseline:
        return None

    rows = []
    for cell in cells:
        prev_cell = baseline.get(cell_id_host_key(cell))
        rows.append(
            build_prev_run_compare_row(
                cell,
                prev_cell,
                headline_metric=headline_metric,
                threshold_pct=threshold_pct,
            )
        )

    return {
        "baseline_json": str(baseline_json_path),
        "headline_metric": headline_metric,
        "threshold_pct": threshold_pct,
        "compatible": True,
        "rows": rows,
    }
