'''Run-to-run comparison panel for inference suite reports (CI / sweep regression).'''

from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Any, List, Mapping, Optional

from cvs.lib.report.compare import metric_delta_pct, metric_ratio, prev_run_change_flags
from cvs.lib.report.formatting import fmt_num
from cvs.lib.report.json_io import cell_id_host_key, index_cells_by_id_host, load_report_json
from cvs.lib.report.metrics import HEADLINE_THROUGHPUT_METRIC

# Backward-compatible alias for callers importing from this module.
THROUGHPUT_METRIC = HEADLINE_THROUGHPUT_METRIC
PREV_RUN_ENV = "CVS_INFERENCE_PREV_REPORT_JSON"
DEFAULT_THRESHOLD_PCT = 5.0


def cell_lookup_key(cell: Mapping[str, Any]) -> tuple[str, str]:
    return cell_id_host_key(cell)


def load_cell_index(path: Path) -> dict[tuple[str, str], dict]:
    data = load_report_json(path)
    if not data:
        return {}
    return index_cells_by_id_host(data)


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
) -> Optional[dict]:
    if not baseline_json_path.is_file():
        return None
    baseline = load_cell_index(baseline_json_path)
    if not baseline:
        return None

    rows = []
    for cell in cells:
        prev_cell = baseline.get(cell_lookup_key(cell))
        actuals = cell.get("actuals") or {}
        prev_actuals = (prev_cell or {}).get("actuals") or {}
        current = actuals.get(headline_metric)
        previous = prev_actuals.get(headline_metric)
        ratio = None
        delta_pct = None
        changed = False
        regression = False
        if current is not None and previous is not None:
            try:
                cur_f = float(current)
                prev_f = float(previous)
                ratio = metric_ratio(cur_f, prev_f)
                delta_pct = metric_delta_pct(cur_f, prev_f)
                changed, regression = prev_run_change_flags(delta_pct, threshold_pct)
            except (TypeError, ValueError):
                pass
        rows.append(
            {
                "cell_id": cell.get("cell_id"),
                "host": cell.get("host"),
                "concurrency": cell.get("concurrency"),
                "current_throughput": current,
                "previous_throughput": previous,
                "compare.prev_run.throughput_ratio": ratio,
                "compare.prev_run.throughput_delta_pct": delta_pct,
                "changed": changed,
                "regression": regression,
            }
        )

    return {
        "baseline_json": str(baseline_json_path),
        "headline_metric": headline_metric,
        "threshold_pct": threshold_pct,
        "rows": rows,
    }


def render_prev_run_panel_html(panel: dict) -> str:
    rows = panel.get("rows") or []
    if not rows:
        return "<p class='muted'>No aligned cells between this run and the baseline JSON.</p>"
    threshold = panel.get("threshold_pct", DEFAULT_THRESHOLD_PCT)
    body = []
    for row in rows:
        delta = row.get("compare.prev_run.throughput_delta_pct")
        delta_s = f"{delta:+.1f}%" if delta is not None else "\u2014"
        row_cls = "fail" if row.get("regression") else ("changed" if row.get("changed") else "")
        body.append(
            f"<tr class='{row_cls}'><td>{html.escape(str(row.get('cell_id', '')))}</td>"
            f"<td>{html.escape(str(row.get('host', '')))}</td>"
            f"<td>{row.get('concurrency', '')}</td>"
            f"<td>{fmt_num(row.get('previous_throughput'))}</td>"
            f"<td>{fmt_num(row.get('current_throughput'))}</td>"
            f"<td>{html.escape(delta_s)}</td></tr>"
        )
    baseline = html.escape(str(panel.get("baseline_json", "")))
    return (
        f"<p class='muted'>Baseline: {baseline} · flag when |delta| &gt; {threshold}%</p>"
        "<table class='results-table'><tr><th>Cell</th><th>Host</th><th>C</th>"
        "<th>Prev tok/s</th><th>Current tok/s</th><th>Delta</th></tr>"
        f"{''.join(body)}</table>"
    )
