'''Run-to-run comparison panel for inference suite reports (CI / sweep regression).'''

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any, List, Mapping, Optional

THROUGHPUT_METRIC = "client.output_throughput"
PREV_RUN_ENV = "CVS_INFERENCE_PREV_REPORT_JSON"
DEFAULT_THRESHOLD_PCT = 5.0


def _fmt_num(value: Any, digits: int = 1) -> str:
    if value is None:
        return "\u2014"
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return html.escape(str(value))


def cell_lookup_key(cell: Mapping[str, Any]) -> tuple[str, str]:
    return (str(cell.get("cell_id", "")), str(cell.get("host", "")))


def load_cell_index(path: Path) -> dict[tuple[str, str], dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    index: dict[tuple[str, str], dict] = {}
    for cell in data.get("cells") or []:
        if isinstance(cell, dict):
            index[cell_lookup_key(cell)] = cell
    return index


def resolve_prev_run_json_path(config_prev_run_json: str = "") -> str:
    return (config_prev_run_json or os.environ.get(PREV_RUN_ENV, "")).strip()


def build_prev_run_panel(
    cells: List[dict],
    baseline_json_path: Path,
    *,
    headline_metric: str = THROUGHPUT_METRIC,
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
                if prev_f != 0:
                    ratio = cur_f / prev_f
                    delta_pct = 100.0 * (cur_f - prev_f) / prev_f
                    changed = abs(delta_pct) > threshold_pct
                    regression = changed and delta_pct < 0
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
            f"<td>{_fmt_num(row.get('previous_throughput'))}</td>"
            f"<td>{_fmt_num(row.get('current_throughput'))}</td>"
            f"<td>{html.escape(delta_s)}</td></tr>"
        )
    baseline = html.escape(str(panel.get("baseline_json", "")))
    return (
        f"<p class='muted'>Baseline: {baseline} · flag when |delta| &gt; {threshold}%</p>"
        "<table class='results-table'><tr><th>Cell</th><th>Host</th><th>C</th>"
        "<th>Prev tok/s</th><th>Current tok/s</th><th>Delta</th></tr>"
        f"{''.join(body)}</table>"
    )
