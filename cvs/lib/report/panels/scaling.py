'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Multi-node scaling panel for inference suite reports (M5).

Independent of inference framework parity (M4). Renders when a run has multiple
hosts or ``nnodes > 1``; optional single-node baseline JSON enables efficiency %.
'''

from __future__ import annotations

import html
from pathlib import Path
from typing import List, Optional

from cvs.lib.report.formatting import fmt_num
from cvs.lib.report.json_io import load_report_json, max_metric_from_cells, sum_metric_from_cells
from cvs.lib.report.metrics import HEADLINE_THROUGHPUT_METRIC

# Backward-compatible alias.
THROUGHPUT_METRIC = HEADLINE_THROUGHPUT_METRIC


def _per_node_rows(cells: List[dict]) -> List[dict]:
    rows = []
    for cell in cells:
        tput = cell.get("actuals", {}).get(HEADLINE_THROUGHPUT_METRIC)
        rows.append(
            {
                "host": cell.get("host"),
                "policy": cell.get("policy"),
                "concurrency": cell.get("concurrency"),
                "isl": cell.get("isl"),
                "osl": cell.get("osl"),
                "throughput": tput,
            }
        )
    return rows


def _baseline_throughput_from_json(path: Path) -> Optional[float]:
    data = load_report_json(path)
    if not data:
        return None
    return max_metric_from_cells(data.get("cells") or [], HEADLINE_THROUGHPUT_METRIC)


def build_scaling_panel(
    *,
    cells: List[dict],
    nnodes: int = 1,
    baseline_json_path: Optional[str] = None,
    headline_metric: str = HEADLINE_THROUGHPUT_METRIC,
) -> Optional[dict]:
    """Build scaling panel payload; returns None when single-node single-host."""
    hosts = {c.get("host") for c in cells if c.get("host")}
    effective_nnodes = max(nnodes, len(hosts))
    if effective_nnodes <= 1:
        return None

    per_node = _per_node_rows(cells)
    cluster_total = sum_metric_from_cells(cells, HEADLINE_THROUGHPUT_METRIC)
    baseline_single = None
    efficiency_pct = None
    if baseline_json_path:
        baseline_single = _baseline_throughput_from_json(Path(baseline_json_path))
        if baseline_single and cluster_total and effective_nnodes > 0:
            expected = baseline_single * effective_nnodes
            if expected > 0:
                efficiency_pct = 100.0 * cluster_total / expected

    return {
        "panel_id": "scaling",
        "title": "Multi-node scaling",
        "nnodes": effective_nnodes,
        "headline_metric": headline_metric,
        "cluster_throughput": cluster_total,
        "baseline_single_node_throughput": baseline_single,
        "compare.scaling.efficiency_pct": efficiency_pct,
        "per_node_rows": per_node,
    }


def render_scaling_panel_html(panel: dict) -> str:
    eff = panel.get("compare.scaling.efficiency_pct")
    eff_html = (
        f"<div class='summary-stat'>{fmt_num(eff, 0)}<span class='headline-unit'>%</span></div>"
        f"<div class='summary-meta'>Scaling efficiency vs single-node baseline</div>"
        if eff is not None
        else "<p class='muted'>No single-node baseline JSON — per-node breakdown only.</p>"
    )
    rows = panel.get("per_node_rows") or []
    table_rows = "".join(
        f"<tr><td>{html.escape(str(r.get('host', '')))}</td>"
        f"<td>{html.escape(str(r.get('policy', '')))}</td>"
        f"<td>{r.get('concurrency', '')}</td>"
        f"<td>{fmt_num(r.get('throughput'))}</td></tr>"
        for r in rows
    )
    table = (
        "<table class='results-table'><tr><th>Host</th><th>Policy</th>"
        "<th>Concurrency</th><th>Throughput</th></tr>"
        f"{table_rows}</table>"
        if table_rows
        else "<p class='muted'>No per-node rows.</p>"
    )
    return (
        f"<div class='summary-grid'><article class='summary-card'>"
        f"<h3>Cluster ({panel.get('nnodes', '?')} nodes)</h3>"
        f"<div class='summary-stat'>{fmt_num(panel.get('cluster_throughput'))}"
        f"<span class='headline-unit'>tok/s aggregate</span></div>"
        f"{eff_html}</article></div>"
        f"<div class='results-wrap'>{table}</div>"
    )
