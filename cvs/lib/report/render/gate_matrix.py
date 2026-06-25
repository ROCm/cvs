'''Gate matrix table and compact heatmap for static inference reports.'''

from __future__ import annotations

import html
from typing import Iterable, List, Mapping


def render_gate_matrix_html(gate_matrix: List[dict], tier_order: Iterable[str]) -> str:
    tiers = tuple(tier_order)
    if not gate_matrix:
        return "<p class='muted'>No gate matrix (no cells recorded).</p>"
    header = "<tr><th>Cell</th>" + "".join(f"<th>{html.escape(t)}</th>" for t in tiers) + "</tr>"
    rows = "".join(
        f"<tr><td>{html.escape(row['label'])}</td>"
        + "".join(
            f"<td class='matrix-{html.escape(row['tiers'].get(t, 'na'))}'>"
            f"{html.escape(row['tiers'].get(t, 'na'))}</td>"
            for t in tiers
        )
        + "</tr>"
        for row in gate_matrix
    )
    return f"<table class='matrix'>{header}{rows}</table>"


def render_gate_heatmap_html(gate_matrix: List[dict], tier_order: Iterable[str]) -> str:
    """Compact color grid for screenshots (status only, no text labels in cells)."""
    tiers = tuple(tier_order)
    if not gate_matrix or not tiers:
        return ""
    header = "<tr><th>Cell</th>" + "".join(f"<th>{html.escape(t)}</th>" for t in tiers) + "</tr>"
    rows = []
    for row in gate_matrix:
        cells = "".join(
            f"<td class='heat-{html.escape(row['tiers'].get(t, 'na'))}' "
            f"title='{html.escape(t)}: {html.escape(row['tiers'].get(t, 'na'))}'></td>"
            for t in tiers
        )
        rows.append(f"<tr><td class='heat-label'>{html.escape(row['label'])}</td>{cells}</tr>")
    return (
        "<p class='muted heat-caption'>Compact gate heatmap (hover for tier status)</p>"
        f"<table class='matrix heatmap'>{header}{''.join(rows)}</table>"
    )


def gate_heatmap_css() -> str:
    return """
.heat-caption { margin: 0 0 0.5rem; font-size: 0.8rem; }
table.heatmap { margin-top: 0.75rem; }
table.heatmap td.heat-label { text-align: left; font-size: 0.75rem; max-width: 12rem;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
table.heatmap td[class^='heat-'] { width: 2.25rem; min-width: 2.25rem; height: 1.75rem;
  padding: 0; border-radius: 4px; }
.heat-pass { background: rgba(61,214,140,0.55); }
.heat-fail { background: rgba(255,92,106,0.65); }
.heat-record { background: rgba(107,159,255,0.45); }
.heat-na { background: rgba(92,99,112,0.25); }
"""
