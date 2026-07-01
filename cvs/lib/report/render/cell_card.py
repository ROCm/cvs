'''Per-cell card HTML used in suite reports and pytest-html row extras.'''

from __future__ import annotations

import html
from typing import Mapping, Optional

from cvs.lib.report.formatting import fmt_num, pytest_row_link_html

def cell_card_css(*, compact: bool = False) -> str:
    base = """
.cell-card { background: #1a1d27; border: 1px solid #2a2f3d; border-radius: 12px;
  padding: 1.25rem; display: flex; flex-direction: column; gap: 0.75rem;
  font-family: "Segoe UI", system-ui, sans-serif; color: #e8eaef; }
.cell-card-compact { padding: 0.85rem 1rem; gap: 0.5rem; font-size: 0.85rem; }
.headline { font-size: 2.25rem; font-weight: 700; color: #ff6b35; line-height: 1; }
.cell-card-compact .headline { font-size: 1.5rem; }
.headline-unit { font-size: 0.9rem; color: #9aa3b5; margin-left: 0.35rem; }
.cell-sub { font-size: 0.8rem; color: #9aa3b5; }
.cell-mini-tl { display: flex; gap: 4px; min-height: 36px; border-radius: 6px; overflow: hidden; font-size: 0.65rem; }
.cell-mini-seg { display: flex; flex-direction: column; justify-content: center; align-items: center;
  padding: 0.25rem; background: rgba(255,255,255,0.05); min-width: 40px; }
.tl-lbl { font-size: 0.65rem; color: #9aa3b5; text-align: center; }
.tl-val { font-size: 0.8rem; font-weight: 600; color: #ff6b35; }
.chip { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; padding: 0.2rem 0.5rem;
  border-radius: 999px; display: inline-block; margin-right: 0.25rem; }
.chip-pass { background: rgba(61,214,140,0.15); color: #3dd68c; }
.chip-fail { background: rgba(255,92,106,0.15); color: #ff5c6a; }
.chip-record { background: rgba(107,159,255,0.12); color: #6b9fff; }
.chip-na { background: rgba(92,99,112,0.2); color: #5c6370; }
.bar-track { height: 4px; background: #2a2f3d; border-radius: 2px; margin-top: 0.35rem; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 2px; }
.bar-pass { background: #3dd68c; } .bar-fail { background: #ff5c6a; } .bar-record { background: #6b9fff; }
.target, .metric-label, .cell-foot { font-size: 0.7rem; color: #9aa3b5; }
.metric-row-highlight { outline: 1px solid #6b9fff; border-radius: 6px; padding: 0.25rem; }
.margin { font-size: 0.7rem; color: #3dd68c; display: block; }
.margin-fail { color: #ff5c6a; }
.headline-margin { font-size: 0.85rem; color: #3dd68c; margin-top: 0.15rem; }
.headline-margin-fail { color: #ff5c6a; }
.metric-val { font-weight: 600; }
.metric-margin-col { font-size: 0.75rem; color: #9aa3b5; min-width: 5rem; text-align: right; }
.metric-row-grid { display: grid; grid-template-columns: 1fr auto auto; gap: 0.35rem 0.75rem; align-items: baseline; }
"""
    return base


def cell_card_report_css() -> str:
    """Cell card rules using report theme CSS variables (static inference HTML)."""
    return """
.cells { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; }
.cell-card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
  padding: 1.25rem; display: flex; flex-direction: column; gap: 0.75rem; }
.headline { font-size: 2.25rem; font-weight: 700; color: var(--accent); line-height: 1; }
.headline-unit { font-size: 0.9rem; color: var(--muted); margin-left: 0.35rem; }
.cell-sub { font-size: 0.8rem; color: var(--muted); }
.cell-mini-tl { display: flex; gap: 4px; min-height: 36px; border-radius: 6px; overflow: hidden; font-size: 0.65rem; }
.cell-mini-seg { display: flex; flex-direction: column; justify-content: center; align-items: center;
  padding: 0.25rem; background: rgba(255,255,255,0.05); min-width: 40px; }
.chip { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; padding: 0.2rem 0.5rem; border-radius: 999px; }
.chip-pass { background: rgba(61,214,140,0.15); color: var(--pass); }
.chip-fail { background: rgba(255,92,106,0.15); color: var(--fail); }
.chip-record { background: rgba(107,159,255,0.12); color: var(--record); }
.chip-na { background: rgba(92,99,112,0.2); color: var(--na); }
.bar-track { height: 4px; background: var(--border); border-radius: 2px; margin-top: 0.35rem; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 2px; }
.bar-pass { background: var(--pass); } .bar-fail { background: var(--fail); } .bar-record { background: var(--record); }
.target, .metric-label, .cell-foot { font-size: 0.7rem; color: var(--muted); }
.margin { font-size: 0.7rem; color: var(--pass); display: block; } .margin-fail { color: var(--fail); }
"""


def render_cell_lifecycle_html(
    cell_lifecycle: Mapping[str, float],
    labels: tuple[str, ...],
) -> str:
    if not cell_lifecycle:
        return ""
    total = sum(cell_lifecycle.values()) or 1.0
    parts = []
    for lbl in labels:
        sec = cell_lifecycle.get(lbl, 0.0)
        if sec <= 0:
            continue
        pct = 100.0 * sec / total
        parts.append(
            f"<div class='cell-mini-seg' style='flex-grow:{pct:.2f}'>"
            f"<span class='tl-lbl'>{html.escape(lbl.replace('_', ' '))}</span>"
            f"<span class='tl-val'>{sec:.1f}s</span></div>"
        )
    if not parts:
        return ""
    return f"<div class='cell-mini-tl'>{''.join(parts)}</div>"


def _tier_chip(status: str, label: str) -> str:
    return f'<span class="chip chip-{html.escape(status)}">{html.escape(label)}</span>'


def render_cell_card_html(
    cell: dict,
    *,
    tier_order: tuple[str, ...],
    headline_metric: str,
    enforce: bool,
    cell_lifecycle_labels: tuple[str, ...],
    compact: bool = False,
    highlight_metric: Optional[str] = None,
    pytest_html_basename: Optional[str] = None,
) -> str:
    tier_chips = "".join(
        _tier_chip(cell["tiers"].get(t, "na"), t) for t in tier_order
    )
    metric_rows = []
    for m in cell["metrics"]:
        if m["actual"] is None:
            continue
        row_cls = "metric-row"
        if highlight_metric and m["metric"] == highlight_metric:
            row_cls += " metric-row-highlight"
        bar = ""
        if m["bar_pct"] is not None:
            bar = (
                f"<div class='bar-track'><div class='bar-fill bar-{m['status']}' "
                f"style='width:{m['bar_pct']:.0f}%'></div></div>"
            )
        target = ""
        if m["spec"] is not None:
            gate_label = "gate" if enforce else "floor"
            target = f"<span class='target'>{gate_label} {fmt_num(m['spec'].get('value'))}</span>"
        margin = ""
        margin_col = ""
        if m.get("margin"):
            cls = "margin-fail" if m["status"] == "fail" else "margin"
            margin = f"<span class='{cls}'>{html.escape(m['margin'])}</span>"
            margin_col = f"<span class='metric-margin-col {cls}'>{html.escape(m['margin'])}</span>"
        na_margin = "<span class='metric-margin-col'>\u2014</span>"
        if compact:
            metric_rows.append(
                f"<div class='{row_cls}'><div class='metric-label'>{html.escape(m['label'])}</div>"
                f"<div class='metric-val'>{fmt_num(m['actual'])} {html.escape(m['unit'])}</div>"
                f"{bar}{target}{margin}</div>"
            )
        else:
            metric_rows.append(
                f"<div class='{row_cls} metric-row-grid'>"
                f"<div class='metric-label'>{html.escape(m['label'])}</div>"
                f"<div class='metric-val'>{fmt_num(m['actual'])} {html.escape(m['unit'])}</div>"
                f"{margin_col or na_margin}"
                f"<div class='metric-extra' style='grid-column:1/-1'>{bar}{target}</div></div>"
            )

    headline = next((m for m in cell["metrics"] if m["metric"] == headline_metric), None)
    headline_val = fmt_num(headline["actual"]) if headline else "\u2014"
    headline_margin_html = ""
    if headline and headline.get("margin"):
        hm_cls = "headline-margin-fail" if headline.get("status") == "fail" else "headline-margin"
        headline_margin_html = (
            f"<div class='{hm_cls}'>{html.escape(headline['margin'])}</div>"
        )
    mini_tl = render_cell_lifecycle_html(cell.get("cell_lifecycle") or {}, cell_lifecycle_labels)
    card_cls = "cell-card cell-card-compact" if compact else "cell-card"
    host_line = f" &middot; {html.escape(str(cell['host']))}" if cell.get("show_host_in_label") else ""
    pytest_nid = cell.get("pytest_metrics_nodeid") or cell.get("pytest_inference_nodeid")
    pytest_link = ""
    if pytest_html_basename and pytest_nid:
        pytest_link = " &middot; " + pytest_row_link_html(pytest_html_basename, pytest_nid)

    return (
        f"<article class='{card_cls}'>"
        f"<header><div class='cell-title'>{html.escape(str(cell['policy']))}</div>"
        f"<div class='cell-sub'>ISL={cell['isl']} OSL={cell['osl']} &middot; C={cell['concurrency']}</div></header>"
        f"{mini_tl if not compact else ''}"
        f"<div class='headline'>{headline_val}<span class='headline-unit'>tok/s</span></div>"
        f"{headline_margin_html}"
        f"<div class='tiers'>{tier_chips}</div>"
        f"<div class='metrics'>{''.join(metric_rows)}</div>"
        f"<footer class='cell-foot'>{html.escape(cell['cell_id'])}{host_line}{pytest_link}</footer>"
        f"</article>"
    )
