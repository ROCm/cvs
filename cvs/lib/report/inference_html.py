'''Static HTML rendering for inference suite reports.'''

from __future__ import annotations

import html
from typing import List, Tuple

from cvs.lib.report.cell_build import select_summary_cells
from cvs.lib.report.formatting import fmt_num
from cvs.lib.report.formatting import link_or_text_html
from cvs.lib.report.formatting import status_badge_css
from cvs.lib.report.formatting import status_badge_html
from cvs.lib.report.render.cell_card import cell_card_report_css, render_cell_card_html
from cvs.lib.report.render.gate_matrix import (
    gate_heatmap_css,
    gate_matrix_table_css,
    render_gate_heatmap_html,
    render_gate_matrix_html,
)
from cvs.lib.report.render.panel_shell import render_panel_section, render_results_table_html

SESSION_FALLBACK = (
    "container_launch", "sshd_setup", "model_fetch", "server_ready", "client_complete", "teardown"
)


def _render_bar_chart(
    title: str,
    points: List[Tuple[int, float]],
    unit: str,
    *,
    invert: bool = False,
    accent: str = "accent",
) -> str:
    if len(points) < 2:
        return ""
    values = [p[1] for p in points]
    max_val = max(values) or 1.0
    min_val = min(values) or 0.0
    bars = []
    for conc, val in points:
        if invert:
            span = max_val - min_val or max_val or 1.0
            norm = (max_val - val) / span if span else 1.0
            h = max(12.0, 20.0 + 80.0 * norm)
        else:
            h = max(12.0, 100.0 * val / max_val)
        bars.append(
            f"<div class='chart-col'><div class='chart-bar chart-bar-{accent}' "
            f"style='height:{h:.0f}%'></div>"
            f"<div class='chart-x'>C={conc}</div>"
            f"<div class='chart-y'>{fmt_num(val)}</div></div>"
        )
    return (
        f"<div class='chart-panel'><h3>{html.escape(title)}</h3>"
        f"<div class='chart'>{''.join(bars)}</div>"
        f"<div class='chart-unit'>{html.escape(unit)}</div></div>"
    )


def report_css() -> str:
    return """
:root {
  --bg: #0f1117; --panel: #1a1d27; --border: #2a2f3d; --text: #e8eaef; --muted: #9aa3b5;
  --accent: #ff6b35; --accent2: #6b9fff; --accent3: #c77dff;
  --pass: #3dd68c; --fail: #ff5c6a; --record: #6b9fff; --na: #5c6370;
}
* { box-sizing: border-box; }
body { margin: 0; font-family: "Segoe UI", system-ui, sans-serif;
  background: linear-gradient(160deg, #0a0c12 0%, #12151f 40%, #0f1117 100%);
  color: var(--text); line-height: 1.45; padding: 1.5rem; }
.wrap { max-width: 1140px; margin: 0 auto; }
.hero-head { display: flex; flex-wrap: wrap; align-items: flex-start;
  justify-content: space-between; gap: 1rem; margin-bottom: 1.5rem; }
h1 { font-size: 1.75rem; font-weight: 600; margin: 0 0 0.25rem; letter-spacing: -0.02em; }
.subtitle { color: var(--muted); margin: 0; font-size: 0.95rem; }
""" + status_badge_css() + """
.panel { background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
  padding: 1.25rem 1.5rem; margin-bottom: 1.25rem; box-shadow: 0 8px 32px rgba(0,0,0,0.35); }
.panel h2 { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--muted); margin: 0 0 1rem; font-weight: 600; }
.meta-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 0.75rem 1rem; }
.meta-item { display: flex; flex-direction: column; gap: 0.15rem; }
.meta-k { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }
.meta-v { font-size: 0.9rem; font-weight: 500; word-break: break-word; }
.meta-v a { color: var(--accent2); text-decoration: none; }
.meta-v a:hover { text-decoration: underline; }
.notes { font-size: 0.85rem; color: var(--muted); margin-top: 0.75rem; }
.tl-row { display: flex; gap: 3px; min-height: 52px; border-radius: 8px; overflow: hidden; }
.tl-seg { background: linear-gradient(180deg, #2d3548 0%, #232836 100%);
  display: flex; flex-direction: column; justify-content: center; align-items: center;
  padding: 0.35rem; min-width: 48px; border-right: 1px solid var(--border); }
.tl-lbl { font-size: 0.65rem; color: var(--muted); text-align: center; }
.tl-val { font-size: 0.8rem; font-weight: 600; color: var(--accent); }
.summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin-bottom: 1rem; }
.summary-card { background: rgba(255,255,255,0.03); border: 1px solid var(--border); border-radius: 10px; padding: 1rem; }
.summary-card h3 { margin: 0 0 0.5rem; font-size: 0.95rem; }
.summary-stat { font-size: 1.5rem; font-weight: 700; color: var(--accent); }
.summary-meta { font-size: 0.8rem; color: var(--muted); margin-top: 0.35rem; }
.chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1rem; }
.chart-panel { background: rgba(255,255,255,0.02); border: 1px solid var(--border); border-radius: 10px; padding: 1rem; }
.chart-panel h3 { margin: 0 0 0.75rem; font-size: 0.75rem; text-transform: uppercase;
  letter-spacing: 0.06em; color: var(--muted); }
.chart { display: flex; align-items: flex-end; justify-content: center; gap: 1.25rem; height: 160px; padding-top: 0.5rem; }
.chart-col { display: flex; flex-direction: column; align-items: center; width: 64px; height: 100%; }
.chart-bar { width: 40px; border-radius: 6px 6px 2px 2px; margin-top: auto; }
.chart-bar-accent { background: linear-gradient(180deg, var(--accent) 0%, #c44d28 100%); }
.chart-bar-accent2 { background: linear-gradient(180deg, var(--accent2) 0%, #3d5a99 100%); }
.chart-bar-accent3 { background: linear-gradient(180deg, var(--accent3) 0%, #7a3db8 100%); }
.chart-x { font-size: 0.75rem; margin-top: 0.5rem; color: var(--muted); }
.chart-y { font-size: 0.7rem; font-weight: 600; margin-top: 0.2rem; }
.chart-unit { text-align: center; font-size: 0.7rem; color: var(--muted); margin-top: 0.35rem; }
""" + gate_matrix_table_css() + cell_card_report_css() + """
.cell-title { font-weight: 600; font-size: 1.05rem; }
.metric-row { margin-bottom: 0.65rem; }
.metric-label { font-size: 0.75rem; color: var(--muted); }
.metric-val { font-size: 1rem; font-weight: 600; }
.bar-track { height: 4px; background: var(--border); border-radius: 2px; margin-top: 0.35rem; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 2px; }
.bar-pass { background: var(--pass); } .bar-fail { background: var(--fail); } .bar-record { background: var(--record); }
.target { font-size: 0.7rem; color: var(--muted); margin-left: 0.25rem; }
.margin { font-size: 0.7rem; color: var(--pass); display: block; margin-top: 0.15rem; }
.margin-fail { color: var(--fail); }
.cell-foot { font-size: 0.65rem; color: var(--muted); margin-top: auto; padding-top: 0.5rem; border-top: 1px solid var(--border); }
.muted { color: var(--muted); }
footer.page-foot { text-align: center; color: var(--muted); font-size: 0.75rem; margin-top: 2rem; }
.report-nav { display: flex; flex-wrap: wrap; gap: 0.5rem 1rem; margin-bottom: 1.25rem; padding: 0.75rem 1rem;
  background: var(--panel); border: 1px solid var(--border); border-radius: 10px; font-size: 0.8rem; }
.report-nav a { color: var(--accent2); text-decoration: none; }
.report-nav a:hover { text-decoration: underline; }
.viewer-banner { margin-bottom: 1rem; padding: 0.75rem 1rem; border-radius: 8px;
  background: rgba(107,159,255,0.08); border: 1px solid var(--border); font-size: 0.9rem; }
.viewer-banner a { color: var(--accent2); }
@media (max-width: 640px) {
  body { padding: 1rem; }
  .hero-head { flex-direction: column; }
  .headline { font-size: 1.75rem; }
  .chart { gap: 0.75rem; height: 140px; }
  .chart-col { width: 52px; }
}
@media print {
  body { background: #fff; color: #111; padding: 0.5in; }
  .panel, .cell-card, .summary-card, .chart-panel {
    box-shadow: none; break-inside: avoid; background: #fff; border-color: #ccc;
  }
  .status-badge, .chip, .matrix-pass, .matrix-fail, .heat-pass, .heat-fail {
    print-color-adjust: exact; -webkit-print-color-adjust: exact;
  }
  a { color: #06c; }
}
""" + gate_heatmap_css()


def render_report_html(payload: dict) -> str:
    report = payload["report"]
    tier_order = report["metric_tier_order"]
    lifecycle = payload.get("lifecycle") or {}
    all_cells = payload.get("cells") or []
    summary = payload.get("summary") or {}
    cells = all_cells
    if summary.get("mode") == "truncated":
        cells = select_summary_cells(
            all_cells,
            int(summary.get("inline_limit", len(all_cells))),
            gated_tiers=tuple(summary.get("gated_tiers") or ()),
        )
    chart_series = payload.get("chart_series") or {}
    summaries = payload.get("sweep_summaries") or []
    gate_matrix = payload.get("gate_matrix") or []
    results_table = payload.get("results_table") or {}
    overall = payload.get("overall_status", "na")
    enforce = any(
        row[1] == "enforced"
        for row in payload.get("run_card_display", [])
        if row[0] == "Thresholds"
    )

    hero_html = "".join(
        f"<div class='meta-item'><span class='meta-k'>{html.escape(label)}</span>"
        f"<span class='meta-v'>"
        f"{link_or_text_html(value, label) if is_link else html.escape(str(value))}"
        f"</span></div>"
        for label, value, is_link in payload.get("run_card_display", [])
    )
    run_card_notes = payload.get("run_card_notes") or ""
    notes_html = (
        f"<p class='notes'>{html.escape(run_card_notes)}</p>" if run_card_notes else ""
    )

    timeline_total = sum(lifecycle.values()) or 1.0
    timeline_parts = []
    for lbl in report.get("session_lifecycle_labels", ()) or SESSION_FALLBACK:
        sec = lifecycle.get(lbl, 0.0)
        if sec <= 0:
            continue
        pct = 100.0 * sec / timeline_total
        timeline_parts.append(
            f"<div class='tl-seg' style='flex-grow:{pct:.2f}'>"
            f"<span class='tl-lbl'>{html.escape(lbl.replace('_', ' '))}</span>"
            f"<span class='tl-val'>{sec:.1f}s</span></div>"
        )
    timeline_html = "".join(timeline_parts) or "<p class='muted'>No lifecycle timings recorded.</p>"

    summary_html = "".join(
        f"<article class='summary-card'><h3>ISL={html.escape(str(s['isl']))} "
        f"\u00b7 OSL={html.escape(str(s['osl']))}</h3>"
        f"<div class='summary-stat'>{fmt_num(s['max_output_throughput'])} "
        f"<span class='headline-unit'>tok/s</span></div>"
        f"<div class='summary-meta'>Peak at C={s['conc_at_max_tput']}"
        f" &middot; TTFT {fmt_num(s.get('ttft_at_max_tput'))} ms"
        f"{' &middot; saturated at max C' if s.get('saturated') else ''}</div></article>"
        for s in summaries
    ) or "<p class='muted'>No sweep summary (no throughput data).</p>"

    chart_cfg = payload.get("chart_config") or []
    chart_accent = ("accent", "accent2", "accent3")
    chart_parts = []
    for idx, chart in enumerate(chart_cfg):
        points = chart_series.get(chart["suffix"], [])
        part = _render_bar_chart(
            chart["title"],
            points,
            chart["unit"],
            invert=bool(chart.get("invert")),
            accent=chart_accent[idx % 3],
        )
        if part:
            chart_parts.append(part)
    charts_html = (
        f"<div class='chart-grid'>{''.join(chart_parts)}</div>"
        if chart_parts
        else "<p class='muted'>Concurrency charts need two or more sweep cells.</p>"
    )

    matrix_html = render_gate_matrix_html(gate_matrix, tier_order)
    heatmap_html = render_gate_heatmap_html(gate_matrix, tier_order)
    heatmap_section = ""
    heatmap_nav = ""
    if heatmap_html:
        heatmap_section = f"<div id='heatmap' class='heatmap-section'>{heatmap_html}</div>"
        heatmap_nav = "<a href='#heatmap'>Heatmap</a>"

    rt_headers = results_table.get("headers") or []
    rt_rows = results_table.get("rows") or []
    results_html = render_results_table_html(
        rt_headers,
        rt_rows,
        empty_message="No results table rows.",
    )

    cell_lifecycle_labels = tuple(
        report.get("cell_lifecycle_labels") or ("server_ready", "client_complete")
    )
    pytest_basename = (payload.get("provenance") or {}).get("pytest_html_basename", "")
    cell_cards = [
        render_cell_card_html(
            c,
            tier_order=tuple(tier_order),
            headline_metric=report["headline_metric"],
            enforce=enforce,
            cell_lifecycle_labels=cell_lifecycle_labels,
            pytest_html_basename=pytest_basename or None,
        )
        for c in cells
    ]
    cells_banner = ""
    viewer_name = summary.get("viewer_html")
    if summary.get("mode") == "truncated" and viewer_name:
        cells_banner = (
            f"<div class='viewer-banner'>Showing {len(cells)} of {summary.get('total_cells', len(all_cells))} "
            f"cells in this summary. <a href='{html.escape(viewer_name)}'>Open interactive viewer</a> "
            f"for filter and search across all cells.</div>"
        )

    model_label = next((v for lbl, v, _ in payload.get("run_card_display", []) if lbl == "Model"), "run")

    from cvs.lib.report.panels.prev_run import render_prev_run_panel_html

    prev_run_panel = (payload.get("panels") or {}).get("prev_run")
    prev_run_html = ""
    prev_run_nav = ""
    if prev_run_panel:
        prev_run_html = render_panel_section(
            "Run vs baseline",
            render_prev_run_panel_html(prev_run_panel),
            section_id="prev-run",
        )
        prev_run_nav = "<a href='#prev-run'>Prev run</a>"

    viewer_nav = ""
    if viewer_name:
        viewer_nav = f"<a href='{html.escape(viewer_name)}'>Viewer</a>"

    nav = (
        "<nav class='report-nav'>"
        "<a href='#run-card'>Run card</a>"
        "<a href='#lifecycle'>Lifecycle</a>"
        "<a href='#sweep'>Sweep</a>"
        "<a href='#gates'>Gates</a>"
        f"{heatmap_nav}"
        f"{prev_run_nav}"
        "<a href='#cells'>Cells</a>"
        "<a href='#results'>Results</a>"
        f"{viewer_nav}"
        "</nav>"
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(report['title'])} &mdash; {html.escape(str(model_label))}</title>
<style>{report_css()}</style></head><body><div class="wrap">
<div class="hero-head"><div><h1>{html.escape(report['title'])}</h1>
<p class="subtitle">{html.escape(report['subtitle'])}</p></div>{status_badge_html(overall)}</div>
{nav}
<section class="panel" id="run-card"><h2>Run card</h2><div class="meta-grid">{hero_html}</div>{notes_html}</section>
<section class="panel" id="lifecycle"><h2>Lifecycle timeline</h2><div class="tl-row">{timeline_html}</div></section>
<section class="panel" id="sweep"><h2>Sweep analytics</h2><div class="summary-grid">{summary_html}</div>{charts_html}</section>
<section class="panel" id="gates"><h2>Gate matrix</h2><div class="matrix-wrap">{matrix_html}{heatmap_section}</div></section>
{prev_run_html}
<section class="panel" id="cells"><h2>Sweep cells</h2>{cells_banner}<div class="cells">{''.join(cell_cards) or '<p class="muted">No cells.</p>'}</div></section>
<section class="panel" id="results"><h2>Full results</h2><div class="results-wrap">{results_html}</div></section>
<footer class="page-foot">{html.escape(report['footer'])}</footer></div></body></html>"""
