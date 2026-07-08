'''Static HTML rendering for inference suite reports.'''

from __future__ import annotations

import html
from typing import List, Tuple

from cvs.lib.report.cell_build import select_summary_cells
from cvs.lib.report.formatting import (
    fmt_num,
    link_or_text_html,
    status_badge_css,
    status_badge_html,
)
from cvs.lib.report.render.cell_card import cell_card_report_css, render_cell_card_html
from cvs.lib.report.render.gate_matrix import (
    gate_heatmap_css,
    gate_matrix_table_css,
    render_gate_heatmap_html,
    render_gate_matrix_html,
)
from cvs.lib.report.render.panel_shell import render_results_table_html
from cvs.lib.report.inference_payload import sweep_has_multi_shape_comparison
from cvs.lib.report.render.sweep_charts import chart_tooltip_css
from cvs.lib.report.types import DEFAULT_SESSION_LIFECYCLE_LABELS

SESSION_FALLBACK = DEFAULT_SESSION_LIFECYCLE_LABELS


def _chart_group_keys(chart_cfg: list, chart_series: dict) -> List[Tuple[str, str, str]]:
    """Unique ISL/OSL groups that have chartable sweep data."""
    seen: set[Tuple[str, str]] = set()
    keys: List[Tuple[str, str, str]] = []
    for chart in chart_cfg:
        for entry in chart_series.get(chart["suffix"], []):
            if not isinstance(entry, dict):
                continue
            key = (str(entry["isl"]), str(entry["osl"]))
            if key in seen:
                continue
            seen.add(key)
            keys.append((entry["isl"], entry["osl"], entry.get("label") or f"ISL={key[0]} \u00b7 OSL={key[1]}"))
    return keys


def _chart_y_ticks(min_val: float, max_val: float, *, count: int = 5) -> List[float]:
    """Linear y-axis tick values from min to max (inclusive)."""
    if count < 2:
        return [max_val]
    span = max_val - min_val
    if span <= 0:
        return [max_val]
    return [min_val + span * i / (count - 1) for i in range(count)]


def _chart_display_scale(min_val: float, max_val: float) -> Tuple[float, float, List[float]]:
    """Y-axis domain and ticks; pad flat series so the scale is readable."""
    if max_val > min_val:
        domain_min, domain_max = min_val, max_val
    else:
        pad = max(abs(max_val) * 0.08, 1.0)
        domain_min, domain_max = max_val - pad, max_val + pad
    ticks = _chart_y_ticks(domain_min, domain_max)
    return domain_min, domain_max, ticks


def _chart_value_pct(val: float, domain_min: float, domain_max: float) -> float:
    span = domain_max - domain_min or 1.0
    return max(0.0, min(100.0, 100.0 * (val - domain_min) / span))


def _bar_height_pct(val: float, min_val: float, max_val: float) -> float:
    domain_min, domain_max, _ = _chart_display_scale(min_val, max_val)
    return max(8.0, _chart_value_pct(val, domain_min, domain_max))


def _render_bar_chart(
    title: str,
    points: List[Tuple[int, float]],
    unit: str,
    *,
    accent: str = "accent",
) -> str:
    if len(points) < 2:
        return ""
    values = [p[1] for p in points]
    max_val = max(values) or 1.0
    min_val = min(values) or 0.0
    domain_min, domain_max, ticks = _chart_display_scale(min_val, max_val)
    y_labels = "".join(
        f"<span class='chart-ylbl' style='bottom:{_chart_value_pct(t, domain_min, domain_max):.2f}%'>"
        f"{html.escape(fmt_num(t))}</span>"
        for t in ticks
    )
    grid = "".join(
        f"<span class='chart-hline' style='bottom:{_chart_value_pct(t, domain_min, domain_max):.2f}%'></span>"
        for t in ticks
    )
    x_labels = []
    bars = []
    for conc, val in points:
        h = _bar_height_pct(val, min_val, max_val)
        tip = html.escape(f"C={conc}: {fmt_num(val)} {unit}".strip())
        bars.append(
            f"<div class='chart-col'>"
            f"<div class='chart-bar chart-bar-{accent} chart-has-tip' style='height:{h:.1f}%' "
            f"data-tip='{tip}' tabindex='0' role='img' aria-label='{tip}'></div></div>"
        )
        x_labels.append(f"<span class='chart-xlbl'>C={conc}</span>")
    return (
        f"<div class='chart-panel'><h3>{html.escape(title)}</h3>"
        f"<div class='chart-viz'>"
        f"<div class='chart-ywrap'><div class='chart-ylabels'>{y_labels}</div></div>"
        f"<div class='chart-main'>"
        f"<div class='chart-plotbox'><div class='chart-hgrid' aria-hidden='true'>{grid}</div>"
        f"<div class='chart-bars'>{''.join(bars)}</div></div>"
        f"<div class='chart-xrow'>{''.join(x_labels)}</div></div></div>"
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
.chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; }
.chart-panel { background: rgba(255,255,255,0.02); border: 1px solid var(--border); border-radius: 10px; padding: 1rem; }
.chart-panel h3 { margin: 0 0 0.75rem; font-size: 0.75rem; text-transform: uppercase;
  letter-spacing: 0.06em; color: var(--muted); }
.chart-viz { display: flex; align-items: stretch; gap: 0.35rem; }
.chart-ywrap { flex: 0 0 auto; padding-bottom: 1.35rem; }
.chart-ylabels { position: relative; width: 3.1rem; height: 148px; }
.chart-ylbl { position: absolute; right: 0.25rem; transform: translateY(50%);
  font-size: 0.62rem; color: var(--muted); white-space: nowrap; line-height: 1; }
.chart-main { flex: 1; min-width: 0; }
.chart-plotbox { position: relative; height: 148px; border-left: 1px solid rgba(154, 163, 181, 0.55);
  border-bottom: 1px solid rgba(154, 163, 181, 0.55); background: rgba(0, 0, 0, 0.12); }
.chart-hgrid { position: absolute; inset: 0; pointer-events: none; z-index: 0; }
.chart-hline { position: absolute; left: 0; right: 0; height: 0;
  border-top: 1px solid rgba(42, 47, 61, 0.95); }
.chart-bars { position: absolute; inset: 0; z-index: 1; display: flex; align-items: flex-end;
  justify-content: space-around; padding: 0 0.35rem; gap: 0.35rem; }
.chart-col { flex: 1 1 0; max-width: 52px; height: 100%; display: flex; align-items: flex-end;
  justify-content: center; border-left: 1px solid rgba(42, 47, 61, 0.55); }
.chart-col:first-child { border-left: none; }
.chart-bar { width: 100%; max-width: 40px; min-height: 3px; border-radius: 4px 4px 0 0; }
.chart-bar-accent { background: linear-gradient(180deg, var(--accent) 0%, #c44d28 100%); }
.chart-bar-accent2 { background: linear-gradient(180deg, var(--accent2) 0%, #3d5a99 100%); }
.chart-bar-accent3 { background: linear-gradient(180deg, var(--accent3) 0%, #7a3db8 100%); }
.chart-xrow { display: flex; justify-content: space-around; gap: 0.35rem; padding: 0.4rem 0.35rem 0; }
.chart-xlbl { flex: 1 1 0; max-width: 52px; text-align: center; font-size: 0.72rem; color: var(--muted); }
.chart-unit { text-align: center; font-size: 0.7rem; color: var(--muted); margin-top: 0.45rem; }
.chart-sweep-hint { margin: 0 0 0.75rem; font-size: 0.75rem; color: var(--muted); }
.chart-group { margin-bottom: 1.25rem; }
.chart-group:last-child { margin-bottom: 0; }
.chart-group-title { margin: 0 0 0.75rem; font-size: 0.95rem; font-weight: 600; color: var(--text); }
""" + chart_tooltip_css() + gate_matrix_table_css() + cell_card_report_css() + """
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
  .chart-ylabels { width: 2.6rem; height: 132px; }
  .chart-plotbox { height: 132px; }
  .chart-ywrap { padding-bottom: 1.2rem; }
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
    group_keys = _chart_group_keys(chart_cfg, chart_series)
    chart_sections: List[str] = []
    for isl, osl, label in group_keys:
        chart_parts = []
        for idx, chart in enumerate(chart_cfg):
            entry = next(
                (
                    e
                    for e in chart_series.get(chart["suffix"], [])
                    if isinstance(e, dict) and str(e.get("isl")) == str(isl) and str(e.get("osl")) == str(osl)
                ),
                None,
            )
            if not entry:
                continue
            part = _render_bar_chart(
                chart["title"],
                entry["points"],
                chart["unit"],
                accent=chart_accent[idx % 3],
            )
            if part:
                chart_parts.append(part)
        if not chart_parts:
            continue
        title_html = (
            f"<h3 class='chart-group-title'>{html.escape(label)}</h3>"
            if len(group_keys) > 1
            else ""
        )
        chart_sections.append(
            f"<div class='chart-group'>{title_html}<div class='chart-grid'>{''.join(chart_parts)}</div></div>"
        )
    charts_html = (
        "".join(chart_sections)
        if chart_sections
        else "<p class='muted'>Concurrency charts need two or more points per sweep shape.</p>"
    )

    sweep_chart_hint = ""
    if chart_sections:
        sweep_chart_hint = (
            "<p class='chart-sweep-hint'>Per-shape bars use a y/x grid; hover a bar for the exact value.</p>"
        )

    sweep_viewer_banner = ""
    viewer_name = summary.get("viewer_html")
    if sweep_has_multi_shape_comparison(cells) and viewer_name:
        sweep_viewer_banner = (
            "<div class='viewer-banner'>Cross-shape comparison (grouped bars and scaling trends) "
            f"is in the <a href='{html.escape(viewer_name)}'>interactive viewer</a>.</div>"
        )
    elif sweep_has_multi_shape_comparison(cells):
        sweep_viewer_banner = (
            "<div class='viewer-banner'>Cross-shape comparison charts are available in the "
            "interactive viewer sidecar.</div>"
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
    pytest_basename = (payload.get("provenance") or {}).get("pytest_html_href") or (
        (payload.get("provenance") or {}).get("pytest_html_basename", "")
    )
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
    if summary.get("mode") == "truncated" and viewer_name:
        cells_banner = (
            f"<div class='viewer-banner'>Showing {len(cells)} of {summary.get('total_cells', len(all_cells))} "
            f"cells in this summary. <a href='{html.escape(viewer_name)}'>Open interactive viewer</a> "
            f"for filter and search across all cells.</div>"
        )

    model_label = next((v for lbl, v, _ in payload.get("run_card_display", []) if lbl == "Model"), "run")

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
<section class="panel" id="sweep"><h2>Sweep analytics</h2><div class="summary-grid">{summary_html}</div>{sweep_viewer_banner}{sweep_chart_hint}{charts_html}</section>
<section class="panel" id="gates"><h2>Gate matrix</h2><div class="matrix-wrap">{matrix_html}{heatmap_section}</div></section>
<section class="panel" id="cells"><h2>Sweep cells</h2>{cells_banner}<div class="cells">{''.join(cell_cards) or '<p class="muted">No cells.</p>'}</div></section>
<section class="panel" id="results"><h2>Full results</h2><div class="results-wrap">{results_html}</div></section>
<footer class="page-foot">{html.escape(report['footer'])}</footer></div></body></html>"""
