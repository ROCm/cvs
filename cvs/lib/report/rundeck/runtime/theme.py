'''Run Deck theme CSS and shared panel fragments.'''

from __future__ import annotations

import html

from cvs.lib.report.formatting import status_badge_css
from cvs.lib.report.render.cell_card import cell_card_report_css
from cvs.lib.report.render.gate_matrix import gate_heatmap_css, gate_matrix_table_css
from cvs.lib.report.render.sweep_charts import chart_tooltip_css


def report_css() -> str:
    return (
        """
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
"""
        + status_badge_css()
        + """
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
.cmd-pre { margin: 0; padding: 0.85rem 1rem; border-radius: 8px; border: 1px solid var(--border);
  background: rgba(0,0,0,0.25); font-family: Consolas, "Cascadia Mono", monospace;
  font-size: 0.78rem; line-height: 1.4; white-space: pre-wrap; word-break: break-word; overflow-x: auto; }
.cmd-block { margin-bottom: 1rem; }
.cmd-block h3 { margin: 0 0 0.5rem; font-size: 0.75rem; text-transform: uppercase;
  letter-spacing: 0.06em; color: var(--muted); }
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
"""
        + chart_tooltip_css()
        + gate_matrix_table_css()
        + cell_card_report_css()
        + """
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
"""
        + gate_heatmap_css()
    )


def render_launch_panel_html(panel: dict) -> str:
    if not panel:
        return ""
    example = panel.get("example_cell") or ""
    example_note = (
        f"<p class='muted'>Representative commands for first sweep cell "
        f"<code>{html.escape(str(example))}</code>. Server env is written to "
        f"<code>/tmp/server_env_script.sh</code> at runtime.</p>"
        if example
        else "<p class='muted'>Representative launch commands from the variant config. "
        "Server env is written to <code>/tmp/server_env_script.sh</code> at runtime.</p>"
    )
    server_cmd = html.escape(str(panel.get("server_cmd") or ""))
    bench_cmd = html.escape(str(panel.get("bench_cmd") or ""))
    return (
        f"{example_note}"
        f"<div class='cmd-block'><h3>Server</h3><pre class='cmd-pre'>{server_cmd}</pre></div>"
        f"<div class='cmd-block'><h3>Benchmark client</h3><pre class='cmd-pre'>{bench_cmd}</pre></div>"
    )
