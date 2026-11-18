'''Run Deck theme and shared panel fragments.

The theme gives Run Decks, their viewers, and the consolidated report one look,
following the AI Lab design: colour and font tokens, status marks, and the CVS
header. Only the look changes; deck and viewer layout and content stay the same.
Decks offer a light theme next to the default dark one. Viewers stay dark, because
their Chart.js charts are coloured in JavaScript for a dark background.
'''

from __future__ import annotations

import html

from cvs.lib.report.formatting import status_badge_css
from cvs.lib.report.render.cell_card import CellCardConfig, CellCardRenderer
from cvs.lib.report.render.gate_matrix import gate_heatmap_css, gate_matrix_table_css
from cvs.lib.report.render.loss_chart import loss_chart_css
from cvs.lib.report.render.sweep_charts import chart_tooltip_css

THEME_STORAGE_KEY = "cvs-theme"
OVERVIEW_LINK_MARKER = "<!--cvs-overview-link-->"

_DARK_TOKENS = """
  --f-display: 'Space Grotesk', 'Segoe UI', system-ui, Arial, sans-serif;
  --f-body: Inter, 'Segoe UI', system-ui, Arial, sans-serif;
  --f-mono: 'IBM Plex Mono', Consolas, Menlo, monospace;
  --f-canvas: #0A0A0B; --f-surface-1: #121214; --f-surface-2: #1A1A1D; --f-surface-3: #232327;
  --f-border: #2E2E33; --f-border-strong: #3A3A41; --f-text: #F5F5F7; --f-text-2: #ADADB5; --f-text-3: #8A8A93;
  --f-brand-red: #ED1C24; --f-action-red: #C1121C; --f-ok: #00C39A; --f-warn: #FFB020; --f-crit: #FF453A;
  --f-focus-ring: #F5F5F7; --f-series-1: #1AA0F9;
  color-scheme: dark;
"""

_LIGHT_TOKENS = """
  --f-canvas: #FAFAFA; --f-surface-1: #FFFFFF; --f-surface-2: #F2F2F4; --f-surface-3: #E8E8EC;
  --f-border: #DCDCE2; --f-border-strong: #C8C8D0; --f-text: #0A0A0B; --f-text-2: #46464F; --f-text-3: #63636D;
  --f-brand-red: #C1121C; --f-ok: #00795F; --f-warn: #8A5A00; --f-crit: #B3261E; --f-focus-ring: #0A0A0B;
  --f-series-1: #0667F9;
  color-scheme: light;
"""

# The deck and viewer stylesheets read these names; they must stay in step with those stylesheets.
_PAGE_VARIABLES = """
  --bg: var(--f-canvas); --panel: var(--f-surface-1); --panel2: var(--f-surface-2); --border: var(--f-border);
  --text: var(--f-text); --muted: var(--f-text-3); --accent: var(--f-text); --accent2: var(--f-series-1);
  --accent3: var(--f-text-2); --pass: var(--f-ok); --fail: var(--f-crit); --record: var(--f-text-2);
  --na: var(--f-text-3);
"""

_BASE_CSS = """
body { background: var(--f-canvas); color: var(--f-text); font-family: var(--f-body); padding-top: 0; }
h1, h2, h3 { font-family: var(--f-display); }
code, pre, kbd, samp { font-family: var(--f-mono); }
:focus-visible { outline: 2px solid var(--f-focus-ring); outline-offset: 2px; }
.cvs-header { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 16px; min-height: 56px;
  padding: 8px 0; margin: 0 0 1.25rem; border-bottom: 1px solid var(--f-border); font-family: var(--f-body);
  color: var(--f-text); }
.cvs-brand { display: inline-flex; align-items: center; gap: 10px; font: 500 18px var(--f-display); white-space: nowrap; }
.cvs-brand strong { font-weight: 700; }
.cvs-brand svg { width: 28px; height: 28px; flex: 0 0 28px; }
.cvs-divider { width: 1px; height: 18px; background: var(--f-border); }
.cvs-context { font-size: 12px; color: var(--f-text-3); }
.cvs-back { font-size: 12px; font-weight: 500; color: var(--f-text-2); text-decoration: none; white-space: nowrap; }
.cvs-back:hover { color: var(--f-text); text-decoration: underline; }
@media (max-width: 640px) {
  .cvs-divider, .cvs-context { display: none; }
  .cvs-brand { font-size: 16px; }
}
"""

_DECK_CSS = """
.cvs-theme-toggle { margin-left: auto; min-height: 32px; padding: 6px 10px; border: 1px solid var(--f-border);
  border-radius: 6px; background: transparent; color: var(--f-text); font: 500 12px var(--f-body); cursor: pointer; }
.cvs-theme-toggle:hover { background: var(--f-surface-2); }
h1 { font-weight: 500; letter-spacing: -0.035em; }
.subtitle, .notes, .muted, .summary-meta { color: var(--f-text-3); }
.panel { border-radius: 8px; box-shadow: none; }
.panel h2, .cmd-block h3, .chart-panel h3, .meta-k, .tl-group-head .tl-lbl {
  font-family: var(--f-mono); letter-spacing: 0.1em; color: var(--f-text-3); }
.cmd-pre, .sm-iname { font-family: var(--f-mono); }
.cmd-pre, .sm-body { background: var(--f-surface-2); }
.tl-seg, .tl-group { background: var(--f-surface-2); }
.tl-cell, .summary-card, .chart-panel { background: var(--f-surface-1); }
.summary-stat, .tl-val { color: var(--f-text); font-family: var(--f-display); font-weight: 500; }
.summary-stat-pass { color: var(--f-ok); }
.summary-stat-fail { color: var(--f-crit); }
.summary-stat-na { color: var(--f-text-3); }
.chart-plotbox { background: transparent; border-color: var(--f-border-strong); }
.chart-hline { border-top-color: var(--f-border); }
.chart-has-tip::before { background: var(--f-surface-3); color: var(--f-text); box-shadow: none; }
.chart-bar-accent { background: var(--f-series-1); }
.chart-bar-accent2 { background: var(--f-text-2); }
.chart-bar-accent3 { background: var(--f-text-3); }
.chart-bar-pass { background: var(--f-ok); }
.chart-bar-fail { background: var(--f-crit); }
.to-top { background: var(--f-surface-2); color: var(--f-series-1); }
.rd-pager button { background: var(--f-surface-2); color: var(--f-text); }
.report-nav { border-radius: 8px; }
.viewer-banner { background: var(--f-surface-2); }
.matrix th, .results-table th, .status-matrix th, .overview-table th, .results-scroll .results-table th {
  background: var(--f-surface-2); color: var(--f-text-3); font-family: var(--f-mono); }
.sm-details > summary:hover { background: var(--f-surface-2); }
.chip { border-radius: 4px; font-family: var(--f-mono); }
"""

_STATUS_MARKS_CSS = """
.status-badge { display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px; border-radius: 4px;
  border: 1px solid var(--f-border-strong); background: transparent; font: 600 12px/1.4 var(--f-mono);
  letter-spacing: 0.1em; }
.status-badge::before { content: ""; flex: 0 0 8px; width: 8px; height: 8px; }
.status-badge.status-pass { color: var(--f-ok); }
.status-badge.status-pass::before { background: var(--f-ok); border-radius: 50%; }
.status-badge.status-fail { color: var(--f-crit); }
.status-badge.status-fail::before { background: var(--f-crit); transform: rotate(45deg); }
.status-badge.status-record { color: var(--f-text-2); }
.status-badge.status-record::before { border: 1px solid var(--f-text-2); border-radius: 50%; }
.status-badge.status-na { color: var(--f-text-3); }
.status-badge.status-na::before { height: 2px; background: var(--f-text-3); }
"""

_PRINT_CSS = """
@media print {
  :root { --f-canvas: #fff; --f-surface-1: #fff; --f-surface-2: #f4f4f6; --f-surface-3: #eee; --f-border: #ccc;
    --f-border-strong: #bbb; --f-text: #111; --f-text-2: #333; --f-text-3: #555; color-scheme: light; }
  .cvs-header { display: none; }
}
"""

THEME_BOOT_SCRIPT = (
    "<script>(function(){var t;try{t=localStorage.getItem('%s')}catch(e){}"
    "if(t!=='light'&&t!=='dark'){t=window.matchMedia&&matchMedia('(prefers-color-scheme: light)').matches"
    "?'light':'dark'}document.documentElement.dataset.theme=t}())</script>" % THEME_STORAGE_KEY
)

_TOGGLE_JS = (
    "var r=document.documentElement,t=r.dataset.theme==='light'?'dark':'light';"
    "r.dataset.theme=t;try{localStorage.setItem('%s',t)}catch(e){}" % THEME_STORAGE_KEY
)

_BRAND_MARK = (
    '<svg viewBox="0 0 120 120" fill="none" aria-hidden="true"><g stroke="currentColor" stroke-width="7">'
    '<path d="M60 10V110M10 60H110"/><path d="M32 32L88 88M88 32L32 88" opacity=".45"/></g>'
    '<circle cx="60" cy="60" r="16" fill="var(--f-canvas)"/><circle cx="60" cy="60" r="10" fill="var(--f-brand-red)"/>'
    "</svg>"
)


def tokens_css(light=True):
    '''The colour and font tokens, mapped onto the deck and viewer variables; with ``light``, also the light theme.'''
    css = ":root {%s%s}\n" % (_DARK_TOKENS, _PAGE_VARIABLES)
    if light:
        css += ":root[data-theme='light'] {%s}\n" % _LIGHT_TOKENS
    return css


def theme_css(page="deck"):
    '''Stylesheet that themes a deck (``page="deck"``) or a viewer (``page="viewer"``).

    It must come after the page's own stylesheet, so that its rules win.
    '''
    deck = page == "deck"
    return tokens_css(light=deck) + _BASE_CSS + _STATUS_MARKS_CSS + (_DECK_CSS if deck else "") + _PRINT_CSS


def viewer_theme_head():
    '''The ``<style>`` element that goes at the end of a viewer's ``<head>``.'''
    return "<style>%s</style>" % theme_css("viewer")


def header_html(context, toggle=False):
    '''The CVS header: brand, page context, the slot for the consolidated report's back-link, and a theme toggle.'''
    button = ""
    if toggle:
        button = (
            '<button class="cvs-theme-toggle" type="button" aria-label="Switch between light and dark theme" '
            'onclick="%s">Light / dark</button>' % _TOGGLE_JS
        )
    return (
        '<div class="cvs-header" role="banner"><span class="cvs-brand">%s<span><strong>AMD</strong> '
        'Cluster Validation Suite</span></span><span class="cvs-divider" aria-hidden="true"></span>'
        '<span class="cvs-context">%s</span>%s%s</div>'
        % (_BRAND_MARK, html.escape(context), OVERVIEW_LINK_MARKER, button)
    )


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
.tl-seg, .tl-group { background: linear-gradient(180deg, #2d3548 0%, #232836 100%);
  border-right: 1px solid var(--border); }
.tl-seg, .tl-cell { display: flex; flex-direction: column; justify-content: center;
  align-items: center; padding: 0.35rem; overflow: hidden; }
.tl-seg { min-width: 48px; }
.tl-cell { min-width: 64px; background: rgba(255, 255, 255, 0.03); }
.tl-group { display: flex; flex-direction: column; min-width: 150px; }
.tl-group-head { display: flex; justify-content: center; align-items: baseline; gap: 0.4rem;
  padding: 0.4rem 0.35rem; border-bottom: 1px solid var(--border); }
.tl-group-body { display: flex; flex: 1; gap: 3px; }
.tl-lbl { font-size: 0.65rem; color: var(--muted); text-align: center; overflow-wrap: anywhere; }
.tl-group-head .tl-lbl { text-transform: uppercase; letter-spacing: 0.05em; }
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
.chart-bars-dense { gap: 2px; padding: 0 0.15rem; justify-content: stretch; }
.chart-bars-dense .chart-col { border-left: none; max-width: none; }
.chart-viz-dense .chart-ywrap { padding-bottom: 0; }
.chart-col { flex: 1 1 0; min-width: 0; max-width: 52px; height: 100%; display: flex; align-items: flex-end;
  justify-content: center; border-left: 1px solid rgba(42, 47, 61, 0.55); }
.chart-col:first-child { border-left: none; }
.chart-panel-wide { grid-column: 1 / -1; }
.chart-cluster { gap: 3px; }
.chart-cluster .chart-bar { flex: 1 1 0; width: auto; max-width: 18px; min-width: 4px; }
.chart-xrow-cluster .chart-xlbl { max-width: none; }
.series-legend { display: flex; flex-wrap: wrap; gap: 0.55rem 0.8rem; margin: -0.15rem 0 0.35rem;
  font-size: 0.72rem; color: var(--muted); }
.series-swatch { display: inline-block; width: 0.65rem; height: 0.65rem; margin-right: 0.3rem;
  border-radius: 2px; vertical-align: -1px; }
.chart-bar { width: 100%; max-width: 40px; min-height: 3px; border-radius: 4px 4px 0 0; }
.chart-bar-accent { background: linear-gradient(180deg, var(--accent) 0%, #c44d28 100%); }
.chart-bar-accent2 { background: linear-gradient(180deg, var(--accent2) 0%, #3d5a99 100%); }
.chart-bar-accent3 { background: linear-gradient(180deg, var(--accent3) 0%, #7a3db8 100%); }
.chart-bar-pass { background: linear-gradient(180deg, var(--pass) 0%, #1f8a58 100%); }
.chart-bar-fail { background: linear-gradient(180deg, var(--fail) 0%, #a32d3a 100%); }
.chart-bar-na { background: var(--na); }
.chart-threshold { position: absolute; left: 0; right: 0; height: 0; border-top: 1.5px dashed var(--accent);
  z-index: 2; pointer-events: none; }
.metric-note { font-size: 0.75rem; color: var(--muted); margin: -0.35rem 0 0.65rem; }
.chart-xrow { display: flex; justify-content: space-around; gap: 0.35rem; padding: 0.4rem 0.35rem 0; }
.chart-xlbl { flex: 1 1 0; min-width: 0; max-width: 52px; text-align: center; font-size: 0.68rem;
  color: var(--muted); line-height: 1.2; }
.chart-xlbl-line { display: block; overflow-wrap: anywhere; }
.chart-unit { text-align: center; font-size: 0.7rem; color: var(--muted); margin-top: 0.45rem; }
.chart-sweep-hint { margin: 0 0 0.75rem; font-size: 0.75rem; color: var(--muted); }
.chart-group { margin-bottom: 1.25rem; }
.chart-group:last-child { margin-bottom: 0; }
.chart-group-title { margin: 0 0 0.75rem; font-size: 0.95rem; font-weight: 600; color: var(--text); }
"""
        + chart_tooltip_css()
        + loss_chart_css()
        + gate_matrix_table_css()
        + CellCardRenderer(CellCardConfig(theme="report")).get_css()
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
.to-top { display: block; width: max-content; margin: 0.85rem auto 0; padding: 0.3rem 0.75rem;
  border: 1px solid var(--border); border-radius: 6px; background: #12151f; color: var(--accent2);
  text-decoration: none; font-size: 0.8rem; }
.to-top:hover { border-color: var(--accent); }
.report-nav { display: flex; flex-wrap: wrap; gap: 0.5rem 1rem; margin-bottom: 1.25rem; padding: 0.75rem 1rem;
  background: var(--panel); border: 1px solid var(--border); border-radius: 10px; font-size: 0.8rem; }
.report-nav a { color: var(--accent2); text-decoration: none; }
.report-nav a:hover { text-decoration: underline; }
.viewer-banner { margin-bottom: 1rem; padding: 0.75rem 1rem; border-radius: 8px;
  background: rgba(107,159,255,0.08); border: 1px solid var(--border); font-size: 0.9rem; }
.viewer-banner a { color: var(--accent2); }
.status-matrix { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.status-matrix th, .status-matrix td { border: 1px solid var(--border); text-align: center; vertical-align: top; padding: 0; }
.status-matrix th { background: rgba(255,255,255,0.04); color: var(--muted); font-size: 0.7rem;
  text-transform: uppercase; padding: 0.5rem 0.65rem; }
.status-matrix td.sm-node, .status-matrix th.sm-node { text-align: left; font-weight: 500;
  padding: 0.6rem 0.7rem; white-space: nowrap; }
.status-matrix td.sm-pass { background: rgba(61,214,140,0.08); }
.status-matrix td.sm-fail { background: rgba(255,92,106,0.10); }
.status-matrix td.sm-na { color: var(--na); padding: 0.6rem; }
.sm-details > summary { list-style: none; cursor: pointer; padding: 0.55rem 0.5rem;
  display: flex; flex-direction: column; align-items: center; gap: 4px; }
.sm-details > summary::-webkit-details-marker { display: none; }
.sm-details > summary:hover { background: rgba(255,255,255,0.03); }
.sm-count { font-size: 0.68rem; color: var(--muted); }
.sm-caret::after { content: "\\25bc items"; font-size: 0.6rem; color: var(--muted); }
.sm-details[open] > summary .sm-caret::after { content: "\\25b2 hide"; }
.sm-body { padding: 0.5rem 0.6rem 0.7rem; border-top: 1px dashed var(--border);
  background: rgba(0,0,0,0.2); text-align: left; }
.sm-summary { font-size: 0.72rem; margin-bottom: 0.4rem; }
.sm-item { display: flex; justify-content: space-between; align-items: center; gap: 0.75rem;
  font-size: 0.75rem; padding: 0.2rem 0; border-bottom: 1px solid rgba(255,255,255,0.04); }
.sm-item:last-of-type { border-bottom: none; }
.sm-iname { font-family: Consolas, "Cascadia Mono", monospace; }
.sm-imsg { color: var(--muted); font-size: 0.7rem; }
.sm-links { margin-top: 0.5rem; display: flex; gap: 0.75rem; }
.sm-links a { color: var(--accent2); text-decoration: none; font-size: 0.72rem; }
.sm-links a:hover { text-decoration: underline; }
.sm-hint { font-size: 0.78rem; margin-top: 0.75rem; }
.rd-pager { display: flex; align-items: center; justify-content: flex-end; gap: 0.75rem;
  margin-top: 0.75rem; font-size: 0.8rem; color: var(--muted); }
.rd-pager button { background: #12151f; color: var(--text); border: 1px solid var(--border);
  border-radius: 6px; padding: 0.3rem 0.7rem; cursor: pointer; }
.rd-pager button:hover { border-color: var(--accent); }
.overview-split { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1rem; }
.overview-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
.overview-table th, .overview-table td { text-align: left; padding: 0.35rem 0.4rem;
  border-bottom: 1px solid var(--border); }
.overview-table th { color: var(--muted); font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.04em; }
.ov-fail { white-space: nowrap; }
.fail-track { display: inline-block; width: 4.5rem; height: 6px; margin-left: 0.4rem; background: var(--border);
  border-radius: 3px; vertical-align: middle; overflow: hidden; }
.fail-fill { display: block; height: 100%; background: var(--fail); }
.ov-stats { grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); }
.summary-stat-pass { color: var(--pass); }
.summary-stat-fail { color: var(--fail); }
.summary-stat-na { color: var(--na); }
.ov-stack { display: flex; height: 6px; border-radius: 3px; overflow: hidden; background: var(--border); margin-top: 0.5rem; }
.ov-stack span { display: block; height: 100%; }
.ov-stack-pass { background: var(--pass); }
.ov-stack-fail { background: var(--fail); }
.ov-stack-na { background: var(--na); }
.chart-grid-heat { grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); }
.chart-grid-heat .chart-panel { min-width: 0; }
.hm-wrap { overflow-x: auto; }
.hm-table { border-collapse: collapse; font-size: 0.72rem; }
.hm-table th, .hm-table td { border: 1px solid var(--border); padding: 0.25rem 0.4rem; text-align: center; }
.hm-table th { color: var(--muted); font-weight: 500; }
.hm-pass { background: rgba(61,214,140,0.35); }
.hm-fail { background: rgba(255,92,106,0.35); }
.hm-na { color: var(--na); }
.results-table th .th-unit { text-transform: none; }
.results-table td.al-left { text-align: left; }
.results-table td.al-right { text-align: right; }
.results-scroll .results-table th { position: sticky; top: 0; z-index: 1;
  background: linear-gradient(rgba(255,255,255,0.04), rgba(255,255,255,0.04)), var(--panel); }
@media (max-width: 640px) {
  body { padding: 1rem; }
  .hero-head { flex-direction: column; }
  .headline { font-size: 1.75rem; }
  .chart-ylabels { width: 2.6rem; height: 132px; }
  .chart-plotbox { height: 132px; }
  .chart-ywrap { padding-bottom: 1.2rem; }
  .overview-split { grid-template-columns: 1fr; }
  .fail-track { width: 2.5rem; }
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
  .to-top { display: none; }
}
"""
        + gate_heatmap_css()
        + theme_css("deck")
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
