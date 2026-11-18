'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

The AI Lab look shared by Run Decks, their viewers, and the consolidated report:
colour and font tokens, status marks, and the CVS header. Only the look changes;
deck and viewer layout and content stay the same.

Decks offer a light theme next to the default dark one. Viewers stay dark, because
their Chart.js charts are coloured in JavaScript for a dark background.
'''

import html

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
.ail-header { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 16px; min-height: 56px;
  padding: 8px 0; margin: 0 0 1.25rem; border-bottom: 1px solid var(--f-border); font-family: var(--f-body);
  color: var(--f-text); }
.ail-brand { display: inline-flex; align-items: center; gap: 10px; font: 500 18px var(--f-display); white-space: nowrap; }
.ail-brand strong { font-weight: 700; }
.ail-brand svg { width: 28px; height: 28px; flex: 0 0 28px; }
.ail-divider { width: 1px; height: 18px; background: var(--f-border); }
.ail-context { font-size: 12px; color: var(--f-text-3); }
.ail-back { font-size: 12px; font-weight: 500; color: var(--f-text-2); text-decoration: none; white-space: nowrap; }
.ail-back:hover { color: var(--f-text); text-decoration: underline; }
@media (max-width: 640px) {
  .ail-divider, .ail-context { display: none; }
  .ail-brand { font-size: 16px; }
}
"""

_DECK_CSS = """
.ail-theme { margin-left: auto; min-height: 32px; padding: 6px 10px; border: 1px solid var(--f-border);
  border-radius: 6px; background: transparent; color: var(--f-text); font: 500 12px var(--f-body); cursor: pointer; }
.ail-theme:hover { background: var(--f-surface-2); }
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
  .ail-header { display: none; }
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
    '''The AI Lab tokens, mapped onto the deck and viewer variables; with ``light``, also the light theme.'''
    css = ":root {%s%s}\n" % (_DARK_TOKENS, _PAGE_VARIABLES)
    if light:
        css += ":root[data-theme='light'] {%s}\n" % _LIGHT_TOKENS
    return css


def theme_css(page="deck"):
    '''Stylesheet that gives a deck (``page="deck"``) or a viewer (``page="viewer"``) the AI Lab look.

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
            '<button class="ail-theme" type="button" aria-label="Switch between light and dark theme" '
            'onclick="%s">Light / dark</button>' % _TOGGLE_JS
        )
    return (
        '<div class="ail-header" role="banner"><span class="ail-brand">%s<span><strong>AMD</strong> '
        'Cluster Validation Suite</span></span><span class="ail-divider" aria-hidden="true"></span>'
        '<span class="ail-context">%s</span>%s%s</div>'
        % (_BRAND_MARK, html.escape(context), OVERVIEW_LINK_MARKER, button)
    )
