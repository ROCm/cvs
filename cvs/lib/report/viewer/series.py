'''Series Run Deck viewer (RCCL and other ``series`` dataset profiles).

Written by ``generate_rundeck.py`` when a ``series`` profile sets ``interactive_viewer``.
Separate from the inference explorer in ``interactive.html``; same basename and
embedded-JSON loading pattern as ``status_matrix.py``.
'''

import html
import json
from pathlib import Path

from cvs.lib.report.artifacts import export_payload
from cvs.lib.report.rundeck.render import render_series_viewer_body
from cvs.lib.report.viewer.paging import page_tables_js

_TEMPLATE_PATH = Path(__file__).with_name("series.html")


def _embedded_json_script(payload):
    """Inline JSON so the viewer works when opened via file:// (fetch is blocked)."""
    raw = json.dumps(export_payload(payload), separators=(",", ":"), default=str)
    safe = raw.replace("<", "\\u003c")
    return f'<script type="application/json" id="embedded-report-json">{safe}</script>'


def write_series_viewer(out_html, *, json_basename, title, subtitle="", embed_payload=None):
    """Write the series viewer HTML that loads report data embedded or via sibling JSON."""
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    if not template.strip():
        raise FileNotFoundError(f"Viewer template is empty or missing: {_TEMPLATE_PATH}")
    embedded = _embedded_json_script(embed_payload) if embed_payload is not None else ""
    deck_html = ""
    deck_nav = ""
    if embed_payload is not None:
        deck_html, deck_nav = render_series_viewer_body(embed_payload)
    subtitle = subtitle or "Interactive series explorer (loads sibling JSON sidecar)"
    doc = (
        template.replace("__TITLE__", html.escape(title))
        .replace("__SUBTITLE__", html.escape(subtitle))
        .replace("__JSON_PATH__", json.dumps(json_basename))
        .replace("__DECK_NAV__", deck_nav)
        .replace("__DECK_SECTIONS__", deck_html)
        .replace("__EMBEDDED_JSON__", embedded)
        .replace("__PAGE_TABLES_JS__", page_tables_js())
    )
    out_html.write_text(doc, encoding="utf-8")
    return out_html
