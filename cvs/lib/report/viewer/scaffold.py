'''Interactive viewer: filterable table, Chart.js concurrency charts, heatmap, gate matrix.'''

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Mapping, Optional

from cvs.lib.report.artifacts import export_payload

_TEMPLATE_PATH = Path(__file__).with_name("interactive.html")


def _embedded_json_script(payload: Mapping[str, Any]) -> str:
    """Inline JSON so the viewer works when opened via file:// (fetch is blocked)."""
    raw = json.dumps(export_payload(payload), separators=(",", ":"), default=str)
    # Keep serialized JSON from closing the surrounding <script> tag.
    safe = raw.replace("<", "\\u003c")
    return f'<script type="application/json" id="embedded-report-json">{safe}</script>'


def write_interactive_viewer(
    out_html: Path,
    *,
    json_basename: str,
    title: str,
    subtitle: str = "Interactive sweep explorer (loads sibling JSON sidecar)",
    tier_order: tuple[str, ...] = ("throughput", "record"),
    embed_payload: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write a static viewer HTML that loads report data embedded or via sibling JSON."""
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    if not template.strip():
        raise FileNotFoundError(f"Viewer template is empty or missing: {_TEMPLATE_PATH}")
    tier_js = "[" + ",".join(json.dumps(t) for t in tier_order) + "]"
    embedded = _embedded_json_script(embed_payload) if embed_payload is not None else ""
    doc = (
        template.replace("__TITLE__", html.escape(title))
        .replace("__SUBTITLE__", html.escape(subtitle))
        .replace("__JSON_PATH__", json.dumps(json_basename))
        .replace("__TIER_ORDER__", tier_js)
        .replace("__EMBEDDED_JSON__", embedded)
    )
    out_html.write_text(doc, encoding="utf-8")
    return out_html


def viewer_basename_for(report_basename: str) -> str:
    return f"{report_basename}_viewer.html"
