'''Interactive viewer: filterable table, Chart.js concurrency charts, heatmap, gate matrix.'''

from __future__ import annotations

import html
import json
from pathlib import Path

_TEMPLATE_PATH = Path(__file__).with_name("interactive.html")


def write_interactive_viewer(
    out_html: Path,
    *,
    json_basename: str,
    title: str,
    subtitle: str = "Interactive sweep explorer (loads sibling JSON sidecar)",
    tier_order: tuple[str, ...] = ("throughput", "record"),
) -> Path:
    """Write a static viewer HTML that fetches ``json_basename`` from the same directory."""
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    tier_js = "[" + ",".join(json.dumps(t) for t in tier_order) + "]"
    doc = (
        template.replace("__TITLE__", html.escape(title))
        .replace("__SUBTITLE__", html.escape(subtitle))
        .replace("__JSON_PATH__", json.dumps(json_basename))
        .replace("__TIER_ORDER__", tier_js)
    )
    out_html.write_text(doc, encoding="utf-8")
    return out_html


def viewer_basename_for(report_basename: str) -> str:
    return f"{report_basename}_viewer.html"
