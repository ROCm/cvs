'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Dedicated interactive viewer for the JAX MaxText training Run Deck.

Unlike the inference viewer (concurrency/ISL sweeps), the training viewer plots
per-metric bar charts and per-tag line charts (all sweeps overlaid) DYNAMICALLY
with Chart.js from the embedded JSON payload -- the raw TensorBoard series
collected per sweep. No PNGs.
'''

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Mapping, Optional

from cvs.lib.report.artifacts import export_payload

_TEMPLATE_PATH = Path(__file__).with_name("training_viewer.html")


def _embedded_json_script(payload: Mapping[str, Any]) -> str:
    """Inline JSON so the viewer works via file:// (fetch is blocked there)."""
    raw = json.dumps(export_payload(payload), separators=(",", ":"), default=str)
    safe = raw.replace("<", "\\u003c")  # never close the surrounding <script>
    return f'<script type="application/json" id="embedded-report-json">{safe}</script>'


def write_training_viewer(out_html, *, title, subtitle, deck_basename, embed_payload=None):
    """Write the training viewer HTML with the payload embedded for offline use."""
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    if not template.strip():
        raise FileNotFoundError(f"Training viewer template is empty or missing: {_TEMPLATE_PATH}")
    embedded = _embedded_json_script(embed_payload) if embed_payload is not None else ""
    doc = (
        template.replace("__TITLE__", html.escape(title or "JAX MaxText Run Deck"))
        .replace("__SUBTITLE__", html.escape(subtitle or ""))
        .replace("__DECK_HREF__", html.escape(f"{deck_basename}.html"))
        .replace("__EMBEDDED_JSON__", embedded)
    )
    out_html.write_text(doc, encoding="utf-8")
    return out_html


def training_viewer_basename_for(report_basename: str) -> str:
    return f"{report_basename}_viewer.html"


__all__ = ["write_training_viewer", "training_viewer_basename_for"]
