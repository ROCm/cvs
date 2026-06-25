'''Shared HTML formatting helpers for suite reports.'''

from __future__ import annotations

import html
import re
from typing import Any


def fmt_num(value: Any, digits: int = 1) -> str:
    if value is None:
        return "\u2014"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if abs(f) >= 1000:
        return f"{f:,.{digits}f}"
    return f"{f:.{digits}f}"


def status_badge_html(status: str) -> str:
    labels = {"pass": "PASS", "fail": "FAIL", "record": "RECORD", "na": "N/A"}
    return (
        f'<span class="status-badge status-{html.escape(status)}">'
        f"{html.escape(labels.get(status, status.upper()))}</span>"
    )


def link_or_text_html(url: str, label: str) -> str:
    if not url:
        return "\u2014"
    safe_url = html.escape(url)
    safe_label = html.escape(label)
    if re.match(r"^https?://", url):
        return f'<a href="{safe_url}" target="_blank" rel="noopener">{safe_label}</a>'
    return f"<span>{safe_label}</span>"
